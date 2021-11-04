#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/videoio_c.h>
#include "darknet.h"
#include "BYTETracker.h"

//#define DISPLAY

using namespace cv;
using namespace std;

/*
struct timeval t0, t1, t2, t3;
printf("ms cost,%ld,%ld,%ld\n", 
        (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000,
        (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000,
        (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000
        );
*/
int WriteFile(const char *filename, void *buf, int size, const char *mode) {
    FILE *fp = fopen(filename, mode);
    if(fp == NULL) {
        printf("fopen %s failed\n", filename);
        return -1;
    }
    if(buf != NULL) {
        fwrite(buf, 1, size, fp);
    }
    fclose(fp);
    return 0;
}

int CalMax(float *buf, int len, float &val, int &index, float max) {
    index = 0;
    val = -99999.0;
    for (int i = 0; i < len; ++i) {
        if(buf[i] > val && buf[i] < max) {
            val = buf[i];
            index = i;
        }
    }
    return 0;
}

static int get_detections(detection *dets, int nboxes, float thresh, int classes, image *frame, vector<Object>& objects) {
    for(int i = 0; i < nboxes; ++i) {
        int classid = -1;
        float score = 0.0;
        for(int j = 0; j < classes; ++j){
            if(dets[i].prob[j] > thresh && dets[i].prob[j] > score){
                classid = j;
                score = dets[i].prob[j];
            }
        }
        if(classid >= 0){
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*frame->w;
            int right = (b.x+b.w/2.)*frame->w;
            int top   = (b.y-b.h/2.)*frame->h;
            int bottom   = (b.y+b.h/2.)*frame->h;
            if(left < 0) left = 0;
            if(right > frame->w-1) right = frame->w-1;
            if(top < 0) top = 0;
            if(bottom > frame->h-1) bottom = frame->h-1;
            int w = right - left;
            int h = bottom - top;
            if(w > frame->w/2 || h > frame->h/2 || w*h < 20) {
                continue;
            }
            Object det;
            det.rect.x = left;
            det.rect.y = top;
            det.rect.width = w;
            det.rect.height = h;
            det.label = classid;
            det.prob = score;
            objects.push_back(det);
        }
    }
    
    return 0;
}

static int get_out(network *net, int layer_num, tvm::runtime::PackedFunc get_output) {
    int size;
    const DLTensor* dptr;
    tvm::runtime::NDArray out;
    for(int i = 0; i < layer_num; i ++) {
        layer *layer = net->layers + i;
        layer->type = YOLO;

        out = get_output(i * 4 + 3);
        dptr = out.operator->();
        size = tvm::runtime::GetDataSize(*dptr);
        int* layer_attr = (int* )malloc(size);
        //if(i == 0)gettimeofday(&t1, NULL);
        out.CopyToBytes(layer_attr,  size);
        //if(i == 0)gettimeofday(&t2, NULL);
        layer->batch = 1;
        layer->n = layer_attr[0];
        layer->c = layer_attr[1];
        layer->h = layer_attr[2];
        layer->w = layer_attr[3];
        layer->classes = layer_attr[4];
        free(layer_attr);

        out = get_output(i * 4 + 2);
        dptr = out.operator->();
        size = tvm::runtime::GetDataSize(*dptr);
        if(layer->biases == NULL) {
            layer->biases = (float*)malloc(size);
        }
        out.CopyToBytes(layer->biases,  size);

        out = get_output(i * 4 + 1);
        dptr = out.operator->();
        size = tvm::runtime::GetDataSize(*dptr);
        if(layer->mask == NULL) {
            layer->mask = (int*)malloc(size);
        }
        out.CopyToBytes(layer->mask,  size);

        out = get_output(i * 4 + 0);
        dptr = out.operator->();
        size = tvm::runtime::GetDataSize(*dptr);
        if(layer->output == NULL) {
            layer->output = (float*)malloc(size);
        }
        out.CopyToBytes(layer->output, size);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    DLTensor* input;
    int device_type = kDLCUDA;
    //int device_type = kDLCPU;
    int device_id = 0;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int ndim = 4;
    int c = 3, w = 416, h = 416;
    int64_t shape[4] = {1, c, w, h};
    DLDevice dev = {(DLDeviceType)device_type, device_id};

    if(argc < 2) {
        printf("usage: ./deploy video\n");
        return 0;
    }
    // init detector
    std::string lib = "libyolov3.so";
    char **names = get_labels((char *)"coco.names");
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib);
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    char *video_file = argv[1];
    VideoCapture capture(video_file);
    if (!capture.isOpened()) {
        printf("open %s failed\n", video_file);
        return -1;
    }
    int width = capture.get(CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int fps = capture.get(CV_CAP_PROP_FPS);
    printf("open %s success, %dx%d\n", video_file, width, height);

    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
    size_t size = tvm::runtime::GetDataSize(*input);
    printf("input size:%ld, shape:\n", size);
    for(int i = 0; i < input->ndim; i ++) {
        printf("%ld ", input->shape[i]);
    }
    printf("\n");

    int cnt = 0;
    int classes = 8;
    //float thresh = 0.2;
    float thresh = 0.5;
    float nms_thresh = 0.45;
    int yolo_layer_num = 3;
    network *net = (network *)calloc(1, sizeof(network));
    net->n = yolo_layer_num;
    net->w = w;
    net->h = h;
    net->layers = (layer *)calloc(net->n, sizeof(layer));

    // init bytetrack
    BYTETracker tracker(fps, 30);

#ifdef DISPLAY
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 1);
    namedWindow("test", WINDOW_NORMAL);
#endif
    Mat img;
    while(1) {
        //gettimeofday(&t0, NULL);
        capture >> img;
        if(img.empty()) {
            printf("read end, %s\n", video_file);
            break;
        }
        image frame = mat_to_image_ex(&img);
        image frame_s = letterbox_image(frame, w, h);
        TVMArrayCopyFromBytes(input, frame_s.data, size);
        set_input("data", input);
        run();
        get_out(net, yolo_layer_num, get_output);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, width, height, thresh, 0.5, 0, 1, &nboxes);
        do_nms_sort(dets, nboxes, classes, nms_thresh);
        vector<Object> objects;
        get_detections(dets, nboxes, thresh, classes, &frame, objects);
        //gettimeofday(&t3, NULL);
        vector<STrack> output_stracks = tracker.update(objects);
        printf("cnt:%d, obj:%ld\n", cnt, output_stracks.size());
#ifdef DISPLAY
        CvMat _img = cvMat(img);
        for(unsigned int i = 0; i < output_stracks.size(); i++) {
            int track_id = output_stracks[i].track_id;
            int label = output_stracks[i].label;
            vector<float> tlwh = output_stracks[i].tlwh;
            cvRectangle(&_img, cvPoint(tlwh[0], tlwh[1]), 
                    cvPoint(tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]), cvScalar(0,255,0));
            cvPutText(&_img, format("%d:%s", track_id, names[label]).c_str(), 
                    cvPoint(tlwh[0], tlwh[1]), &font, cvScalar(0,0,255));
        }
        imshow("test", img);
        waitKey(10);
#endif
        free_detections(dets, nboxes);
        free_image(frame_s);
        free_image(frame);
        cnt ++; 
    }

    for(int i = 0; i < yolo_layer_num; i ++) {
        layer *layer = net->layers + i;
        if(layer->biases != NULL) {
            free(layer->biases);
        }
        if(layer->mask != NULL) {
            free(layer->mask);
        }
        if(layer->output != NULL) {
            free(layer->output);
        }
    }
    free(net->layers);
    free(net);
    TVMArrayFree(input);
    free(names);
    printf("run %s ok\n", lib.c_str());

    return 0;
}

