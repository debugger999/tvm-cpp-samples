#include <stdio.h>
#include <unistd.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "darknet.h"

using namespace cv;

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

static int get_detections(detection *dets, int nboxes, 
        float thresh, char **names, int classes, image *frame) {
    for(int i = 0; i < nboxes; ++i){
        int cls_id = -1;
        float score = 0.0;
        for(int j = 0; j < classes; ++j){
            if(dets[i].prob[j] > thresh && dets[i].prob[j] > score){
                cls_id = j;
                score = dets[i].prob[j];
            }
        }
        if(cls_id >= 0){
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*frame->w;
            int right = (b.x+b.w/2.)*frame->w;
            int top   = (b.y-b.h/2.)*frame->h;
            int bot   = (b.y+b.h/2.)*frame->h;
            if(left < 0) left = 0;
            if(right > frame->w-1) right = frame->w-1;
            if(top < 0) top = 0;
            if(bot > frame->h-1) bot = frame->h-1;
            printf("class:%d,name:%s,score:%f,left:%d,top:%d,right:%d,bot:%d\n", 
                    cls_id, names[cls_id], score, left, top, right, bot);
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
        out.CopyToBytes(layer_attr,  size);
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
    float thresh = 0.2;
    float nms_thresh = 0.45;
    int yolo_layer_num = 3;
    network *net = (network *)calloc(1, sizeof(network));
    net->n = yolo_layer_num;
    net->w = w;
    net->h = h;
    net->layers = (layer *)calloc(net->n, sizeof(layer));
    while(1) {
        printf("cnt:%d\n", cnt);
        image frame = get_image_from_stream(&capture);
        if(!frame.data) {
            printf("read end, %s\n", video_file);
            break;
        }
        image frame_s = letterbox_image(frame, w, h);
        TVMArrayCopyFromBytes(input, frame_s.data, size);
        set_input("data", input);
        run();
        get_out(net, yolo_layer_num, get_output);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, width, height, thresh, 0.5, 0, 1, &nboxes);
        do_nms_sort(dets, nboxes, classes, nms_thresh);
        printf("nboxes:%d\n", nboxes);
        get_detections(dets, nboxes, thresh, names, classes, &frame);
        cnt ++; 
        //if(cnt >= 3)break;
        free_detections(dets, nboxes);
        free_image(frame_s);
        free_image(frame);
        usleep(10000);
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

