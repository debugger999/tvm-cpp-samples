#include <stdio.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

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

static int NhwcToNchw(unsigned char *src, unsigned char *dst, int c, int w, int h) {
    int n = 0;
    for(int i = 0; i < h; i ++) {
        for(int j = 0; j < w; j ++) {
            for(int k = 0; k < c; k ++) {
                int m = w*h*k + w*i + j;
                dst[m] = src[n++];
            }
        }
    }
    return 0;
}

static int ReadImg(const char *filename, int c, int w, int h, float **buf) {
    Mat dst;
    Mat image = imread(filename);
    if(image.empty()) {
        printf("fread %s failed\n", filename);
        return -1;
    }
    resize(image, image, Size(w, h));
    cvtColor(image, dst, COLOR_BGR2RGB);
    int size = dst.rows*dst.cols;
    unsigned char *rgb = (unsigned char *)malloc(size*c);
    NhwcToNchw(dst.data, rgb, c, w, h);
    //WriteFile("test_rgb_resize.rgb", rgb, size*c, "wb");
    float v;
    float mean[3] = {0.485, 0.456, 0.406};
    float stddev[3] = {0.229, 0.224, 0.225};
    float *ptr = (float *)malloc(size*c*4);
    int offset = 0;
    for(int i = 0; i < c; i ++) {
        for(int j = 0; j < size; j ++) {
            v = rgb[offset]/255.0;
            v = (v - mean[i])/stddev[i];
            ptr[offset] = v;
            offset += 1;
        }
    }
    *buf = ptr;
    return 0;
}

static int CalMax(float *buf, int len, float &val, int &index, float max) {
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

int main(int argc, char *argv[]) {
    DLTensor* input;
    DLTensor* output;
    int device_type = kDLCUDA;
    int device_id = 0;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    DLDevice dev = {(DLDeviceType)device_type, device_id};

    std::string lib = "libresnet50.so";
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib);
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    int ndim = 4;
    int c = 3, w = 224, h = 224;
    int64_t shape[4] = {1, c, w, h};
    DLDataType data_type = {(uint8_t)dtype_code, (uint8_t)dtype_bits, (uint16_t)dtype_lanes};
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
    float *buf_ = NULL;
    size_t size = tvm::runtime::GetDataSize(*input);
    ReadImg("imagenet_cat.png", c, w, h, &buf_);
    TVMArrayCopyFromBytes(input, buf_, size);
    free(buf_);
    printf("input size:%ld\nshape:\n", size);
    for(int i = 0; i < input->ndim; i ++) {
        printf("%ld ", input->shape[i]);
    }
    printf("\n");
    set_input("data", input);
    run();
    ndim = 2;
    int64_t out_shape[2] = {1, 1000};
    TVMArrayAlloc(out_shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output);
    get_output(0, output);
    size = tvm::runtime::GetDataSize(*output);
    printf("output size:%ld\nshape:\n", size);
    for(int i = 0; i < output->ndim; i ++) {
        printf("%ld ", output->shape[i]);
    }
    printf("\n");
    float *buf = (float *)malloc(size);
    TVMArrayCopyToBytes(output, buf, size);
    float val;
    int index;
    printf("result:\n");
    CalMax(buf, 1000, val, index, 100);
    printf("index:%d,val:%f\n", index, val);
    CalMax(buf, 1000, val, index, 14.198225);
    printf("index:%d,val:%f\n", index, val);
    CalMax(buf, 1000, val, index, 14.075565);
    printf("index:%d,val:%f\n", index, val);
    CalMax(buf, 1000, val, index, 11.287909);
    printf("index:%d,val:%f\n", index, val);
    CalMax(buf, 1000, val, index, 8.276128);
    printf("index:%d,val:%f\n", index, val);

    TVMArrayFree(input);
    TVMArrayFree(output);
    free(buf);
    printf("run %s ok\n", lib.c_str());

    return 0;
}

