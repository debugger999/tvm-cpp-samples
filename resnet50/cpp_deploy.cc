/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>

#include <cstdio>

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  ICHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y);
  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    printf("%f:%f\n", static_cast<float*>(y->data)[i], i + 1.0f);
    ICHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
  }
  LOG(INFO) << "Finish verification...";
  TVMArrayFree(x);
  TVMArrayFree(y);
}

void DeploySingleOp() {
  // Normally we can directly
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("lib/test_addone_dll.so");
  LOG(INFO) << "Verify dynamic loading from test_addone_dll.so";
  Verify(mod_dylib, "addone");
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  LOG(INFO) << "Verify load function from system lib";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  Verify(mod_syslib, "addonesys");
}

void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/test_relay_add.so");
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      static_cast<float*>(x->data)[i * 2 + j] = i * 2 + j;
    }
  }
  // set the right input
  set_input("x", x);
  // run the code
  run();
  // get the output
  get_output(0, y);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      printf("%f:%d\n", static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
      ICHECK_EQ(static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
    }
  }
  LOG(INFO) << "Running graph executor ok";
}

void mem_test(){
    DLTensor* x;
    DLTensor* y;
    int ndim = 3;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_id = 0;
    int64_t shape[3] = {3,1920,1080};
    int device_type;
    device_type = kDLCUDA;
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    device_type = kDLCPU;
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
    printf("##test, line:%d\n", __LINE__);
    int len = 10;
    for (int i = 0; i < len; ++i) {
        static_cast<float*>(y->data)[i] = i;
    }
    printf("##test, line:%d\n", __LINE__);
    for (int i = 0; i < len; ++i) {
        printf("y[%d]:%f\n", i, ((float*)(y->data))[i]);
    }
    size_t size = tvm::runtime::GetDataSize(*y);
    printf("##test, line:%d, size:%ld\n", __LINE__, size);
    if(TVMArrayCopyFromBytes(x, y->data, size) != 0) {
        printf("##test, line:%d, err\n", __LINE__);
    }
    printf("##test, line:%d, copy ok\n", __LINE__);
    for (int i = 0; i < len; ++i) {
        ((float*)(y->data))[i] = 0;
    }
    for (int i = 0; i < len; ++i) {
        printf("y[%d]:%f\n", i, ((float*)(y->data))[i]);
    }
    printf("##test, line:%d\n", __LINE__);
    if(TVMArrayCopyToBytes(x, y->data, size) != 0) {
        printf("##test, line:%d, err\n", __LINE__);
    }
    for (int i = 0; i < len; ++i) {
        printf("y[%d]:%f\n", i, ((float*)(y->data))[i]);
    }
    printf("##test, line:%d\n", __LINE__);
    TVMArrayFree(x);
    TVMArrayFree(y);
    printf("run %s test ok\n", __func__);
}

void test_cuda_mem() {
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("lib/test_add_mem.so");
    LOG(INFO) << "Verify dynamic loading from test_add_mem.so";
    // Get the function from the module.
    tvm::runtime::PackedFunc f = mod.GetFunction("myadd");
    ICHECK(f != nullptr);
    DLTensor* a;
    DLTensor* b;
    DLTensor* c;
    int ndim = 1;
    int len = 1024;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCUDA;
    int device_id = 0;
    int64_t shape[1] = {len};
    DLDataType data_type = {(uint8_t)dtype_code, (uint8_t)dtype_bits, (uint16_t)dtype_lanes};
    DLDevice _dev = {(DLDeviceType)device_type, device_id};
    auto _shape = std::vector<int64_t>(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
      _shape[i] = shape[i];
    }

    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &a);
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &b);
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &c);

    size_t size = tvm::runtime::GetDataSize(*a);
    float *buf = (float *)malloc(size);
    for (int i = 0; i < len; ++i) {
        buf[i] = i;
    }
    TVMArrayCopyFromBytes(a, buf, size);
    for (int i = 0; i < len; ++i) {
        buf[i] = 2*i;
    }
    TVMArrayCopyFromBytes(b, buf, size);
    tvm::runtime::NDArray::Container* container = new tvm::runtime::NDArray::Container(a->data, _shape, data_type, _dev);
    DLTensor dl_tensor = container->dl_tensor;

    f(&dl_tensor, b, c);

    TVMArrayCopyToBytes(&dl_tensor, buf, size);
    for (int i = 0; i < 10; ++i) {
        printf("%f ", buf[i]);
    }
    printf("\n");
    TVMArrayCopyToBytes(b, buf, size);
    for (int i = 0; i < 10; ++i) {
        printf("%f ", buf[i]);
    }
    printf("\n");
    TVMArrayCopyToBytes(c, buf, size);
    for (int i = 0; i < 10; ++i) {
        printf("%f ", buf[i]);
    }
    printf("\n");
    TVMArrayFree(a);
    TVMArrayFree(b);
    TVMArrayFree(c);
    delete container;
    printf("run %s test ok\n", __func__);
}

int main(void) {
  //DeploySingleOp();
  //DeployGraphExecutor();
  //mem_test();
  test_cuda_mem();
  return 0;
}
