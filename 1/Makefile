TVM_ROOT=$(shell cd ..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I/usr/local/include/opencv4\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build 
PKG_LDFLAGS += -ltvm_runtime
PKG_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
PKG_LDFLAGS += -ldl -pthread

.PHONY: clean all

all:
	$(CXX) $(PKG_CFLAGS) -o test deploy.cpp $(PKG_LDFLAGS)

clean:
	rm -f test
