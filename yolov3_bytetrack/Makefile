DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -Wall -O2 -fPIC\
	-I ${TVM_HOME}/include\
	-I ${DMLC_CORE}/include\
	-I ${TVM_HOME}/3rdparty/dlpack/include\
	-I /usr/local/include/opencv4\
	-I darknet/inc\
	-I bytetrack/inc\
	-DOPENCV\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L ${TVM_HOME}/build 
PKG_LDFLAGS += -L darknet/lib
PKG_LDFLAGS += -L bytetrack/lib
PKG_LDFLAGS += -ltvm_runtime
PKG_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio
PKG_LDFLAGS += -ldarknet
PKG_LDFLAGS += -lbytetrack
PKG_LDFLAGS += -ldl -pthread
PKG_LDFLAGS += -fPIE -Wl,-rpath,darknet/lib,-rpath,bytetrack/lib

.PHONY: clean all

all:
	$(CXX) $(PKG_CFLAGS) -o deploy deploy.cpp $(PKG_LDFLAGS)

clean:
	rm -f deploy
