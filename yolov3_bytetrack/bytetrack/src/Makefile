CPP = g++
AR:=ar
CFLAGS = -std=c++11
CFLAGS += -Wall -O2 
CFLAGS += -Wno-deprecated-declarations 
CFLAGS += -Wno-sign-compare -Wno-reorder
CFLAGS += -Wno-unused-result
CFLAGS += -fPIC
#CFLAGS += -g

CFLAGS += -I../inc
CFLAGS += -I /usr/local/include/opencv4

LDFLAGS =
LDFLAGS += -llzma -ldl -lz -lm -lpthread -lrt

all:
	$(CPP) $(CFLAGS) -c *.cpp
	$(CPP) -shared -o ../lib/libbytetrack.so *.o
clean:
	rm -f *.o ../lib/*

