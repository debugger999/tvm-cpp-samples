import sys
import os
import time
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from scipy.special import softmax
from tvm.relay.testing.darknet import __darknetffi__
import matplotlib.pyplot as plt
import cv2

target = "llvm"

if len(sys.argv) < 5:
  print("usage: python3 build.py weights cfg libdarknet img")
  exit()
weights_path = sys.argv[1]
cfg_path = sys.argv[2]
lib_darknet = sys.argv[3]
img_path = sys.argv[4]

c = 3
h = 416
w = 416
batch_size = 1
dtype = "float32"
input_name = "data"
dylib_path = os.path.join("./libyolov3.so")

if not os.path.exists(dylib_path):
  print("load", lib_darknet, weights_path, cfg_path, "...")
  DARKNET_LIB = __darknetffi__.dlopen(lib_darknet)
  net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
  print("load ok,", net.c, net.h, net.w)
  data = np.empty([batch_size, net.c, net.h, net.w], dtype)
  print("converting to relay functions ...")
  mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
  print("convert ok")
  print("compiling the model", dylib_path, "...")
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)
      lib.export_library(dylib_path)
  print("compile ok")

print("load", dylib_path, "...")
loaded_lib = tvm.runtime.load_module(dylib_path)
print("load ok")
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(loaded_lib["default"](dev))
print("run ok")

print("resize", img_path, "to", w, "x", h, ", channel:", c)
_shape,img_data = tvm.relay.testing.darknet.load_image(img_path, w, h)
img_data = tvm.nd.array(img_data.astype(dtype))
print("resize ok")
module.set_input(input_name, img_data)
print("graph run ...")
module.run()
print("run ok")

tvm_out = []
for i in range(3):
    layer_out = {}
    layer_out["type"] = "Yolo"
    # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
    layer_attr =module.get_output(i * 4 + 3).numpy()
    layer_out["biases"] =module.get_output(i * 4 + 2).numpy()
    layer_out["mask"] =module.get_output(i * 4 + 1).numpy()
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
    layer_out["output"] =module.get_output(i * 4).numpy().reshape(out_shape)
    layer_out["classes"] = layer_attr[4]
    tvm_out.append(layer_out)
#print(len(tvm_out), tvm_out)

thresh = 0.5
nms_thresh = 0.45
classes = 80
# do the detection and bring up the bounding boxes
im_h, im_w, _ = _shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
    (w, h), (im_w, im_h), thresh, 1, tvm_out
)
tvm.relay.testing.yolo_detection.do_nms_sort(dets, classes, nms_thresh)

coco_path = "coco.names"
with open(coco_path) as f:
    content = f.readlines()
names = [x.strip() for x in content]
for det in dets:
    valid, detection = tvm.relay.testing.yolo_detection.get_detections_ex(_shape, det, thresh, names, classes)
    if valid:
        print(
            "class:{} score:{} left:{} top:{} right:{} bottom:{}".format(
                detection["classname"],
                detection["score"],
                detection["left"],
                detection["top"],
                detection["right"],
                detection["bot"],
            )
        )

'''
font_path = "arial.ttf"
img = tvm.relay.testing.darknet.load_image_color(img_path)
tvm.relay.testing.yolo_detection.show_detections(img, dets, thresh, names, classes)
tvm.relay.testing.yolo_detection.draw_detections(
    font_path, img, dets, thresh, names, classes
)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
'''

