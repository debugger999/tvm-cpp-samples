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

target = "cuda"

if len(sys.argv) < 4:
  print("usage: python3 build.py model lable img")
  exit()
model_path = sys.argv[1]
labels_path = sys.argv[2]
img_path = sys.argv[3]


c = 3
w = 224
h = 224
input_name = "data"
dylib_path = os.path.join("./", "libresnet50.so")
if not os.path.exists(dylib_path):
  print("load " + model_path + " ...")
  onnx_model = onnx.load(model_path)
  print("load ok")

  # Same with netron viewer:
  # inputs:
  #   id:data
  #   type:float32{1,3,224,224}
  shape_dict = {input_name: (1, c, w, h)}
  print("get mod/params from relay.frontend ...")
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
  print("get mod/params ok")

  print("relay.build ...")
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)
      lib.export_library(dylib_path)
  print("relay.build " + dylib_path + " ok")

print("resize " + img_path + " to", w, "x", h)
resized_image = Image.open(img_path).resize((w, h))
resized_image = np.transpose(resized_image, (2, 0, 1))
img_data = np.asarray(resized_image).astype("float32")
# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

print("load " + dylib_path + " ...")
loaded_lib = tvm.runtime.load_module(dylib_path)
print("load ok")

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input(input_name, img_data)
print("run ...")
module.run()
print("run ok")
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("rank='%d',class='%s',   probability=%f" % (rank, labels[rank], scores[rank]))













'''
#mnist test
print("load " + model_path + " ...")
onnx_model = onnx.load(model_path)
print("load ok")
print(type(onnx_model))
shape = {"Input3": (1, 1, 28, 28)}
mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
module = graph_executor.GraphModule(lib["default"](dev))
digit_2 = Image.open("digit-2.jpg").resize((28, 28))
digit_2 = np.asarray(digit_2).astype("float32")
digit_2 = np.expand_dims(digit_2, axis=0)
module.set_input("Input3", tvm.nd.array(digit_2))
print("run ...")
module.run()
print("run ok")
output_shape = (1, 10)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
print(tvm_output)
'''


