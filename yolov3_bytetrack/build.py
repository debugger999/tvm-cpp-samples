import sys
import os
import time
import onnx
import argparse
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
from yolox.tracker.byte_tracker import BYTETracker

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "yolov3.weights",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "yolov3.cfg",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "libdarknet2.0.so",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "palace.mp4",
        type=str,
        default="",
        help="",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

if len(sys.argv) < 5:
  print("usage: python3 build.py weights cfg libdarknet video")
  exit()
weights_path = sys.argv[1]
cfg_path = sys.argv[2]
lib_darknet = sys.argv[3]
img_path = sys.argv[4]
video_path = sys.argv[4]

c = 3
h = 416
w = 416
batch_size = 1
dtype = "float32"
input_name = "data"
dylib_path = os.path.join("./libyolov3.so")

target = "cuda"
print("target:", target)
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

fps = 30
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("resize", img_path, width, "x", height, "to", w, "x", h, ", channel:", c)
vid_writer = cv2.VideoWriter(
    "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
)

cnt = 0
args = make_parser().parse_args()
tracker = BYTETracker(args, frame_rate=fps)
while True:
  ret_val, img = cap.read()
  if not ret_val:
      print("read file end")
      break
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_data = tvm.relay.testing.darknet.load_image_ex(img_rgb, w, h)
  img_data = tvm.nd.array(img_data.astype(dtype))
  module.set_input(input_name, img_data)
  module.run()
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
  thresh = 0.2
  nms_thresh = 0.45
  classes = 8
  # do the detection and bring up the bounding boxes
  im_h, im_w, _ = img_rgb.shape
  dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
      (w, h), (im_w, im_h), thresh, 1, tvm_out
  )
  tvm.relay.testing.yolo_detection.do_nms_sort(dets, classes, nms_thresh)
  coco_path = "coco.names"
  with open(coco_path) as f:
      content = f.readlines()
  names = [x.strip() for x in content]
  print("##", cnt)
  final_dets = []
  for det in dets:
      detection = tvm.relay.testing.yolo_detection.get_detections_ex(img_rgb.shape, det, thresh, classes)
      if len(detection) > 0:
          cls_id = detection["cls_id"]
          name = names[cls_id]
          if name != 'person' and name != 'bicycle' and name != 'car' and \
                  name != 'motorbike' and name != 'bus' and name != 'truck':
              continue
          score = detection["score"]
          x1 = detection["left"]
          y1 = detection["top"]
          x2 = detection["right"]
          y2 = detection["bot"]
          _det = np.array([x1, y1, x2, y2, score, cls_id])
          final_dets.append(_det)
          #print("detection,", cls_id, name, score, x1, y1, x2, y2)
  if len(final_dets) == 0:
      continue
  outputs = np.concatenate(final_dets)
  outputs = outputs.reshape(len(final_dets), 6)
  online_targets = tracker.update(outputs, [height, width], [height, width])
  online_tlwhs = []
  online_ids = []
  online_scores = []
  for t in online_targets:
      tlwh = t.tlwh
      tid = t.track_id
      cls_id = int(t.cls_id)
      vertical = tlwh[2] / tlwh[3] > 1.6
      if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
          online_tlwhs.append(tlwh)
          online_ids.append(tid)
          online_scores.append(t.score)
          id_text = '{}'.format(int(tid))
          class_name = names[cls_id]
          x1 = int(tlwh[0])
          y1 = int(tlwh[1])
          x2 = int(tlwh[0] + tlwh[2])
          y2 = int(tlwh[1] + tlwh[3])
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
          cv2.putText(img, id_text + ":" + class_name, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
  vid_writer.write(img)
  #cv2.imshow("test", img)
  #cv2.waitKey(1)
  cnt += 1

