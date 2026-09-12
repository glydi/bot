#!/bin/sh
# Fetch the object-detection model sense-vision uses for the `object`
# modality into $GLYDI_MODELS_DIR/vision (default ../../models/vision from
# this crate, i.e. <repo>/models/vision).
#
# Preference order, first one that downloads wins:
#   1. yolov8n.onnx  (ultralytics YOLOv8 nano; no official ONNX asset is
#      published, so the mirrors below may 404 -- that is expected)
#   2. yolov5n.onnx  (ultralytics YOLOv5 v7.0 release asset; official,
#      GPL-3.0, 3.9 MB, input [1,3,640,640], output [1,25200,85])
#
# The Rust side (src/objects.rs) reads the output shape from the model
# metadata and handles both layouts, so either file works unchanged.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dir=${GLYDI_MODELS_DIR:-"$here/../../../models"}/vision
mkdir -p "$dir"
try() { # $1 = file, $2 = url
  if [ -s "$dir/$1" ]; then echo "already have $dir/$1"; exit 0; fi
  echo "trying $2"
  if curl -fsSL --max-time 120 -o "$dir/$1.part" "$2" && [ "$(head -c 1 "$dir/$1.part" | od -An -tx1 | tr -d ' ')" = "08" ]; then
    mv "$dir/$1.part" "$dir/$1"; echo "saved $dir/$1"; exit 0
  fi
  rm -f "$dir/$1.part"
}
try yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
try yolov8n.onnx https://huggingface.co/Kalray/yolov8n/resolve/main/yolov8n.onnx
try yolov5n.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
echo "no YOLO model could be downloaded" >&2; exit 1
