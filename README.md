# ONNX Object Detection Runtime

## ONNX Environment

```bash
$ conda create -y -n onnx-inference python==3.10 pip
$ pip install opencv-python numpy onnxruntime supervision click pandas
```

Run inference with the CLI

```bash
python cli.py inference \
    --image-folder ./data \
    --model-path ./work_dirs/faster_rcnn.onnx \
    --output-dir ./output \
    --batch-size 4
```