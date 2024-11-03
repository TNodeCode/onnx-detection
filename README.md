# ONNX Object Detection Runtime

## ONNX Environment

```bash
$ conda create -y -n onnx-inference python==3.10 pip
$ pip install -r requirements.txt
```

Run inference with the CLI

- Create a directory named `data` and place some images within it
- Create a directory named `work_dirs` and place your detection model ONNX file within it
- Create a script named `run.sh` with the content shown below and run this script.

```bash
python cli.py inference \
    --image-folder ./data \
    --model-path ./work_dirs/faster_rcnn.onnx \
    --output-dir ./output \
    --batch-size 4
```