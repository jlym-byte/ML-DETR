# ML-DETR

Minimal source-code repository for ML-DETR based on Ultralytics RT-DETR.

## Architecture

<p align="center">
  <img src="assets/ml-detr-ecie.png" alt="ML-DETR overall architecture with ECIE" width="100%">
</p>

<p align="center"><b>Figure 1. ML-DETR overall architecture with ECIE.</b></p>

<p align="center">
  <img src="assets/toglsr-ssmamba.png" alt="TOGLSR and SSMamba modules" width="100%">
</p>

<p align="center"><b>Figure 2. TOGLSR and SSMamba modules.</b></p>

## Structure

```text
train.py
test.py
requirements.txt
assets/                                      # README figures
tools/benchmark_latency.py                  # inference latency and FPS benchmark
tools/visualize_heatmap.py                  # CAM heatmap visualization
ultralytics/nn/modules/block.py              # includes ECIE, TOGLSR, SSMamba
ultralytics/cfg/models/ml-detr/              # ML-DETR model YAML files
ultralytics/cfg/models/rt-detr/              # baseline RT-DETR YAML files
ultralytics/models/rtdetr/                   # RT-DETR train/val/predict/model code
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

This repository targets the full ML-DETR environment. `requirements.txt` includes `mamba-ssm` and `causal-conv1d` because `ml-detr.yaml` uses `SSMamba` in `ultralytics/nn/modules/block.py`.

## Train

Edit `dataset/data.yaml`, then run:

```bash
python train.py --data dataset/data.yaml --model ultralytics/cfg/models/ml-detr/ml-detr.yaml
```

Common options:

```bash
python train.py --data path/to/data.yaml --epochs 150 --batch 8 --device 0 --name ml-detr
```

## Test

Evaluate a trained checkpoint and save YOLO metrics plus paper-ready summary tables:

```bash
python test.py --weights path/to/best.pt --data dataset/data.yaml --split test --device 0
```

The script writes `paper_data.txt` under the evaluation run directory. COCO official metrics are added when a matching annotation json is found or passed with `--coco-ann-json`.

## Tools

Benchmark inference latency and FPS:

```bash
python tools/benchmark_latency.py --weights path/to/best.pt --device 0 --half
```

Generate CAM heatmaps for images:

```bash
python tools/visualize_heatmap.py --weights path/to/best.pt --source path/to/images --output runs/heatmaps --layers 20 23 26
```

For RT-DETR baseline configs, use `--layers 19 22 25`.

## Model Configs

```text
ultralytics/cfg/models/ml-detr/ml-detr.yaml
ultralytics/cfg/models/ml-detr/ml-detr-ecie.yaml
ultralytics/cfg/models/ml-detr/ml-detr-ecie-toglsr.yaml
ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml
ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml
ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml
```
