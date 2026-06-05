import argparse
import warnings

from ultralytics import RTDETR
from ultralytics.utils import SETTINGS


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML-DETR.")
    parser.add_argument("--model", default="ultralytics/cfg/models/ml-detr/ml-detr.yaml", help="Model yaml or .pt path.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Leave empty for auto.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="ml-detr", help="Experiment name.")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.1, help="Final learning-rate factor.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--resume", default=None, help="Resume checkpoint path, or true.")
    parser.add_argument("--cache", action="store_true", help="Cache images for training.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    SETTINGS["wandb"] = False
    SETTINGS["clearml"] = False

    args = parse_args()
    model = RTDETR(args.model)

    train_kwargs = {
        "data": args.data,
        "cache": args.cache,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "amp": args.amp,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
