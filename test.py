import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

from ultralytics import RTDETR
from ultralytics.utils import SETTINGS, yaml_load
from ultralytics.utils.torch_utils import model_info


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ML-DETR and export paper-ready metrics.")
    parser.add_argument("--weights", default="/weights/best.pt", help="Path to a trained .pt checkpoint.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset yaml path.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Leave empty for auto.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold passed to model.val().")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold passed to model.val().")
    parser.add_argument("--project", default="runs/test", help="Output project directory.")
    parser.add_argument("--name", default="ml-detr", help="Evaluation run name.")
    parser.add_argument("--coco-ann-json", default=None, help="Optional COCO annotation json for official COCOeval metrics.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format prediction txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in prediction txt files.")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false", help="Disable predictions.json export.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing output directory.")
    parser.set_defaults(save_json=True)
    return parser.parse_args()


def metric_format(value):
    if value is None:
        return "nan"
    value = float(value)
    if np.isnan(value) or value < 0:
        return "nan"
    return f"{value:.4f}"


def to_numpy(value):
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def resolve_path(path, base):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def get_weight_size_mb(path):
    return Path(path).stat().st_size / 1024 / 1024


def find_coco_annotation(data_yaml, split, override=None):
    if override:
        path = Path(override)
        return path if path.is_file() else None

    data_yaml = Path(data_yaml)
    try:
        data = yaml_load(data_yaml)
    except Exception:
        data = {}

    base = Path(data.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    candidates = [
        data_yaml.with_name(f"{split}.json"),
        base / f"{split}.json",
        base / "annotations" / f"{split}.json",
        base / "annotations" / f"instances_{split}.json",
        base / "annotations" / f"instances_{split}2017.json",
    ]

    split_value = data.get(split)
    if isinstance(split_value, str):
        split_path = resolve_path(split_value, base)
        candidates.extend([
            split_path / f"{split}.json",
            split_path.with_suffix(".json"),
            split_path.parent / f"{split}.json",
            split_path.parent.parent / f"{split}.json",
        ])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def build_model_info_table(model, result, weights):
    _, params, _, flops = model_info(model.model)
    speed = result.speed
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    table = PrettyTable()
    table.title = "Model Info"
    table.field_names = [
        "GFLOPs",
        "Parameters",
        "Preprocess s/img",
        "Inference s/img",
        "Postprocess s/img",
        "FPS end-to-end",
        "FPS inference",
        "Model size",
    ]
    table.add_row([
        f"{flops:.1f}",
        f"{params:,}",
        f"{preprocess_ms / 1000:.6f}",
        f"{inference_ms / 1000:.6f}",
        f"{postprocess_ms / 1000:.6f}",
        f"{1000 / total_ms:.2f}" if total_ms > 0 else "nan",
        f"{1000 / inference_ms:.2f}" if inference_ms > 0 else "nan",
        f"{get_weight_size_mb(weights):.1f} MB",
    ])
    return table


def build_yolo_metrics_table(result):
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    box = result.box
    precision = to_numpy(getattr(box, "p", None))
    recall = to_numpy(getattr(box, "r", None))
    f1 = to_numpy(getattr(box, "f1", None))
    ap50 = to_numpy(getattr(box, "ap50", None))
    ap = to_numpy(getattr(box, "ap", None))
    all_ap = to_numpy(getattr(box, "all_ap", None))

    table = PrettyTable()
    table.title = "YOLO Metrics"
    table.field_names = ["Class", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]

    class_count = max(len(names), len(precision), len(recall), len(ap50), len(ap))
    for idx in range(class_count):
        class_name = names.get(idx, str(idx))
        ap75 = all_ap[idx, 5] if all_ap.ndim == 2 and idx < all_ap.shape[0] and all_ap.shape[1] > 5 else np.nan
        table.add_row([
            class_name,
            metric_format(precision[idx] if idx < len(precision) else np.nan),
            metric_format(recall[idx] if idx < len(recall) else np.nan),
            metric_format(f1[idx] if idx < len(f1) else np.nan),
            metric_format(ap50[idx] if idx < len(ap50) else np.nan),
            metric_format(ap75),
            metric_format(ap[idx] if idx < len(ap) else np.nan),
        ])

    results = result.results_dict
    mean_ap75 = np.nanmean(all_ap[:, 5]) if all_ap.ndim == 2 and all_ap.shape[1] > 5 else np.nan
    table.add_row([
        "all",
        metric_format(results.get("metrics/precision(B)", np.nan)),
        metric_format(results.get("metrics/recall(B)", np.nan)),
        metric_format(np.nanmean(f1) if len(f1) else np.nan),
        metric_format(results.get("metrics/mAP50(B)", np.nan)),
        metric_format(mean_ap75),
        metric_format(results.get("metrics/mAP50-95(B)", np.nan)),
    ])
    return table


def align_prediction_image_ids(annotation_json, prediction_json):
    """Align prediction image_id values with COCO annotation image ids.

    Ultralytics may export custom-dataset predictions with image_id set to the
    image filename stem, while COCO annotations often use integer ids. COCOeval
    requires both sides to use the same image_id values.
    """
    annotation_json = Path(annotation_json)
    prediction_json = Path(prediction_json)

    with annotation_json.open("r", encoding="utf-8") as file:
        annotation_data = json.load(file)
    with prediction_json.open("r", encoding="utf-8") as file:
        prediction_data = json.load(file)

    annotation_images = annotation_data.get("images", [])
    annotation_ids = {image["id"] for image in annotation_images}
    prediction_ids = {item["image_id"] for item in prediction_data}

    if not prediction_ids or prediction_ids <= annotation_ids:
        return prediction_json

    stem_to_id = {
        Path(image["file_name"]).stem: image["id"]
        for image in annotation_images
        if "file_name" in image and "id" in image
    }

    aligned_predictions = []
    missing = 0
    for item in prediction_data:
        image_id = item.get("image_id")
        if image_id in annotation_ids:
            aligned_predictions.append(item)
            continue

        mapped_id = stem_to_id.get(str(image_id))
        if mapped_id is None:
            missing += 1
            continue

        aligned_item = dict(item)
        aligned_item["image_id"] = mapped_id
        aligned_predictions.append(aligned_item)

    aligned_json = prediction_json.with_name(f"{prediction_json.stem}_aligned.json")
    with aligned_json.open("w", encoding="utf-8") as file:
        json.dump(aligned_predictions, file, ensure_ascii=False)

    print(
        "COCOeval image_id alignment: "
        f"{len(aligned_predictions)} predictions matched, {missing} predictions skipped. "
        f"Saved {aligned_json}."
    )
    return aligned_json


def build_coco_metrics_table(annotation_json, prediction_json):
    if not annotation_json or not Path(annotation_json).is_file():
        print("COCOeval skipped: annotation json was not found.")
        return None
    if not prediction_json or not Path(prediction_json).is_file():
        print(f"COCOeval skipped: prediction json was not found at {prediction_json}.")
        return None

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        print(f"COCOeval skipped: pycocotools import failed: {exc}")
        return None

    try:
        annotation = COCO(str(annotation_json))
        prediction_json = align_prediction_image_ids(annotation_json, prediction_json)
        prediction = annotation.loadRes(str(prediction_json))
        evaluator = COCOeval(annotation, prediction, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    except Exception as exc:
        print(f"COCOeval skipped: evaluation failed: {exc}")
        return None

    stats = evaluator.stats
    table = PrettyTable()
    table.title = "COCO Official Metrics"
    table.field_names = ["AP50", "AP50-95", "AP75", "AP_small", "AP_medium", "AP_large"]
    table.add_row([
        metric_format(stats[1]),
        metric_format(stats[0]),
        metric_format(stats[2]),
        metric_format(stats[3]),
        metric_format(stats[4]),
        metric_format(stats[5]),
    ])
    return table


def write_summary(save_dir, tables):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / "paper_data.txt"
    with output.open("w", encoding="utf-8") as file:
        for index, table in enumerate(tables):
            if index:
                file.write("\n")
            file.write(str(table))
            file.write("\n")
    return output


def main():
    SETTINGS["wandb"] = False
    SETTINGS["clearml"] = False

    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = RTDETR(str(weights))
    val_kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "save_json": args.save_json,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    if args.iou is not None:
        val_kwargs["iou"] = args.iou

    result = model.val(**val_kwargs)
    if model.task != "detect":
        return

    print("\n" + "=" * 72)
    print("Paper-ready evaluation summary")
    print("Use the following tables for reported model complexity, speed, and accuracy.")
    print("=" * 72)

    model_info_table = build_model_info_table(model, result, weights)
    yolo_metrics_table = build_yolo_metrics_table(result)
    print(model_info_table)
    print(yolo_metrics_table)

    tables = [model_info_table, yolo_metrics_table]
    prediction_json = Path(result.save_dir) / "predictions.json"
    annotation_json = find_coco_annotation(args.data, args.split, args.coco_ann_json)
    coco_metrics_table = build_coco_metrics_table(annotation_json, prediction_json)
    if coco_metrics_table is not None:
        print(coco_metrics_table)
        tables.append(coco_metrics_table)

    output = write_summary(result.save_dir, tables)
    print(f"Saved paper-ready metrics to {output}")


if __name__ == "__main__":
    main()
