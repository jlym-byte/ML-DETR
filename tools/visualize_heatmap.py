from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import select_device

CAM_METHODS = (
    "GradCAM",
    "GradCAMPlusPlus",
    "XGradCAM",
    "EigenCAM",
    "HiResCAM",
    "LayerCAM",
    "RandomCAM",
    "EigenGradCAM",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def import_grad_cam():
    try:
        cam_module = importlib.import_module("pytorch_grad_cam")
        image_utils = importlib.import_module("pytorch_grad_cam.utils.image")
    except ImportError as exc:
        raise SystemExit("Missing dependency: install heatmap support with `pip install grad-cam==1.5.4`.") from exc
    return cam_module, image_utils.show_cam_on_image, image_utils.scale_cam_image


def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: height width")


def letterbox(image: np.ndarray, new_shape: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    shape = image.shape[:2]
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    resized_shape = int(round(shape[1] * scale)), int(round(shape[0] * scale))
    pad_w = new_shape[1] - resized_shape[0]
    pad_h = new_shape[0] - resized_shape[1]
    pad_w /= 2
    pad_h /= 2

    if shape[::-1] != resized_shape:
        image = cv2.resize(image, resized_shape, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def extract_predictions(output: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if output.ndim == 3:
        return output[0]
    if output.ndim == 2:
        return output
    raise ValueError(f"Unsupported prediction shape: {tuple(output.shape)}")


class RTDETRActivationsAndGradients:
    def __init__(self, model: torch.nn.Module, target_layers: list[torch.nn.Module], reshape_transform=None) -> None:
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, inputs, output) -> None:
        activation = self.reshape_transform(output) if self.reshape_transform is not None else output
        self.activations.append(activation.detach().cpu())

    def save_gradient(self, module, inputs, output) -> None:
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def store_gradient(grad):
            grad = self.reshape_transform(grad) if self.reshape_transform is not None else grad
            self.gradients = [grad.detach().cpu()] + self.gradients

        output.register_hook(store_gradient)

    @staticmethod
    def post_process(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = predictions[:, 4:]
        boxes = predictions[:, :4]
        scores = logits.max(dim=1).values
        indices = torch.argsort(scores, descending=True)
        return logits[indices], boxes[indices]

    def __call__(self, x: torch.Tensor):
        self.gradients = []
        self.activations = []
        predictions = extract_predictions(self.model(x))
        logits, boxes = self.post_process(predictions)
        return [[logits, boxes]]

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()


class RTDETRTarget(torch.nn.Module):
    def __init__(self, target_type: str, conf_threshold: float, top_ratio: float) -> None:
        super().__init__()
        self.target_type = target_type
        self.conf_threshold = conf_threshold
        self.top_ratio = top_ratio

    def forward(self, data):
        logits, boxes = data
        count = max(1, int(logits.size(0) * self.top_ratio))
        values = []
        for index in range(count):
            score = logits[index].max()
            if float(score) < self.conf_threshold:
                break
            if self.target_type in {"class", "all"}:
                values.append(score)
            if self.target_type in {"box", "all"}:
                values.extend(boxes[index, coord] for coord in range(4))
        if values:
            return sum(values)
        return logits.max() * 0.0


class RTDETRHeatmap:
    def __init__(
        self,
        weights: Path,
        device: str,
        method: str,
        layers: list[int],
        target_type: str,
        conf_threshold: float,
        top_ratio: float,
        show_boxes: bool,
        renormalize: bool,
        imgsz: tuple[int, int],
        use_letterbox: bool,
    ) -> None:
        cam_module, show_cam_on_image, scale_cam_image = import_grad_cam()
        self.show_cam_on_image = show_cam_on_image
        self.scale_cam_image = scale_cam_image
        self.device = select_device(device)
        self.model = attempt_load_weights(str(weights), self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

        max_layer = len(self.model.model) - 1
        invalid_layers = [layer for layer in layers if layer < 0 or layer > max_layer]
        if invalid_layers:
            raise ValueError(f"Invalid layer indices {invalid_layers}; valid range is 0..{max_layer}.")

        target_layers = [self.model.model[layer] for layer in layers]
        method_class = getattr(cam_module, method)
        try:
            self.cam = method_class(model=self.model, target_layers=target_layers)
        except TypeError:
            self.cam = method_class(self.model, target_layers)
        self.cam.activations_and_grads = RTDETRActivationsAndGradients(self.model, target_layers, None)

        names = getattr(self.model, "names", None) or {}
        self.names = names if isinstance(names, dict) else {index: name for index, name in enumerate(names)}
        self.colors = np.random.default_rng(0).integers(0, 255, size=(max(len(self.names), 1), 3), dtype=np.int64)
        self.target = RTDETRTarget(target_type, conf_threshold, top_ratio)
        self.conf_threshold = conf_threshold
        self.show_boxes = show_boxes
        self.renormalize = renormalize
        self.imgsz = imgsz
        self.use_letterbox = use_letterbox

    def preprocess(self, image_path: Path) -> tuple[torch.Tensor, np.ndarray, tuple[int, int]]:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        original_shape = bgr.shape[:2]
        if self.use_letterbox:
            bgr = letterbox(bgr, self.imgsz)
        else:
            bgr = cv2.resize(bgr, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_float = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)
        return tensor, image_float, original_shape

    def post_process(self, predictions: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        logits = predictions[:, 4:]
        boxes = predictions[:, :4]
        scores, classes = logits.max(dim=1, keepdim=True)
        keep = scores.squeeze(1) > self.conf_threshold
        if not keep.any():
            return predictions.new_zeros((0, 6))

        height, width = shape
        boxes = xywh2xyxy(boxes[keep])
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return torch.cat([boxes, scores[keep], classes[keep].float()], dim=1)

    def draw_detections(self, image: np.ndarray, detections: torch.Tensor) -> np.ndarray:
        for detection in detections.detach().cpu().numpy():
            x1, y1, x2, y2, score, class_id = detection
            class_id = int(class_id)
            color = tuple(int(value) for value in self.colors[class_id % len(self.colors)])
            label = self.names.get(class_id, str(class_id))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                image,
                f"{label} {score:.2f}",
                (int(x1), max(0, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        return image

    def renormalize_cam(self, detections: torch.Tensor, image_float: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
        renormalized = np.zeros_like(grayscale_cam, dtype=np.float32)
        for box in detections[:, :4].detach().cpu().numpy().astype(np.int32):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, grayscale_cam.shape[1] - 1), min(y2, grayscale_cam.shape[0] - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            renormalized[y1:y2, x1:x2] = self.scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        return self.show_cam_on_image(image_float, self.scale_cam_image(renormalized), use_rgb=True)

    def process(self, image_path: Path, output_path: Path) -> None:
        tensor, image_float, original_shape = self.preprocess(image_path)
        try:
            grayscale_cam = self.cam(input_tensor=tensor, targets=[self.target])
        except TypeError:
            grayscale_cam = self.cam(tensor, [self.target])
        grayscale_cam = grayscale_cam[0]

        predictions = extract_predictions(self.model(tensor))
        detections = self.post_process(predictions, image_float.shape[:2])
        if self.renormalize and len(detections):
            cam_image = self.renormalize_cam(detections, image_float, grayscale_cam)
        else:
            cam_image = self.show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
        if self.show_boxes and len(detections):
            cam_image = self.draw_detections(cam_image, detections)

        cam_image = cv2.resize(cam_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cam_image).save(output_path)


def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create class activation heatmaps for ML-DETR predictions.")
    parser.add_argument("--weights", type=Path, default=Path("best.pt"), help="Path to a trained .pt checkpoint.")
    parser.add_argument("--source", type=Path, default=Path("image.jpg"), help="Image file or directory of images.")
    parser.add_argument("--output", type=Path, default=Path("runs/heatmaps"), help="Output directory.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 640], help="Input size: 640 or 640 640.")
    parser.add_argument("--device", default="", help="CUDA device such as 0, or cpu. Empty uses auto-select.")
    parser.add_argument("--method", choices=CAM_METHODS, default="EigenCAM", help="CAM method.")
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 23, 26], help="Target layer indices.")
    parser.add_argument("--target-type", choices=("class", "box", "all"), default="all", help="CAM target objective.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--top-ratio", type=float, default=0.1, help="Fraction of top predictions used as CAM targets.")
    parser.add_argument("--show-boxes", action="store_true", help="Draw predicted boxes on the heatmap.")
    parser.add_argument("--renormalize", action="store_true", help="Normalize CAM values inside predicted boxes.")
    parser.add_argument("--letterbox", action="store_true", help="Preserve aspect ratio with padding instead of resizing directly.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively when --source is a directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    output = Path(args.output)

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not 0 < args.top_ratio <= 1:
        raise ValueError("--top-ratio must be in the range (0, 1].")

    runner = RTDETRHeatmap(
        weights=weights,
        device=args.device,
        method=args.method,
        layers=args.layers,
        target_type=args.target_type,
        conf_threshold=args.conf_thres,
        top_ratio=args.top_ratio,
        show_boxes=args.show_boxes,
        renormalize=args.renormalize,
        imgsz=parse_imgsz(args.imgsz),
        use_letterbox=args.letterbox,
    )

    images = list(iter_images(source, args.recursive))
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")

    for image_path in tqdm(images, desc="Heatmaps"):
        output_path = output / f"{image_path.stem}_heatmap.png"
        runner.process(image_path, output_path)
    print(f"Saved {len(images)} heatmap(s) to {output}")


if __name__ == "__main__":
    main()
