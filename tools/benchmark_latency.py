from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import select_device


def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: height width")


def weight_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, iterations: int) -> np.ndarray:
    times = []
    with torch.inference_mode():
        for _ in tqdm(range(iterations), desc="Benchmark"):
            synchronize(device)
            start = time.perf_counter()
            _ = model(inputs)
            synchronize(device)
            times.append(time.perf_counter() - start)
    return np.asarray(times, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ML-DETR inference latency and FPS.")
    parser.add_argument("--weights", type=Path, default=Path("best.pt"), help="Path to a trained .pt checkpoint.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 640], help="Input size: 640 or 640 640.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size used for synthetic inputs.")
    parser.add_argument("--device", default="", help="CUDA device such as 0 or 0,1, or cpu. Empty uses auto-select.")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of timed iterations.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format.")
    parser.add_argument("--no-fuse", dest="fuse", action="store_false", help="Skip Conv-BN fusion when supported.")
    parser.set_defaults(fuse=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    height, width = parse_imgsz(args.imgsz)
    device = select_device(args.device, batch=args.batch)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = attempt_load_weights(str(weights), device=device).to(device).eval()
    if args.fuse and hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception as exc:  # Fusion is an optimization, not a correctness requirement.
            print(f"Warning: model fusion skipped: {exc}")

    use_half = args.half and device.type == "cuda"
    if args.half and not use_half:
        print("Warning: --half is only available on CUDA; using FP32 instead.")

    dtype = torch.float16 if use_half else torch.float32
    model = model.half() if use_half else model.float()
    inputs = torch.randn(args.batch, 3, height, width, device=device, dtype=dtype)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        inputs = inputs.to(memory_format=torch.channels_last)

    print("=" * 72)
    print(f"Weights:       {weights}")
    print(f"Weight size:   {weight_size_mb(weights):.2f} MB")
    print(f"Device:        {device}")
    print(f"Input size:    {height}x{width}")
    print(f"Batch size:    {args.batch}")
    print(f"Precision:     {'FP16' if use_half else 'FP32'}")
    print(f"Iterations:    warmup={args.warmup}, timed={args.iterations}")
    print("=" * 72)

    with torch.inference_mode():
        for _ in tqdm(range(args.warmup), desc="Warmup"):
            _ = model(inputs)
    synchronize(device)

    times = benchmark(model, inputs, device, args.iterations)
    per_image_ms = times / args.batch * 1000.0
    fps = 1000.0 / per_image_ms.mean()

    print("\nInference latency")
    print("=" * 72)
    print(f"Mean latency:   {per_image_ms.mean():.3f} ms/image")
    print(f"Median latency: {np.median(per_image_ms):.3f} ms/image")
    print(f"P95 latency:    {np.percentile(per_image_ms, 95):.3f} ms/image")
    print(f"Std latency:    {per_image_ms.std():.3f} ms/image")
    print(f"FPS:            {fps:.2f} images/s")
    print("=" * 72)


if __name__ == "__main__":
    main()
