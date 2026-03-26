"""
CUDA accelerated optical flow for game videos.

Usage:
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output --method farneback
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output --method nvidia2 --save-video
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def hsv_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray,
                 max_magnitude: float = 30.0) -> np.ndarray:
    """Convert optical flow (x, y) to HSV color visualization.

    Args:
        max_magnitude: Fixed upper bound for flow magnitude (pixels/frame).
                       Flows beyond this are clipped to 255.
    """
    mag = cv2.magnitude(flow_x, flow_y)
    ang = cv2.phase(flow_x, flow_y, angleInDegrees=True)

    hsv = np.zeros((*flow_x.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = np.clip(mag * (255.0 / max_magnitude), 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _sample_grid(flow_x, flow_y, step, scale, min_mag=0.5):
    """Sample flow on a grid and return arrays of start/end points and magnitudes."""
    h, w = flow_x.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    gx, gy = np.meshgrid(xs, ys)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()

    fx = flow_x[gy_flat, gx_flat] * scale
    fy = flow_y[gy_flat, gx_flat] * scale
    mag = np.sqrt(fx * fx + fy * fy) / scale  # original magnitude
    mask = mag >= min_mag

    return gx_flat[mask], gy_flat[mask], fx[mask], fy[mask], mag[mask]


def arrow_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray, frame: np.ndarray,
                   step: int = 16, scale: float = 2.0,
                   max_magnitude: float = 30.0) -> np.ndarray:
    """Draw arrow overlay on the original frame."""
    vis = frame.copy()
    sx, sy, fx, fy, mag = _sample_grid(flow_x, flow_y, step, scale)

    t = np.clip(mag / max_magnitude, 0, 1)
    g = (255 * (1 - t)).astype(np.int32)
    r = (255 * t).astype(np.int32)
    ex = (sx + fx).astype(np.int32)
    ey = (sy + fy).astype(np.int32)

    for i in range(len(sx)):
        cv2.arrowedLine(vis, (int(sx[i]), int(sy[i])), (int(ex[i]), int(ey[i])),
                        (0, int(g[i]), int(r[i])), 1, tipLength=0.3)

    return vis


def vector_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray,
                    step: int = 12, scale: float = 2.0,
                    max_magnitude: float = 30.0) -> np.ndarray:
    """Draw colored vector field on black background. Hue = direction."""
    h, w = flow_x.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    sx, sy, fx, fy, mag = _sample_grid(flow_x, flow_y, step, scale)
    if len(sx) == 0:
        return vis

    ang = np.degrees(np.arctan2(fy, fx)) % 360
    hue = (ang / 2).astype(np.uint8)
    val = np.clip(mag / max_magnitude * 255, 0, 255).astype(np.uint8)

    # Batch HSV->BGR conversion
    hsv_row = np.zeros((1, len(sx), 3), dtype=np.uint8)
    hsv_row[0, :, 0] = hue
    hsv_row[0, :, 1] = 255
    hsv_row[0, :, 2] = val
    bgr_row = cv2.cvtColor(hsv_row, cv2.COLOR_HSV2BGR)

    ex = (sx + fx).astype(np.int32)
    ey = (sy + fy).astype(np.int32)

    for i in range(len(sx)):
        color = (int(bgr_row[0, i, 0]), int(bgr_row[0, i, 1]), int(bgr_row[0, i, 2]))
        cv2.arrowedLine(vis, (int(sx[i]), int(sy[i])), (int(ex[i]), int(ey[i])),
                        color, 1, tipLength=0.3)

    return vis


def create_flow_calculator(method: str):
    """Create a CUDA optical flow calculator."""
    if method == "nvidia1":
        # NvidiaOpticalFlow_1_0 - fast, hardware-accelerated (Turing+ GPUs)
        return "nvidia1"
    elif method == "nvidia2":
        # NvidiaOpticalFlow_2_0 - newer hardware flow (Ampere+ GPUs)
        return "nvidia2"
    elif method == "brox":
        return cv2.cuda.BroxOpticalFlow_create(
            alpha=0.197, gamma=50.0, scale_factor=0.8,
            inner_iterations=5, outer_iterations=150, solver_iterations=10
        )
    elif method == "tvl1":
        return cv2.cuda.OpticalFlowDual_TVL1_create()
    elif method == "farneback":
        return cv2.cuda.FarnebackOpticalFlow_create(
            numLevels=5, pyrScale=0.5, fastPyramids=False,
            winSize=13, numIters=10, polyN=5, polySigma=1.1
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use: brox, tvl1, farneback, nvidia1, nvidia2")


def calc_flow_nvidia(cap, width, height, version):
    """Use NvidiaOpticalFlow hardware acceleration."""
    if version == "nvidia1":
        nv = cv2.cuda.NvidiaOpticalFlow_1_0_create(
            (width, height),
            perfPreset=cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_SLOW,
        )
    else:
        nv = cv2.cuda.NvidiaOpticalFlow_2_0_create(
            (width, height),
            perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
        )
    return nv


def process_video(input_path: str, output_dir: str, method: str, save_video: bool,
                  save_raw: bool, max_frames: int, side_by_side: bool = False,
                  max_magnitude: float = 30.0, vis_mode: str = "hsv"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames > 0:
        total = min(total, max_frames)

    print(f"Input : {input_path.name} ({width}x{height}, {fps:.1f}fps, {total} frames)")
    print(f"Method: {method} (CUDA)")
    print(f"Output: {output_dir}")

    # Video writer for visualization
    writer = None
    if save_video:
        suffix = f"_flow_{method}_sbs" if side_by_side else f"_flow_{method}"
        out_path = str(output_dir / f"{input_path.stem}{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = width * 2 if side_by_side else width
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, height))
        print(f"Saving video: {out_path}")

    use_nvidia = method in ("nvidia1", "nvidia2")

    if use_nvidia:
        nv_flow = calc_flow_nvidia(cap, width, height, method)
    else:
        flow_calc = create_flow_calculator(method)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame", file=sys.stderr)
        sys.exit(1)

    if use_nvidia:
        # NvidiaOpticalFlow works on grayscale on GPU
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gpu_prev.upload(prev_gray)

    # Per-second flow magnitude accumulator
    frames_per_sec = int(round(fps))
    mag_accum = 0.0  # sum of absolute flow magnitudes in current second
    sec_idx = 0
    csv_path = output_dir / f"{input_path.stem}_flow_mag_{method}.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["second", "flow_magnitude_sum"])
    print(f"Saving CSV: {csv_path}")

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_nvidia:
            result = nv_flow.calc(prev_gray, curr_gray, None)
            # calc() returns (flow_ndarray_int16, None) at 1/4 resolution
            flow_raw = result[0] if isinstance(result, tuple) else result
            if isinstance(flow_raw, cv2.cuda.GpuMat):
                flow_raw = flow_raw.download()
            # Convert int16 to float and resize to original resolution
            flow = flow_raw.astype(np.float32)
            if flow.shape[0] != height or flow.shape[1] != width:
                scale_y = height / flow.shape[0]
                scale_x = width / flow.shape[1]
                flow_x = cv2.resize(flow[..., 0], (width, height)) * scale_x
                flow_y = cv2.resize(flow[..., 1], (width, height)) * scale_y
            else:
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]
        else:
            gpu_curr.upload(curr_gray)
            gpu_flow = flow_calc.calc(gpu_prev, gpu_curr, None)
            flow = gpu_flow.download()
            # Split flow channels
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

        # Accumulate flow magnitude per second
        mag = cv2.magnitude(flow_x, flow_y)
        mag_accum += float(np.sum(mag))
        if frame_idx % frames_per_sec == 0:
            csv_writer.writerow([sec_idx, f"{mag_accum:.2f}"])
            mag_accum = 0.0
            sec_idx += 1

        # Visualization
        if vis_mode == "arrow":
            vis = arrow_flow_vis(flow_x, flow_y, frame, max_magnitude=max_magnitude)
        elif vis_mode == "vector":
            vis = vector_flow_vis(flow_x, flow_y, max_magnitude=max_magnitude)
        else:
            vis = hsv_flow_vis(flow_x, flow_y, max_magnitude)

        if writer:
            if side_by_side:
                combined = np.hstack([frame, vis])
                writer.write(combined)
            else:
                writer.write(vis)

        # Save raw .npy flow every 30 frames (or all if save_raw)
        if save_raw and frame_idx % 30 == 0:
            raw_path = output_dir / f"flow_{frame_idx:06d}.npy"
            np.save(str(raw_path), np.stack([flow_x, flow_y], axis=-1))

        # Save sample visualization images
        if frame_idx % 60 == 0:
            vis_path = output_dir / f"flow_vis_{frame_idx:06d}.png"
            cv2.imwrite(str(vis_path), vis)

        if use_nvidia:
            prev_gray = curr_gray
        else:
            gpu_prev.upload(curr_gray)

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed
            print(f"  [{frame_idx}/{total}] {fps_proc:.1f} fps", flush=True)

    # Write remaining accumulated frames
    if mag_accum > 0:
        csv_writer.writerow([sec_idx, f"{mag_accum:.2f}"])
    csv_file.close()

    elapsed = time.time() - t_start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")

    if writer:
        writer.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="CUDA Optical Flow for game videos")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--method", "-m", default="farneback",
                        choices=["brox", "tvl1", "farneback", "nvidia1", "nvidia2"],
                        help="Optical flow algorithm (default: farneback)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save flow visualization as video")
    parser.add_argument("--save-raw", action="store_true",
                        help="Save raw flow as .npy every 30 frames")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0 = all)")
    parser.add_argument("--side-by-side", "-sbs", action="store_true",
                        help="Output original and flow side by side")
    parser.add_argument("--max-mag", type=float, default=30.0,
                        help="Fixed max magnitude for visualization (pixels/frame, default: 30)")
    parser.add_argument("--vis-mode", default="hsv",
                        choices=["hsv", "arrow", "vector"],
                        help="Visualization mode: hsv (color), arrow (overlay on frame), vector (field on black)")

    args = parser.parse_args()

    # Print CUDA info
    print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("Error: No CUDA device found", file=sys.stderr)
        sys.exit(1)
    cv2.cuda.printCudaDeviceInfo(0)

    process_video(args.input, args.output, args.method,
                  args.save_video, args.save_raw, args.max_frames,
                  args.side_by_side, args.max_mag, args.vis_mode)


if __name__ == "__main__":
    main()
