"""
CUDA accelerated optical flow for game videos.

Usage:
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output --method farneback
    python3 optical_flow.py --input /data/input/video.mp4 --output /data/output --method nvidia2 --save-video
"""

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import numpy as np

# --- Visualization functions ---

def hsv_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray,
                 max_magnitude: float = 30.0) -> np.ndarray:
    """Convert optical flow (x, y) to HSV color visualization."""
    mag = cv2.magnitude(flow_x, flow_y)
    ang = cv2.phase(flow_x, flow_y, angleInDegrees=True)

    hsv = np.zeros((*flow_x.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
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
    mag = np.sqrt(fx * fx + fy * fy) / scale
    mask = mag >= min_mag

    return gx_flat[mask], gy_flat[mask], fx[mask], fy[mask], mag[mask]


_ARROW_TIP_COS = np.cos(np.radians(150))
_ARROW_TIP_SIN = np.sin(np.radians(150))
_NUM_COLOR_BINS = 16


def _build_arrow_segments(sx, sy, fx, fy, tip_len=0.3):
    """Build shaft + arrowhead line segments as (N*3, 2, 2) int32 array."""
    ex = sx + fx
    ey = sy + fy
    n = len(sx)

    # Shaft segments
    shafts = np.stack([
        np.stack([sx, sy], axis=-1),
        np.stack([ex, ey], axis=-1),
    ], axis=1)  # (N, 2, 2)

    # Arrowhead wings via rotation
    dx = -fx * tip_len
    dy = -fy * tip_len
    wing1_x = ex + dx * _ARROW_TIP_COS - dy * _ARROW_TIP_SIN
    wing1_y = ey + dx * _ARROW_TIP_SIN + dy * _ARROW_TIP_COS
    wing2_x = ex + dx * _ARROW_TIP_COS + dy * _ARROW_TIP_SIN
    wing2_y = ey - dx * _ARROW_TIP_SIN + dy * _ARROW_TIP_COS

    tips = np.stack([ex, ey], axis=-1)  # (N, 2)
    w1 = np.stack([wing1_x, wing1_y], axis=-1)
    w2 = np.stack([wing2_x, wing2_y], axis=-1)

    wing1_segs = np.stack([tips, w1], axis=1)
    wing2_segs = np.stack([tips, w2], axis=1)

    # Combine: (N*3, 2, 2)
    all_segs = np.concatenate([shafts, wing1_segs, wing2_segs], axis=0)
    return all_segs.astype(np.int32)


def arrow_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray, frame: np.ndarray,
                   step: int = 16, scale: float = 2.0,
                   max_magnitude: float = 30.0) -> np.ndarray:
    """Draw arrow overlay on the original frame using batched polylines."""
    vis = frame.copy()
    sx, sy, fx, fy, mag = _sample_grid(flow_x, flow_y, step, scale)
    if len(sx) == 0:
        return vis

    # Color bins: green(0) -> yellow -> red(1)
    t = np.clip(mag / max_magnitude, 0, 1)
    bins = np.clip((t * _NUM_COLOR_BINS).astype(np.int32), 0, _NUM_COLOR_BINS - 1)

    all_segs = _build_arrow_segments(sx, sy, fx, fy)
    n = len(sx)
    # all_segs has 3*n segments: [shafts(n), wing1(n), wing2(n)]
    # bins applies to each arrow, so repeat for 3 segment types
    seg_bins = np.tile(bins, 3)

    for b in range(_NUM_COLOR_BINS):
        mask = seg_bins == b
        if not np.any(mask):
            continue
        bt = (b + 0.5) / _NUM_COLOR_BINS
        color = (0, int(255 * (1 - bt)), int(255 * bt))
        segs = all_segs[mask]
        # polylines expects list of (num_points, 1, 2) arrays
        lines = [s.reshape(-1, 1, 2) for s in segs]
        cv2.polylines(vis, lines, isClosed=False, color=color, thickness=1)

    return vis


def vector_flow_vis(flow_x: np.ndarray, flow_y: np.ndarray,
                    step: int = 12, scale: float = 2.0,
                    max_magnitude: float = 30.0) -> np.ndarray:
    """Draw colored vector field on black background using batched polylines."""
    h, w = flow_x.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    sx, sy, fx, fy, mag = _sample_grid(flow_x, flow_y, step, scale)
    if len(sx) == 0:
        return vis

    ang = np.degrees(np.arctan2(fy, fx)) % 360
    hue = (ang / 2).astype(np.uint8)
    val = np.clip(mag / max_magnitude * 255, 0, 255).astype(np.uint8)

    # Batch HSV->BGR for color bins
    bins = np.clip((hue / 180.0 * _NUM_COLOR_BINS).astype(np.int32), 0, _NUM_COLOR_BINS - 1)

    all_segs = _build_arrow_segments(sx, sy, fx, fy)
    n = len(sx)
    seg_bins = np.tile(bins, 3)

    # Precompute bin colors from representative hue/val
    for b in range(_NUM_COLOR_BINS):
        mask_pts = bins == b
        if not np.any(mask_pts):
            continue
        avg_hue = int(np.mean(hue[mask_pts]))
        avg_val = int(np.mean(val[mask_pts]))
        hsv_px = np.array([[[avg_hue, 255, avg_val]]], dtype=np.uint8)
        bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)
        color = (int(bgr_px[0, 0, 0]), int(bgr_px[0, 0, 1]), int(bgr_px[0, 0, 2]))

        seg_mask = seg_bins == b
        segs = all_segs[seg_mask]
        lines = [s.reshape(-1, 1, 2) for s in segs]
        cv2.polylines(vis, lines, isClosed=False, color=color, thickness=1)

    return vis


# --- Flow calculator setup ---

def create_flow_calculator(method: str):
    """Create a CUDA optical flow calculator."""
    if method == "nvidia1":
        return "nvidia1"
    elif method == "nvidia2":
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


# --- FFmpeg pipe helpers (P3) ---

def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _has_nvenc():
    """Check if h264_nvenc encoder is available."""
    if not _has_ffmpeg():
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def _has_nvdec():
    """Check if h264_cuvid decoder is available."""
    if not _has_ffmpeg():
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=5
        )
        return "h264_cuvid" in r.stdout
    except Exception:
        return False


class FFmpegReader:
    """Read video frames via FFmpeg subprocess pipe."""

    def __init__(self, path: str, width: int, height: int, use_hwdec: bool = False):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if use_hwdec:
            cmd += ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        cmd += ["-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=self.frame_size * 4)

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame

    def release(self):
        self.proc.stdout.close()
        self.proc.wait()


class FFmpegWriter:
    """Write video frames via FFmpeg subprocess pipe."""

    def __init__(self, path: str, width: int, height: int, fps: float, use_nvenc: bool = False):
        self.width = width
        self.height = height
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
        ]
        if use_nvenc:
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        cmd += ["-pix_fmt", "yuv420p", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=width * height * 3 * 4)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


# --- Pipeline threads (P2) ---

_SENTINEL = None


def _reader_thread(cap, max_frames, read_queue):
    """Read frames and put into queue."""
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            read_queue.put((frame_idx, frame))
            break
        read_queue.put((frame_idx, frame))
    read_queue.put(_SENTINEL)


def _writer_thread(write_queue, writer, output_dir, input_stem):
    """Write visualization frames from queue."""
    while True:
        item = write_queue.get()
        if item is _SENTINEL:
            break
        frame_idx, vis, frame, side_by_side, save_raw, flow_xy = item

        if writer:
            if side_by_side:
                combined = np.hstack([frame, vis])
                writer.write(combined)
            else:
                writer.write(vis)

        if save_raw and frame_idx % 30 == 0:
            raw_path = output_dir / f"flow_{frame_idx:06d}.npy"
            np.save(str(raw_path), flow_xy)

        if frame_idx % 60 == 0:
            vis_path = output_dir / f"flow_vis_{frame_idx:06d}.png"
            cv2.imwrite(str(vis_path), vis)


# --- Main processing ---

def process_video(input_path: str, output_dir: str, method: str, save_video: bool,
                  save_raw: bool, max_frames: int, side_by_side: bool = False,
                  max_magnitude: float = 30.0, vis_mode: str = "hsv"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe video metadata with OpenCV first
    probe_cap = cv2.VideoCapture(str(input_path))
    if not probe_cap.isOpened():
        print(f"Error: Cannot open {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = probe_cap.get(cv2.CAP_PROP_FPS)
    width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_cap.release()

    if max_frames > 0:
        total = min(total, max_frames)

    print(f"Input : {input_path.name} ({width}x{height}, {fps:.1f}fps, {total} frames)")
    print(f"Method: {method} (CUDA)")
    print(f"Vis   : {vis_mode}")
    print(f"Output: {output_dir}")

    # P3: Choose reader backend
    use_ffmpeg = _has_ffmpeg()
    use_hwdec = use_ffmpeg and _has_nvdec()
    use_nvenc = use_ffmpeg and _has_nvenc()

    if use_ffmpeg:
        cap = FFmpegReader(str(input_path), width, height, use_hwdec)
        print(f"Decode: FFmpeg {'(NVDEC)' if use_hwdec else '(CPU)'}")
    else:
        cap = cv2.VideoCapture(str(input_path))
        print("Decode: OpenCV")

    # P3: Choose writer backend
    writer = None
    out_path = None
    if save_video:
        suffix = f"_flow_{method}_{vis_mode}"
        if side_by_side:
            suffix += "_sbs"
        out_path = str(output_dir / f"{input_path.stem}{suffix}.mp4")
        out_w = width * 2 if side_by_side else width

        if use_ffmpeg:
            writer = FFmpegWriter(out_path, out_w, height, fps, use_nvenc)
            print(f"Encode: FFmpeg {'(NVENC)' if use_nvenc else '(libx264)'}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, height))
            print("Encode: OpenCV mp4v")
        print(f"Saving video: {out_path}")

    use_nvidia = method in ("nvidia1", "nvidia2")

    if use_nvidia:
        nv_flow = calc_flow_nvidia(cap, width, height, method)
    else:
        flow_calc = create_flow_calculator(method)

    # P1: Pre-allocate GPU matrices for non-nvidia methods
    if not use_nvidia:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_frame = cv2.cuda_GpuMat()

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame", file=sys.stderr)
        sys.exit(1)

    # P1: GPU-side cvtColor
    if use_nvidia:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gpu_frame.upload(prev_frame)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        gpu_prev.upload(gpu_gray.download())

    # Per-second flow magnitude accumulator
    frames_per_sec = int(round(fps))
    mag_accum = 0.0
    sec_idx = 0
    csv_path = output_dir / f"{input_path.stem}_flow_mag_{method}.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["second", "flow_magnitude_sum"])
    print(f"Saving CSV: {csv_path}")

    # P2: Start reader and writer threads
    read_queue = Queue(maxsize=4)
    write_queue = Queue(maxsize=4)

    reader = threading.Thread(
        target=_reader_thread,
        args=(cap, max_frames, read_queue),
        daemon=True,
    )

    writer_thread = threading.Thread(
        target=_writer_thread,
        args=(write_queue, writer, output_dir, input_path.stem),
        daemon=True,
    )

    reader.start()
    writer_thread.start()

    frame_idx = 0
    t_start = time.time()

    while True:
        item = read_queue.get()
        if item is _SENTINEL:
            break

        frame_idx, frame = item

        # P1: GPU-side color conversion for non-nvidia methods
        if use_nvidia:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = nv_flow.calc(prev_gray, curr_gray, None)
            flow_raw = result[0] if isinstance(result, tuple) else result
            if isinstance(flow_raw, cv2.cuda.GpuMat):
                flow_raw = flow_raw.download()
            flow = flow_raw.astype(np.float32)
            if flow.shape[0] != height or flow.shape[1] != width:
                scale_y = height / flow.shape[0]
                scale_x = width / flow.shape[1]
                flow_x = cv2.resize(flow[..., 0], (width, height)) * scale_x
                flow_y = cv2.resize(flow[..., 1], (width, height)) * scale_y
            else:
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]
            prev_gray = curr_gray
        else:
            gpu_frame.upload(frame)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gpu_curr = gpu_gray
            gpu_flow = flow_calc.calc(gpu_prev, gpu_curr, None)

            # P1: GPU-side magnitude for CSV accumulation
            gpu_channels = cv2.cuda.split(gpu_flow)
            gpu_mag = cv2.cuda.magnitude(gpu_channels[0], gpu_channels[1])
            mag_sum_arr = cv2.cuda.absSum(gpu_mag)
            mag_accum += mag_sum_arr[0]

            flow = gpu_flow.download()
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            gpu_prev = gpu_curr

        # Accumulate flow magnitude per second (nvidia path still uses CPU)
        if use_nvidia:
            mag = cv2.magnitude(flow_x, flow_y)
            mag_accum += float(np.sum(mag))

        if frame_idx % frames_per_sec == 0:
            csv_writer.writerow([sec_idx, f"{mag_accum:.2f}"])
            mag_accum = 0.0
            sec_idx += 1

        # Visualization (P0: batched drawing)
        if vis_mode == "arrow":
            vis = arrow_flow_vis(flow_x, flow_y, frame, max_magnitude=max_magnitude)
        elif vis_mode == "vector":
            vis = vector_flow_vis(flow_x, flow_y, max_magnitude=max_magnitude)
        else:
            vis = hsv_flow_vis(flow_x, flow_y, max_magnitude)

        # Send to writer thread (P2)
        flow_xy = np.stack([flow_x, flow_y], axis=-1) if save_raw else None
        write_queue.put((frame_idx, vis, frame, side_by_side, save_raw, flow_xy))

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed
            print(f"  [{frame_idx}/{total}] {fps_proc:.1f} fps", flush=True)

    # Flush remaining CSV
    if mag_accum > 0:
        csv_writer.writerow([sec_idx, f"{mag_accum:.2f}"])
    csv_file.close()

    # Signal writer thread to finish
    write_queue.put(_SENTINEL)
    writer_thread.join()

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
