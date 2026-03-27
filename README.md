# valo_of - CUDA Optical Flow for Game Videos

CUDA-accelerated optical flow extraction from game videos (e.g. VALORANT). Runs inside Docker with NVIDIA GPU support.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (Turing or newer recommended)

## Quick Start

```bash
# 1. Build the Docker image (compiles OpenCV 4.9.0 with CUDA from source)
docker compose build

# 2. Place your video in the input directory
cp your_video.mp4 input/video.mp4

# 3. Run
docker compose up
```

## Usage

```bash
docker compose run --rm optical-flow \
  --input /data/input/video.mp4 \
  --output /data/output \
  --method farneback \
  --save-video \
  --save-raw \
  --max-frames 300 \
  --side-by-side \
  --max-mag 30
```

### Options

| Option | Default | Description |
|---|---|---|
| `--input`, `-i` | (required) | Input video path |
| `--output`, `-o` | (required) | Output directory |
| `--method`, `-m` | `farneback` | Optical flow algorithm |
| `--save-video` | off | Save flow visualization as MP4 |
| `--save-raw` | off | Save raw flow as `.npy` every 30 frames |
| `--max-frames` | `0` (all) | Limit number of frames to process |
| `--side-by-side`, `-sbs` | off | Output original + flow side by side |
| `--max-mag` | `30.0` | Max magnitude for HSV visualization |
| `--vis-mode` | `hsv` | Visualization mode: `hsv`, `arrow`, `vector` |

### Optical Flow Methods

| Method | Flag | Notes |
|---|---|---|
| Farneback | `farneback` | Dense, good balance of speed and quality |
| TVL1 | `tvl1` | Higher quality, slower |
| Brox | `brox` | Research-grade, slowest |
| NVIDIA 1.0 | `nvidia1` | Hardware-accelerated (Turing+ GPU) |
| NVIDIA 2.0 | `nvidia2` | Hardware-accelerated (Ampere+ GPU) |

## Output

| File | Description |
|---|---|
| `flow_vis_NNNNNN.png` | HSV visualization (every 60 frames) |
| `*_flow_*.mp4` | Full visualization video (`--save-video`) |
| `flow_NNNNNN.npy` | Raw `(H, W, 2)` float32 flow (`--save-raw`, every 30 frames) |
| `*_flow_mag_*.csv` | Per-second flow magnitude sum |

### Visualization Modes

| Mode | Description |
|---|---|
| `hsv` | HSV color map (hue=direction, value=magnitude) |
| `arrow` | Arrows overlaid on the original frame (green=slow, red=fast) |
| `vector` | Colored arrow field on black background (hue=direction) |

## Performance

The pipeline includes several optimizations for throughput:

- **Batched arrow/vector drawing**: Color-binned `cv2.polylines()` instead of per-arrow Python loops
- **GPU-side processing**: `cv2.cuda.cvtColor()`, `cv2.cuda.magnitude()`, `cv2.cuda.absSum()` to minimize CPU-GPU transfers
- **Threaded pipeline**: Separate reader, compute, and writer threads for concurrent I/O and GPU compute
- **FFmpeg HW encode/decode**: Automatic NVDEC/NVENC detection with fallback to CPU codecs

## Architecture

```
valo_of/
├── Dockerfile           # OpenCV 4.9.0 + CUDA + FFmpeg on nvidia/cuda:12.2.0-devel-ubuntu22.04
├── docker-compose.yml   # NVIDIA runtime, mounts input/ (ro) and output/
├── optical_flow.py      # Threaded pipeline with GPU acceleration
├── input/               # Place input videos here
└── output/              # Results are written here
```

## License

MIT
