# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA-accelerated optical flow extraction from game videos, running inside Docker (NVIDIA GPU required).

## Build & Run

```bash
# Build the Docker image (takes a while — compiles OpenCV with CUDA from source)
docker compose build

# Run with default settings (farneback, saves visualization video)
cp your_video.mp4 input/video.mp4
docker compose up

# Run with custom options
docker compose run --rm optical-flow \
  --input /data/input/video.mp4 \
  --output /data/output \
  --method nvidia2 \
  --save-video \
  --save-raw \
  --max-frames 300
```

## Architecture

- **Dockerfile** — Builds OpenCV 4.9.0 with CUDA (`cudaoptflow`, `cudaimgproc`) on `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- **optical_flow.py** — Single-script pipeline: reads video → computes optical flow on GPU → outputs HSV visualization images/video and optional raw `.npy` flow
- **docker-compose.yml** — Wires up NVIDIA runtime, mounts `input/` (read-only) and `output/`

## Optical Flow Methods

| Method | Flag | Notes |
|---|---|---|
| Farneback | `farneback` | Default. Dense, good balance of speed/quality |
| TVL1 | `tvl1` | Higher quality, slower |
| Brox | `brox` | Research-grade, slowest |
| NVIDIA 1.0 | `nvidia1` | Hardware-accelerated (Turing+ GPU) |
| NVIDIA 2.0 | `nvidia2` | Hardware-accelerated (Ampere+ GPU) |

## Output

- `output/flow_vis_NNNNNN.png` — HSV visualization every 60 frames
- `output/*_flow_*.mp4` — Full visualization video (with `--save-video`)
- `output/flow_NNNNNN.npy` — Raw `(H, W, 2)` float32 flow every 30 frames (with `--save-raw`)
