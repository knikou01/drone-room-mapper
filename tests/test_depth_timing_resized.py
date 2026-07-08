"""Quick check: how long does inference actually take at the real drone
video stream resolution (640x360), not the much larger phone photo
resolution we tested with. Run from your existing real photo:

    python tests/test_depth_timing_resized.py photo.jpg

This resizes the image to 640x360 first, matching DJIDroneVideoStream's
confirmed output ("Capturing frames: 640x360 RGB" in the connection logs),
then times inference. Compare against inference_interval_s=3.0 in
DroneDepthConfig -- if timing here is close to or exceeds 3.0s, that
config value needs to be raised so the background thread doesn't fall
behind.
"""
import sys
import time
from PIL import Image as PILImage

MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_depth_timing_resized.py <path_to_photo>")
        sys.exit(1)

    from transformers import pipeline
    print(f"Loading {MODEL_NAME} ...")
    pipe = pipeline(task="depth-estimation", model=MODEL_NAME)
    print("Model loaded.\n")

    img_full = PILImage.open(sys.argv[1]).convert("RGB")
    print(f"Original size: {img_full.size}")

    img_resized = img_full.resize((640, 360))
    print(f"Resized to: {img_resized.size} (matches DJIDroneVideoStream output)\n")

    # Run twice -- first call sometimes includes warmup overhead (e.g. lazy
    # CUDA/MKL init), second call is more representative of steady-state.
    for i in range(2):
        t0 = time.time()
        result = pipe(img_resized)
        elapsed = time.time() - t0
        label = "warmup" if i == 0 else "steady-state"
        print(f"Run {i+1} ({label}): {elapsed:.2f}s")

    print(f"\nCurrent DroneDepthConfig.inference_interval_s default: 3.0s")
    print("If steady-state timing above is close to or exceeds this, raise it in the blueprint.")


if __name__ == "__main__":
    main()
