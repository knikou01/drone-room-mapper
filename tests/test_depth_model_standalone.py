"""Standalone sanity check for the Depth Anything V2 model, independent of
DimOS, the drone connection, or any camera pipeline. Run this directly:

    python test_depth_model_standalone.py path/to/any_indoor_photo.jpg

If you don't have a photo handy, run with no arguments and it generates a
synthetic test image (a gradient) -- not realistic depth output, but
confirms the model loads, runs, and returns the expected shape/dtype.

This isolates whether DroneDepthModule's model usage is correct, separate
from whether real camera frames are reaching it.
"""

import sys
import numpy as np
from PIL import Image as PILImage

MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def make_synthetic_image():
    """640x360 gradient, not a real scene, just exercises the pipeline."""
    w, h = 640, 360
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(200, 50, w, dtype=np.uint8)[None, :].repeat(h, axis=0)[:, :, 0] if False else 100
    arr[:, :, 2] = 150
    return PILImage.fromarray(arr)


def main():
    print(f"Loading {MODEL_NAME} ...")
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=MODEL_NAME)
    print("Model loaded.")

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Using real image: {img_path}")
        img = PILImage.open(img_path).convert("RGB")
    else:
        print("No image given -- using a synthetic gradient (sanity check only, not realistic depth)")
        img = make_synthetic_image()

    print(f"Input image size: {img.size}")

    import time
    t0 = time.time()
    result = pipe(img)
    elapsed = time.time() - t0

    # result["depth"] is a 0-255 normalized PIL image for visualization only.
    # result["predicted_depth"] is the actual metric tensor -- confirmed from
    # transformers' depth_estimation.py source. Already resized to match
    # the input image dimensions.
    depth_tensor = result["predicted_depth"]
    depth_np = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

    print(f"\nInference time: {elapsed:.2f}s")
    print(f"Output type (predicted_depth): {type(depth_tensor)}")
    print(f"Output array shape: {depth_np.shape}, dtype: {depth_np.dtype}")
    print(f"Depth value range: min={depth_np.min():.3f}, max={depth_np.max():.3f}, mean={depth_np.mean():.3f}")

    # Sanity checks
    assert depth_np.ndim == 2, f"expected 2D depth map, got {depth_np.ndim}D"
    assert depth_np.shape == (img.size[1], img.size[0]), \
        f"depth shape {depth_np.shape} doesn't match image (h,w)=({img.size[1]},{img.size[0]})"
    assert np.all(np.isfinite(depth_np)), "depth map contains non-finite values"
    assert depth_np.min() >= 0, f"depth should be non-negative, got min={depth_np.min()}"

    # For the "metric-indoor" variant, values should plausibly be in metres
    # for an indoor scene (typically 0.1m - 10m range).
    if depth_np.max() > 100:
        print("\nWARNING: max depth > 100 -- this does NOT look like metres. "
              "Check that you're using the *Metric* model variant, not relative depth.")
    else:
        print("\nDepth value range looks plausible for metric (metres) output.")

    print("\nAll sanity checks passed. Model API usage matches what DroneDepthModule expects.")


if __name__ == "__main__":
    main()
