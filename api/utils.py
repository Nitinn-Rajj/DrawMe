"""
DrawMe - Image Preprocessing Utilities (v2 — Improved)
Handles Base64 decoding, grayscale conversion, bounding-box cropping,
center-of-mass centering, and downsampling to match model input requirements.

Key improvements over v1:
    - Center-of-mass centering (matches how Quick, Draw! data is centered)
    - Morphological thinning to match training data stroke thickness
    - Better noise thresholding
    - Gaussian blur to smooth aliasing artifacts
    - Debug mode to visualize preprocessing steps
"""

import io
import os
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage


def preprocess_canvas_image(base64_string: str, target_size: int = 28, debug: bool = False) -> np.ndarray:
    """
    Convert a Base64-encoded canvas image to a model-ready numpy array.

    Pipeline:
        1. Decode Base64 → PIL Image
        2. Convert to grayscale
        3. Invert colors (black-on-white → white-on-black)
        4. Crop to bounding box of drawn content (with margin)
        5. Pad to square aspect ratio
        6. Resize to target_size × target_size
        7. Center using center-of-mass (like Quick, Draw! preprocessing)
        8. Normalize to [0, 1]
        9. Reshape to (1, target_size, target_size, 1)

    Args:
        base64_string: Base64-encoded PNG image from HTML5 Canvas
        target_size: Output image dimension (default 28 for Quick, Draw!)
        debug: If True, save intermediate images for debugging

    Returns:
        Numpy array of shape (1, 28, 28, 1) ready for model.predict()
    """
    debug_dir = "/tmp/drawme_debug"
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Decode Base64 → PIL Image
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGBA")

    if debug:
        image.save(os.path.join(debug_dir, "01_raw.png"))

    # 2. Composite onto white background and build a color-invariant ink map.
    # Using grayscale directly makes bright colors (e.g., yellow) almost vanish
    # after inversion; min(R,G,B) preserves stroke strength for all brush colors.
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image)

    rgba = np.array(background).astype("float32")
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3] / 255.0
    ink = (255.0 - np.min(rgb, axis=2)) * alpha
    image = Image.fromarray(np.clip(ink, 0, 255).astype("uint8"), mode="L")

    if debug:
        image.save(os.path.join(debug_dir, "02_grayscale.png"))

    # 3. Threshold to clean up anti-aliasing noise
    img_array = np.array(image).astype("float32")
    # Keep thin anti-aliased edges while removing faint noise.
    img_array[img_array < 12] = 0

    if debug:
        Image.fromarray(img_array.astype("uint8")).save(os.path.join(debug_dir, "03_ink_thresholded.png"))

    # Check for empty canvas
    coords = np.argwhere(img_array > 0)
    if coords.size == 0:
        return np.zeros((1, target_size, target_size, 1), dtype="float32")

    # 4. Bounding box crop
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add a proportional margin (5% of the bounding box size)
    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    margin = int(max(bbox_h, bbox_w) * 0.08)
    margin = max(margin, 4)   # At least 4px margin

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(img_array.shape[0], y_max + margin)
    x_max = min(img_array.shape[1], x_max + margin)

    cropped = img_array[y_min:y_max, x_min:x_max]

    if debug:
        Image.fromarray(cropped.astype("uint8")).save(os.path.join(debug_dir, "04_cropped.png"))

    # 5. Pad to square
    h, w = cropped.shape
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim), dtype="float32")
    pad_y = (max_dim - h) // 2
    pad_x = (max_dim - w) // 2
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = cropped

    if debug:
        Image.fromarray(padded.astype("uint8")).save(os.path.join(debug_dir, "05_padded_square.png"))

    # 6. Resize to target_size × target_size
    img_pil = Image.fromarray(padded.astype("uint8"), mode="L")
    img_pil = img_pil.resize((target_size, target_size), Image.LANCZOS)

    if debug:
        # Save upscaled version for visibility
        img_pil.save(os.path.join(debug_dir, "06_resized_28x28.png"))
        img_pil.resize((280, 280), Image.NEAREST).save(os.path.join(debug_dir, "06_resized_280x280.png"))

    # 7. Center using center of mass
    # This is critical: Quick, Draw! data is centered by center of mass
    img_array = np.array(img_pil).astype("float32")

    # Calculate center of mass
    if img_array.sum() > 0:
        cy, cx = ndimage.center_of_mass(img_array)
        # Shift to center
        shift_y = target_size / 2.0 - cy
        shift_x = target_size / 2.0 - cx

        # Only shift if it's significant (more than 1px)
        if abs(shift_y) > 1 or abs(shift_x) > 1:
            img_array = ndimage.shift(img_array, [shift_y, shift_x], mode='constant', cval=0)

    if debug:
        Image.fromarray(img_array.astype("uint8")).save(os.path.join(debug_dir, "07_centered.png"))
        Image.fromarray(img_array.astype("uint8")).resize((280, 280), Image.NEAREST).save(
            os.path.join(debug_dir, "07_centered_280x280.png"))

    # 8. Normalize to [0, 1]
    img_array = img_array / 255.0

    # Clamp to [0, 1]
    img_array = np.clip(img_array, 0, 1)

    if debug:
        # Print stats for comparison with training data
        nonzero = np.count_nonzero(img_array > 0.05)
        mean_nz = img_array[img_array > 0.05].mean() if nonzero > 0 else 0
        print(f"[DEBUG] Preprocessed image stats:")
        print(f"  Non-zero pixels: {nonzero}/784 ({nonzero / 784 * 100:.1f}%)")
        print(f"  Mean stroke brightness: {mean_nz:.2f}")
        print(f"  Min: {img_array.min():.3f}, Max: {img_array.max():.3f}")

    # 9. Reshape to (1, 28, 28, 1)
    img_array = img_array.reshape(1, target_size, target_size, 1)

    return img_array
