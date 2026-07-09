# ComfyUI Crop and Paste Nodes

Custom nodes for ComfyUI that provide crop and paste functionality for working with masks.

## Nodes

### CNP Crop
Crops an image to the bounding box of a mask, preserving metadata for pasting back later.

**Inputs:**
- `image` - Input image (HWC or BHWC tensor, float32 [0, 1])
- `mask` - Binary mask (HW or BHW tensor, float32 [0, 1])
- `padding_mode` - Padding mode: `"pixels"` (0–1024) or `"percent"` (0–100 of bbox diagonal)
- `padding_pixels` - Extra pixels around mask crop (used when mode is `"pixels"`)
- `padding_percent` - Extra padding as percentage of bbox diagonal (used when mode is `"percent"`)
- `resize_enabled` - Whether to resize crop to target longer side
- `longer_side` - Target size for longer dimension when resizing (256–4096)
- `divisible_by` - Ensure crop dimensions divisible by this factor (dropdown: 2, 4, 8, 16, 32, 64)

**Outputs:**
- `cropped_image` - Cropped image (may be resized)
- `cropped_mask` - Cropped mask (may be resized)
- `pipe` - Dict with `{metadata, original_images, masks}` for paste operation

### CNP Paste
Pastes a cropped image back onto the original using metadata from the crop operation. Uses mask-based alpha blending: the cropped image is composited onto the original canvas weighted by the mask.

**Inputs:**
- `cropped_image` - Image to paste
- `pipe` - Metadata from CNP Crop

**Outputs:**
- `image` - Result with cropped image blended onto original

## Technical Details

- **Padding modes**: Percent mode computes `round(bbox_diag * percent / 100.0)` where `bbox_diag = sqrt(w² + h²)`.
- **Divisible by**: Crop box is centered on mask bbox center. Half-extent is rounded to nearest factor, then doubled: `hw = max(factor, round(half / factor) * factor)`, `w = 2 * hw`. Clamped to image bounds.
- **Mask interpolation**: Always `torch.nn.functional.interpolate(mode="nearest")` — never PIL resize, which corrupts binary masks.
- **Image resize**: Uses `PIL.Image.Resampling.LANCZOS` when resize is enabled.

## Installation

Use ComfyUI Manager:
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "Crop and Paste"
4. Click Install

Or manually:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-crop-and-paste.git
```

## Dependencies

- torch>=2.0
- numpy
- pillow

## License

MIT
