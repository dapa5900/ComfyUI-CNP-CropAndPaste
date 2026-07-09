# ComfyUI-DPNodes

Custom nodes for ComfyUI that provide crop and paste functionality for working with masks.

## Nodes

### CNP Crop
Crops an image to the bounding box of a mask, preserving metadata for pasting back later.

**Inputs:**
- `image` - Input image (HWC tensor)
- `mask` - Binary mask (HW tensor)
- `padding` - Extra pixels around mask in crop
- `resize_enabled` - Whether to resize crop to target longer side
- `longer_side` - Target size for longer dimension when resizing
- `divisible_by` - Ensure crop dimensions divisible by this factor (2, 4, 8, 16, 32, 64)

**Outputs:**
- `cropped_image` - Cropped image
- `cropped_mask` - Cropped mask
- `pipe` - Metadata for paste operation

### CNP Paste
Pastes a cropped image back onto the original using metadata from the crop operation.

**Inputs:**
- `cropped_image` - Image to paste
- `pipe` - Metadata from CNP Crop

**Outputs:**
- `image` - Result with cropped image blended onto original

## Installation

Use ComfyUI Manager:
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "DPNodes"
4. Click Install

Or manually:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-dpnodes.git
python main.py
```

## Dependencies

- torch>=2.0
- numpy
- pillow
- comfyui-custom-nodes>=1.0.0

## License

MIT