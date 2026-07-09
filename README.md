# ComfyUI-CNP-CropAndPaste

Two custom nodes for ComfyUI that let you crop images to mask bounding boxes and paste them back with alpha blending — handy for inpainting workflows, object isolation, and compositing.

## Nodes

- **CNP Crop** — Crops an image to the mask's bounding box. Supports pixel and percent-based padding, optional resize to a target longer side, and divisible-by centering for clean tile sizes.
- **CNP Paste** — Composites the cropped image back onto the original canvas using mask-based alpha blending.

## Installation

**ComfyUI Manager:**
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "CNP Crop And Paste"
4. Click Install

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dapa5900/ComfyUI-CNP-CropAndPaste.git
```

## License

MIT
