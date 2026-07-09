# ComfyUI Crop and Paste Nodes

Two nodes: `CNPCrop` (crops image to mask bbox + pipes metadata) and `CNPPaste` (composites cropped image back onto original using pipe).

## Critical conventions

- **Node class name = registration key** in `NODE_CLASS_MAPPINGS`. Must match exactly.
- **PIPE dict** = `{"metadata": tensor[N, [x, y, w, h, nw, nh]], "original_images": tensor, "masks": tensor}`.
  - `nw, nh` are **stored but never read** by CNPPaste (only `x, y, w, h` used).
- **CNPPaste blend**: `crop_img * mask_exp + canvas * (1.0 - mask_exp)`.
- **Empty mask** → return full image unchanged (early continue in loop).

## Tensor shapes

| Type | Shape | dtype | Range |
|------|-------|-------|-------|
| IMAGE | (B, H, W, 3) or (H, W, 3) | float32 | [0, 1] |
| MASK | (B, H, W) or (H, W) | float32 | [0, 1] |
| PIPE metadata | (N, 6) | float32 | pixel coords |

Always unsqueeze batch dim for uniform handling. Use `.permute(2,0,1)` for CHW conversion.

## Padding modes

- `padding_mode` dropdown: `"pixels"` (INT 0–1024) or `"percent"` (INT 0–100).
- Percent mode computes `round(bbox_diag * padding_percent / 100.0)` where `bbox_diag = sqrt(w² + h²)`.

## Divisible_by centering

Crop box is **centered on mask bbox center** (`(x_min + x_max) / 2.0`) after ceiling dimensions to the next multiple. This makes crop position resolution-independent. Clamp to image bounds after centering.

## Mask interpolation

**Always** `torch.nn.functional.interpolate(..., mode="nearest")`. Never PIL resize — it interpolates float values and corrupts binary masks.

## Image resize quality

Use `PIL.Image.Resampling.LANCZOS` when `resize_enabled=True`. Convert via `(tensor * 255).astype(np.uint8)` → PIL → `np.array(...) / 255.0` → tensor.

## Commands

```powershell
# Syntax check only (no test suite, no lint config committed)
python -m py_compile nodes/crop_and_paste_nodes.py
python -m py_compile __init__.py
```

**No tests exist** — must verify manually in ComfyUI. No `ruff`, `mypy`, or `pre-commit` config in `pyproject.toml`.

## Web assets

`web/` directory served at `/cpweb` via `__init__.py` (conditional on `PromptServer.instance`). Optional.
