# ComfyUI Crop and Paste Nodes

Two nodes: `CNPCrop` (crops image to mask bbox + pipes metadata) and `CNPPaste` (composites cropped image back onto original using pipe).

## Critical conventions

- **Node class name = registration key** in `NODE_CLASS_MAPPINGS`. Must match exactly.
- **PIPE dict** = `{"metadata": tensor[N, [x, y, w, h, nw, nh]], "original_images": tensor, "masks": tensor}`.
  - `masks` stores the **cropped/resized** masks (one per batch item), **not** the original full-image masks.
  - `nw, nh` are **stored but never read** by CNPPaste (only `x, y, w, h` used).
  - **Uniform crops assumed across batch** — `torch.stack` on cropped masks of differing sizes would fail.
- **CNPPaste blend**: `crop_img * mask_exp + canvas * (1.0 - mask_exp)` where `mask_exp = mask.unsqueeze(-1).expand(-1, -1, 3)`.
- **Empty mask**: CNPCrop passes full image/mask through with metadata `[0, 0, W, H, W, H]` (early continue in crop loop); in CNPPaste the all-zero mask makes the blend a no-op — there is **no** early continue there.
- **CNPPaste always resizes** `cropped_image` back to the original padded crop box `(w, h)` with LANCZOS (output may be any size, not just `nw, nh`), re-interpolates the mask to the actual resized shape, and slices channels to `:3` **before** the PIL conversion (`Image.fromarray(mode="RGB")` raises on 4-channel data).

## CNPPaste mask adjustment (`adjust_mask` helper)

- Widgets: `mask_grow` (INT −256..256: positive = dilate via `max_pool2d`, negative = erode via `-max_pool2d(-m)`), `mask_blur` (INT 0..256 Gaussian radius), `adjust_device` (`"auto"`/`"cpu"`/`"cuda"`, default `"auto"`). Defaults `0, 0` = legacy hard-mask behavior.
- Grow runs on `grow_device` (`auto` = CUDA if available; `"cuda"` falls back silently to CPU without CUDA) then returns to the input device — CPU and CUDA grow are bit-identical (4000² @ grow=64: ~9 s CPU vs ~17 ms CUDA). Blur stays PIL/CPU regardless.
- Applied **live at paste time** on `mask_resized_final` (final paste-box pixel size), grow before blur. Pipe is never modified — stateless, no accumulation across runs.
- Grow uses **two separable `max_pool2d` passes** (kernels `(1, 2r+1)` then `(2r+1, 1)`) — a single `(2r+1)²` kernel is O(k²)/pixel and pathologically slow at large radii (grow=64 on 1500²: ~60 s vs ~1 s separable). Result is bit-identical (square SE = Minkowski sum of the two bars).
- **Device discipline**: PIL roundtrips (`GaussianBlur`, crop resize) land on CPU — `adjust_mask` returns to the input device and `crop_img_resized` is moved to `canvas.device` before the blend. Never let mask and canvas tensors end up on different devices.
- Blur uses PIL `ImageFilter.GaussianBlur` — allowed here because soft values are the point; the "never PIL for masks" rule below applies only to **resizing**.

## Tensor shapes

| Type | Shape | dtype | Range |
|------|-------|-------|-------|
| IMAGE | (B, H, W, 3) or (H, W, 3) | float32 | [0, 1] |
| MASK | (B, H, W) or (H, W) | float32 | [0, 1] |
| PIPE metadata | (N, 6) | float32 | pixel coords |

Always unsqueeze batch dim for uniform handling. Use `.permute(2,0,1)` for CHW conversion (no-resize path only).

## Padding modes

- `padding_mode` dropdown: `"pixels"` (INT 0–1024) or `"percent"` (INT 0–100). Both INT widgets always present but only the active one is consumed.
- Percent mode computes `round(bbox_diag * padding_percent / 100.0)` where `bbox_diag = sqrt(w² + h²)`.

## Divisible_by centering

- `divisible_by` is a **string dropdown** (choices `"2","4","8","16","32","64"`), cast to int at runtime.
- Crop box is centered on mask bbox center. Half-extent from center is **rounded to nearest div_factor** (not ceil), then doubled: `hw = max(div_factor, round(half_w / div_factor) * div_factor)`, `w = 2 * hw`. Clamp to image bounds after centering.

## Mask interpolation

**Always** `torch.nn.functional.interpolate(..., mode="nearest")` when **resizing** masks. Never PIL resize — it interpolates float values and corrupts binary masks.

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

`WEB_DIRECTORY = "./web"` in `__init__.py` tells ComfyUI to serve a `web/` directory. The additional `PromptServer.instance` route is a fallback for older ComfyUI versions. `web/` directory does not currently exist — optional.
