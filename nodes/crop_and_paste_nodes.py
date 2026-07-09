import math
import torch
import numpy as np
from PIL import Image


def find_bbox(mask):
    """Find bounding box of non-zero pixels in mask (H, W)."""
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    non_zero = torch.nonzero(mask)
    if non_zero.shape[0] == 0:
        return None, None, None, None, 0, 0
    y_min, x_min = torch.min(non_zero, dim=0).values
    y_max, x_max = torch.max(non_zero, dim=0).values
    return (
        int(x_min),
        int(x_max),
        int(y_min),
        int(y_max),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1),
    )


class CNPCrop:
    @classmethod
    def INPUT_TYPES(cls):
        div_options = ["2", "4", "8", "16", "32", "64"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_mode": (["pixels", "percent"], {"default": "pixels"}),
                "padding_pixels": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "padding_percent": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "resize_enabled": ("BOOLEAN", {"default": True}),
                "longer_side": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 2},
                ),
                "divisible_by": (div_options, {"default": "2"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "PIPE")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "pipe")
    FUNCTION = "crop_to_mask"
    CATEGORY = "Crop And Paste"

    def crop_to_mask(
        self, image, mask, padding_mode, padding_pixels, padding_percent, resize_enabled, longer_side, divisible_by
    ):
        div_factor = int(divisible_by)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        n_images, img_h, img_w, _ = image.shape

        result_images, result_masks, pipe_masks, pipe_data = [], [], [], []

        for b in range(n_images):
            img, msk = image[b], mask[b]
            x_min, x_max, y_min, y_max, raw_w, raw_h = find_bbox(msk)

            if x_min is None:
                pipe_data.append([0, 0, img_w, img_h, img_w, img_h])
                result_images.append(img)
                result_masks.append(msk)
                pipe_masks.append(msk)
                continue

            if padding_mode == "percent":
                bbox_diag = math.sqrt(raw_w**2 + raw_h**2)
                effective_padding = round(bbox_diag * padding_percent / 100.0)
            else:
                effective_padding = padding_pixels

            x1, y1 = max(0, x_min - effective_padding), max(0, y_min - effective_padding)
            x2, y2 = min(img_w, x_max + effective_padding + 1), min(img_h, y_max + effective_padding + 1)
            w_base, h_base = x2 - x1, y2 - y1

            w = max(div_factor, ((w_base + div_factor - 1) // div_factor) * div_factor)
            h = max(div_factor, ((h_base + div_factor - 1) // div_factor) * div_factor)

            bbox_cx = (x_min + x_max) / 2.0
            bbox_cy = (y_min + y_max) / 2.0

            x1 = round(bbox_cx - w / 2.0)
            y1 = round(bbox_cy - h / 2.0)

            x1 = max(0, min(x1, img_w - w))
            y1 = max(0, min(y1, img_h - h))

            target_img = img[y1 : y1 + h, x1 : x1 + w, :]
            target_mask = msk[y1 : y1 + h, x1 : x1 + w]

            if resize_enabled:
                pil_img = Image.fromarray(
                    (target_img.cpu().numpy() * 255).astype(np.uint8), mode="RGB"
                )
                scale = longer_side / max(w, h)
                nw, nh = (
                    max(div_factor, round((w * scale) / div_factor) * div_factor),
                    max(div_factor, round((h * scale) / div_factor) * div_factor),
                )
                target_img = (
                    torch.tensor(
                        np.array(pil_img.resize((nw, nh), Image.Resampling.LANCZOS)),
                        dtype=torch.float32,
                    )
                    / 255.0
                )
                mask_4d = target_mask.unsqueeze(0).unsqueeze(0)
                target_mask = (
                    torch.nn.functional.interpolate(
                        mask_4d, size=(nh, nw), mode="nearest"
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                nw, nh = (
                    max(div_factor, round(w / div_factor) * div_factor),
                    max(div_factor, round(h / div_factor) * div_factor),
                )
                target_img = (
                    torch.nn.functional.interpolate(
                        target_img.permute(2, 0, 1).unsqueeze(0),
                        size=(nh, nw),
                        mode="bilinear",
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                mask_4d = target_mask.unsqueeze(0).unsqueeze(0)
                target_mask = (
                    torch.nn.functional.interpolate(
                        mask_4d, size=(nh, nw), mode="nearest"
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            result_images.append(target_img)
            result_masks.append(target_mask)
            pipe_masks.append(target_mask)
            pipe_data.append([x1, y1, w, h, nw, nh])

        pipe = {
            "metadata": torch.tensor(pipe_data, dtype=torch.float32),
            "original_images": image,
            "masks": torch.stack(pipe_masks),
        }
        return (
            torch.stack(result_images, dim=0),
            torch.stack(result_masks, dim=0),
            pipe,
        )


class CNPPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cropped_image": ("IMAGE",), "pipe": ("PIPE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_from_pipe"
    CATEGORY = "Crop And Paste"

    def paste_from_pipe(self, cropped_image, pipe):
        meta, masks, orig_imgs = (
            pipe["metadata"],
            pipe["masks"],
            pipe["original_images"],
        )
        result = []
        for b in range(cropped_image.shape[0]):
            x, y, w, h, nw, nh = meta[b].int().tolist()

            # Resize to match original crop box (padded size)
            pil_img = Image.fromarray(
                (cropped_image[b].cpu().numpy() * 255).astype(np.uint8), mode="RGB"
            )
            crop_img_resized = (
                torch.tensor(
                    np.array(pil_img.resize((w, h), Image.Resampling.LANCZOS)),
                    dtype=torch.float32,
                )
                / 255.0
            )

            mask_resized = (
                torch.nn.functional.interpolate(
                    masks[b].unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

            canvas = orig_imgs[b].clone()

            # Correctly size mask and image
            if crop_img_resized.shape[2] == 4:
                crop_img_resized = crop_img_resized[:, :, :3]
            elif crop_img_resized.shape[2] != 3:
                crop_img_resized = crop_img_resized[:, :, :3]

            # Use actual resulting shapes
            actual_h, actual_w = crop_img_resized.shape[0], crop_img_resized.shape[1]
            mask_resized_final = (
                torch.nn.functional.interpolate(
                    mask_resized.unsqueeze(0).unsqueeze(0),
                    size=(actual_h, actual_w),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask_exp = mask_resized_final.unsqueeze(-1).expand(-1, -1, 3)

            # Blend
            canvas[y : y + actual_h, x : x + actual_w, :3] = (
                crop_img_resized * mask_exp
            ) + (canvas[y : y + actual_h, x : x + actual_w, :3] * (1.0 - mask_exp))
            result.append(canvas)

        return (torch.stack(result, dim=0),)


NODE_CLASS_MAPPINGS = {"CNPCrop": CNPCrop, "CNPPaste": CNPPaste}
NODE_DISPLAY_NAME_MAPPINGS = {"CNPCrop": "CNP Crop", "CNPPaste": "CNP Paste"}
