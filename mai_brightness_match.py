import torch
import torch.nn.functional as F

class mAI_BrightnessMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "mode": ([
                        "local_luminance (Best for Tiles)", 
                        "local_color_and_luminance", 
                        "global_luminance", 
                        "global_color_and_brightness",
                        "simple_global_shift"
                    ],
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "local_filter_size": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8, "tooltip": "Only used for Local modes. Lower = matches broad lighting. Higher = matches smaller local details tighter."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    FUNCTION = "match"
    CATEGORY = "mAI/color"

    def match(self, image, reference, mode, strength, local_filter_size):
        if strength == 0.0:
            return (image,)
            
        B_img, H_tgt, W_tgt, C = image.shape
        B_ref = reference.shape[0]
        
        # Protect against batch size mismatches
        ref_to_use = reference.repeat(B_img, 1, 1, 1) if B_ref == 1 and B_img != 1 else reference
            
        # Helper conversions
        def rgb_to_ycbcr(img):
            y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            cb = -0.1687 * img[..., 0] - 0.3313 * img[..., 1] + 0.5 * img[..., 2] + 0.5
            cr = 0.5 * img[..., 0] - 0.4187 * img[..., 1] - 0.0813 * img[..., 2] + 0.5
            return torch.stack([y, cb, cr], dim=-1)

        def ycbcr_to_rgb(img):
            y, cb, cr = img[..., 0], img[..., 1], img[..., 2]
            cb = cb - 0.5
            cr = cr - 0.5
            r = y + 1.402 * cr
            g = y - 0.34414 * cb - 0.71414 * cr
            b = y + 1.772 * cb
            return torch.stack([r, g, b], dim=-1)

        result = image.clone()

        # --- LOCAL SPATIAL MATCHING (Frequency Separation) ---
        if "local" in mode:
            # 1. Resize reference to exactly match the upscaled target resolution
            ref_p = ref_to_use.permute(0, 3, 1, 2)
            ref_aligned = F.interpolate(ref_p, size=(H_tgt, W_tgt), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

            if "luminance" in mode:
                tgt_ycbcr = rgb_to_ycbcr(image)
                ref_ycbcr = rgb_to_ycbcr(ref_aligned)
                tgt_work = tgt_ycbcr[..., 0:1] # Extract just the Y (Luminance) channel
                ref_work = ref_ycbcr[..., 0:1]
            else:
                tgt_work = image
                ref_work = ref_aligned

            # 2. Create the "Lighting Maps" by downscaling then upscaling (Low-Pass Filter)
            tgt_work_p = tgt_work.permute(0, 3, 1, 2)
            ref_work_p = ref_work.permute(0, 3, 1, 2)

            tgt_down = F.interpolate(tgt_work_p, size=(local_filter_size, local_filter_size), mode='area')
            ref_down = F.interpolate(ref_work_p, size=(local_filter_size, local_filter_size), mode='area')

            tgt_lowpass = F.interpolate(tgt_down, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            ref_lowpass = F.interpolate(ref_down, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)

            # 3. Apply the specific lighting of the original region to the upscaled region
            matched_work = tgt_work + (ref_lowpass - tgt_lowpass)

            # 4. Recombine
            if "luminance" in mode:
                result_ycbcr = torch.cat([matched_work, tgt_ycbcr[..., 1:]], dim=-1)
                result = ycbcr_to_rgb(result_ycbcr)
            else:
                result = matched_work

        # --- GLOBAL MATCHING (Old Math) ---
        elif mode == "global_luminance":
            tgt_ycbcr = rgb_to_ycbcr(image)
            ref_ycbcr = rgb_to_ycbcr(ref_to_use)
            tgt_y, ref_y = tgt_ycbcr[..., 0:1], ref_ycbcr[..., 0:1]

            ref_mean, ref_std = ref_y.mean(dim=(1, 2), keepdim=True), ref_y.std(dim=(1, 2), keepdim=True) + 1e-6
            tgt_mean, tgt_std = tgt_y.mean(dim=(1, 2), keepdim=True), tgt_y.std(dim=(1, 2), keepdim=True) + 1e-6

            matched_y = (tgt_y - tgt_mean) / tgt_std * ref_std + ref_mean
            result_ycbcr = torch.cat([matched_y, tgt_ycbcr[..., 1:]], dim=-1)
            result = ycbcr_to_rgb(result_ycbcr)

        elif mode == "global_color_and_brightness":
            ref_mean, ref_std = ref_to_use.mean(dim=(1, 2), keepdim=True), ref_to_use.std(dim=(1, 2), keepdim=True) + 1e-6
            tgt_mean, tgt_std = image.mean(dim=(1, 2), keepdim=True), image.std(dim=(1, 2), keepdim=True) + 1e-6
            result = (image - tgt_mean) / tgt_std * ref_std + ref_mean

        elif mode == "simple_global_shift":
            tgt_ycbcr = rgb_to_ycbcr(image)
            ref_ycbcr = rgb_to_ycbcr(ref_to_use)
            ref_mean = ref_ycbcr[..., 0:1].mean(dim=(1, 2), keepdim=True)
            tgt_mean = tgt_ycbcr[..., 0:1].mean(dim=(1, 2), keepdim=True)
            
            tgt_ycbcr[..., 0:1] = tgt_ycbcr[..., 0:1] + (ref_mean - tgt_mean)
            result = ycbcr_to_rgb(tgt_ycbcr)

        # Apply strength and cleanly clamp to prevent artifacting
        final_image = torch.lerp(image, result, strength)
        final_image = torch.clamp(final_image, 0.0, 1.0)

        return (final_image,)