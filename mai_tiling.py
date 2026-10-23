from fractions import Fraction

import torch
import torch.nn.functional as F


SCALE_TOLERANCE = 1e-6
TILE_SIZE_ADJUSTMENT_FRACTION = 0.25


def _pad_for_tiling(image, pad_x, pad_y):
    if pad_y == 0 and pad_x == 0:
        return image

    image_permuted = image.permute(0, 3, 1, 2)
    height, width = image.shape[1], image.shape[2]
    mode = "reflect"
    if height <= 1 or width <= 1 or pad_y >= height or pad_x >= width:
        mode = "replicate"

    padded = F.pad(image_permuted, (0, pad_x, 0, pad_y), mode=mode)
    return padded.permute(0, 2, 3, 1)


def _require_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _tile_count(image_size, tile_size, overlap):
    if image_size <= tile_size:
        return 1

    stride = tile_size - overlap
    return (image_size - tile_size + stride - 1) // stride + 1


def _round_up_to_divisor(value, divisor):
    return ((value + divisor - 1) // divisor) * divisor


def _tile_size_candidates(requested_size, overlap, divisor):
    lower_bound = max(
        1,
        overlap + 1,
        int(requested_size * (1 - TILE_SIZE_ADJUSTMENT_FRACTION)),
    )
    upper_bound = int(requested_size * (1 + TILE_SIZE_ADJUSTMENT_FRACTION))
    first_candidate = _round_up_to_divisor(lower_bound, divisor)

    if first_candidate <= upper_bound:
        return range(first_candidate, upper_bound + 1, divisor)

    # A divisor larger than the requested search window still needs a valid
    # candidate. In that case, use the first divisible size at or above the
    # requested size and overlap.
    fallback = max(requested_size, overlap + 1)
    fallback = _round_up_to_divisor(fallback, divisor)
    return (fallback,)


def _resolve_tile_size(image_size, requested_size, overlap, divisor, minimize, axis_name):
    requested_size = _require_int(requested_size, f"tile_{axis_name}")
    overlap = _require_int(overlap, "overlap")
    divisor = _require_int(divisor, "tile_divisor")

    if requested_size <= 0:
        raise ValueError(f"tile_{axis_name} must be positive, got {requested_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if divisor <= 0:
        raise ValueError(f"tile_divisor must be positive, got {divisor}")

    if not minimize:
        # Preserve at least the requested capacity while making the tile size
        # valid for the selected model/latent divisor.
        return _round_up_to_divisor(max(requested_size, overlap + 1), divisor)

    candidates = _tile_size_candidates(requested_size, overlap, divisor)
    return min(
        candidates,
        key=lambda size: (
            _tile_count(image_size, size, overlap),
            abs(size - requested_size),
            size,
        ),
    )


def _scaled_int(value, scale, label):
    scaled = Fraction(value) * scale
    if scaled.denominator != 1:
        raise ValueError(
            f"Scale {float(scale):g} creates a non-integer {label}: "
            f"{value} * {float(scale):g} = {float(scaled):g}"
        )
    return scaled.numerator


def _manual_scale_fraction(manual_scale):
    try:
        scale = Fraction(str(float(manual_scale))).limit_denominator(1000000)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"manual_scale must be a positive number, got {manual_scale!r}") from exc

    if scale <= 0:
        raise ValueError(f"manual_scale must be positive, got {manual_scale!r}")
    return scale


def _validate_layout_coords(axis_name, coords, padded_size, tile_size):
    if not isinstance(coords, (list, tuple)) or not coords:
        raise ValueError(f"tile_layout {axis_name}_coords must be a non-empty list")

    clean_coords = []
    for index, coord in enumerate(coords):
        coord = _require_int(coord, f"tile_layout {axis_name}_coords[{index}]")
        if coord < 0 or coord + tile_size > padded_size:
            raise ValueError(
                f"tile_layout {axis_name}_coords[{index}]={coord} is outside the "
                f"padded {axis_name} size {padded_size} for tile size {tile_size}"
            )
        if clean_coords:
            step = coord - clean_coords[-1]
            if step <= 0:
                raise ValueError(
                    f"Invalid tile layout overlap/coordinates: {axis_name}_coords "
                    "must be strictly increasing"
                )
            if step > tile_size:
                raise ValueError(
                    f"Invalid tile layout overlap/coordinates: consecutive "
                    f"{axis_name}_coords leave a gap larger than tile size {tile_size}"
                )
        clean_coords.append(coord)

    expected_last = padded_size - tile_size
    if clean_coords[0] != 0 or clean_coords[-1] != expected_last:
        raise ValueError(
            f"tile_layout {axis_name}_coords must cover the full padded {axis_name} "
            f"range from 0 to {expected_last}"
        )
    return clean_coords


def _resolve_scale(
    tile_width,
    tile_height,
    processed_tile_width,
    processed_tile_height,
    scale_mode,
    manual_scale,
):
    if processed_tile_width <= 0 or processed_tile_height <= 0:
        raise ValueError(
            f"Processed tile size must be positive, got "
            f"{processed_tile_width}x{processed_tile_height}"
        )

    if scale_mode == "auto":
        scale_x = Fraction(processed_tile_width, tile_width)
        scale_y = Fraction(processed_tile_height, tile_height)
        if abs(float(scale_x) - float(scale_y)) > SCALE_TOLERANCE:
            raise ValueError(
                "Processed tile size is not proportional to the original tile size: "
                f"original {tile_width}x{tile_height}, processed "
                f"{processed_tile_width}x{processed_tile_height}, "
                f"scale_x={float(scale_x):g}, scale_y={float(scale_y):g}"
            )
        return scale_x

    if scale_mode == "manual":
        scale = _manual_scale_fraction(manual_scale)
        expected_width = Fraction(tile_width) * scale
        expected_height = Fraction(tile_height) * scale
        if expected_width.denominator != 1 or expected_height.denominator != 1:
            raise ValueError(
                f"manual_scale {float(scale):g} creates a non-integer expected tile "
                f"size from original {tile_width}x{tile_height}"
            )

        expected_width = expected_width.numerator
        expected_height = expected_height.numerator
        if (
            processed_tile_width != expected_width
            or processed_tile_height != expected_height
        ):
            raise ValueError(
                f"manual_scale {float(scale):g} expects processed tile size "
                f"{expected_width}x{expected_height} from original "
                f"{tile_width}x{tile_height}, but received "
                f"{processed_tile_width}x{processed_tile_height}"
            )
        return scale

    raise ValueError(f"scale_mode must be 'auto' or 'manual', got {scale_mode!r}")


def _gradient_mask(tile_height, tile_width, y, x, y_coords, x_coords, device):
    mask_y = torch.ones((tile_height, 1), device=device)
    mask_x = torch.ones((1, tile_width), device=device)

    y_mid = tile_height // 2
    x_mid = tile_width // 2

    mask_y[:y_mid, 0] = torch.linspace(0, 1, y_mid, device=device)
    mask_y[y_mid:, 0] = torch.linspace(1, 0, tile_height - y_mid, device=device)

    mask_x[0, :x_mid] = torch.linspace(0, 1, x_mid, device=device)
    mask_x[0, x_mid:] = torch.linspace(1, 0, tile_width - x_mid, device=device)

    if y == y_coords[0]:
        mask_y[:y_mid, 0] = 1.0
    if y == y_coords[-1]:
        mask_y[y_mid:, 0] = 1.0
    if x == x_coords[0]:
        mask_x[0, :x_mid] = 1.0
    if x == x_coords[-1]:
        mask_x[0, x_mid:] = 1.0

    return (mask_y * mask_x).unsqueeze(0).unsqueeze(3)

class mAI_ImageToTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 8}),
                "tile_divisor": ("INT", {"default": 8, "min": 1, "max": 8192, "step": 1}),
                "minimize_tiles": ("BOOLEAN", {"default": False}),
            }
        }

    # We output a custom 'mAI_TILE_LAYOUT' so the untile node knows EXACTLY how to rebuild it
    RETURN_TYPES = ("IMAGE", "mAI_TILE_LAYOUT")
    RETURN_NAMES = ("tiles", "tile_layout")
    FUNCTION = "split"
    CATEGORY = "mAI/tiling"

    def split(
        self,
        image,
        tile_width,
        tile_height,
        overlap,
        tile_divisor=8,
        minimize_tiles=False,
    ):
        B, H, W, C = image.shape

        requested_tile_width = _require_int(tile_width, "tile_width")
        requested_tile_height = _require_int(tile_height, "tile_height")
        overlap = _require_int(overlap, "overlap")
        tile_divisor = _require_int(tile_divisor, "tile_divisor")
        tile_width = _resolve_tile_size(
            W, requested_tile_width, overlap, tile_divisor, minimize_tiles, "width"
        )
        tile_height = _resolve_tile_size(
            H, requested_tile_height, overlap, tile_divisor, minimize_tiles, "height"
        )
        
        # Ensure stride is at least 1 pixel
        stride_y = max(1, tile_height - overlap)
        stride_x = max(1, tile_width - overlap)

        # Pad the image if the requested tile size is actually larger than the image itself
        pad_y = max(0, tile_height - H)
        pad_x = max(0, tile_width - W)

        padded_image = _pad_for_tiling(image, pad_x, pad_y)

        P_H, P_W = padded_image.shape[1], padded_image.shape[2]

        # Calculate exact Y and X starting coordinates for each tile
        y_coords = list(range(0, P_H - tile_height + 1, stride_y))
        # Force the last tile to hit the exact bottom edge to guarantee full coverage
        if not y_coords or y_coords[-1] != P_H - tile_height:
            y_coords.append(P_H - tile_height)

        x_coords = list(range(0, P_W - tile_width + 1, stride_x))
        # Force the last tile to hit the exact right edge
        if not x_coords or x_coords[-1] != P_W - tile_width:
            x_coords.append(P_W - tile_width)

        # Extract the tiles
        tiles =[]
        for b in range(B):
            for y in y_coords:
                for x in x_coords:
                    tiles.append(padded_image[b:b+1, y:y+tile_height, x:x+tile_width, :])

        # Combine into a standard ComfyUI Image Batch
        tiles_tensor = torch.cat(tiles, dim=0)

        # Create the blueprint for the untile node
        layout = {
            "B": B, "H": H, "W": W,
            "pad_y": pad_y, "pad_x": pad_x,
            "tile_width": tile_width, "tile_height": tile_height,
            "requested_tile_width": requested_tile_width,
            "requested_tile_height": requested_tile_height,
            "tile_divisor": tile_divisor,
            "minimize_tiles": bool(minimize_tiles),
            "original_tile_width": tile_width,
            "original_tile_height": tile_height,
            "expected_input_tile_width": tile_width,
            "expected_input_tile_height": tile_height,
            "y_coords": y_coords, "x_coords": x_coords
        }

        return (tiles_tensor, layout)


class mAI_TilesToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_layout": ("mAI_TILE_LAYOUT",),
                "blend_mode": (["gradient_feathering", "hard_average"],),
                "scale_mode": (["auto", "manual"], {"default": "auto"}),
                "manual_scale": ("FLOAT", {"default": 4.0, "min": 0.25, "max": 16.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "mAI/tiling"

    def merge(self, tiles, tile_layout, blend_mode, scale_mode="auto", manual_scale=4.0):
        if tiles.ndim != 4:
            raise ValueError(
                f"tiles must be a ComfyUI IMAGE tensor shaped "
                f"[batch, height, width, channels], got shape {tuple(tiles.shape)}"
            )

        # Read the blueprint
        B, H, W = tile_layout["B"], tile_layout["H"], tile_layout["W"]
        pad_y, pad_x = tile_layout["pad_y"], tile_layout["pad_x"]
        tile_width, tile_height = tile_layout["tile_width"], tile_layout["tile_height"]
        y_coords, x_coords = tile_layout["y_coords"], tile_layout["x_coords"]

        B = _require_int(B, "tile_layout B")
        H = _require_int(H, "tile_layout H")
        W = _require_int(W, "tile_layout W")
        pad_y = _require_int(pad_y, "tile_layout pad_y")
        pad_x = _require_int(pad_x, "tile_layout pad_x")
        tile_width = _require_int(tile_width, "tile_layout tile_width")
        tile_height = _require_int(tile_height, "tile_layout tile_height")

        if B <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"tile_layout B, H, and W must be positive, got {B}, {H}, {W}")
        if pad_y < 0 or pad_x < 0:
            raise ValueError(f"tile_layout padding must be non-negative, got {pad_x}x{pad_y}")
        if tile_width <= 0 or tile_height <= 0:
            raise ValueError(
                f"tile_layout tile size must be positive, got {tile_width}x{tile_height}"
            )

        P_H, P_W = H + pad_y, W + pad_x
        y_coords = _validate_layout_coords("y", y_coords, P_H, tile_height)
        x_coords = _validate_layout_coords("x", x_coords, P_W, tile_width)

        expected_tile_count = B * len(y_coords) * len(x_coords)
        actual_tile_count = tiles.shape[0]
        if actual_tile_count != expected_tile_count:
            raise ValueError(
                f"Tile count mismatch: tile_layout expects {expected_tile_count} "
                f"tiles ({B} batch x {len(y_coords)} rows x {len(x_coords)} cols), "
                f"but tiles input contains {actual_tile_count}"
            )

        processed_tile_height = tiles.shape[1]
        processed_tile_width = tiles.shape[2]
        scale = _resolve_scale(
            tile_width,
            tile_height,
            processed_tile_width,
            processed_tile_height,
            scale_mode,
            manual_scale,
        )

        scaled_p_h = _scaled_int(P_H, scale, "padded image height")
        scaled_p_w = _scaled_int(P_W, scale, "padded image width")
        final_h = _scaled_int(H, scale, "final image height")
        final_w = _scaled_int(W, scale, "final image width")
        C = tiles.shape[3]

        # Create empty canvases for the reconstructed image and the blending weights
        reconstructed = torch.zeros((B, scaled_p_h, scaled_p_w, C), dtype=torch.float32, device=tiles.device)
        weights = torch.zeros((B, scaled_p_h, scaled_p_w, 1), dtype=torch.float32, device=tiles.device)

        tile_idx = 0
        for b in range(B):
            for y in y_coords:
                for x in x_coords:
                    tile = tiles[tile_idx:tile_idx+1]
                    scaled_y = _scaled_int(y, scale, "tile y coordinate")
                    scaled_x = _scaled_int(x, scale, "tile x coordinate")

                    # Gradient feathering ensures seams are blended softly instead of sharp lines
                    if blend_mode == "gradient_feathering":
                        mask = _gradient_mask(
                            processed_tile_height,
                            processed_tile_width,
                            y,
                            x,
                            y_coords,
                            x_coords,
                            tiles.device,
                        )
                    else:
                        mask = torch.ones((1, processed_tile_height, processed_tile_width, 1), device=tiles.device)

                    # Add the tile to the canvas and add its mask to the weight tracker
                    y_end = scaled_y + processed_tile_height
                    x_end = scaled_x + processed_tile_width
                    reconstructed[b:b+1, scaled_y:y_end, scaled_x:x_end, :] += tile * mask
                    weights[b:b+1, scaled_y:y_end, scaled_x:x_end, :] += mask
                    tile_idx += 1

        # Normalize the canvas by dividing the pixels by the weight map (this guarantees 100% perfect math)
        weights = torch.clamp(weights, min=1e-8)
        reconstructed /= weights

        # Crop out any padding that was added at the start to perfectly match the original input
        final_image = reconstructed[:, :final_h, :final_w, :]
        final_image = torch.clamp(final_image, 0.0, 1.0)
        
        return (final_image,)
