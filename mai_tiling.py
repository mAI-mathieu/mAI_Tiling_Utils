import torch
import torch.nn.functional as F

class mAI_ImageToTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 8}),
            }
        }

    # We output a custom 'mAI_TILE_LAYOUT' so the untile node knows EXACTLY how to rebuild it
    RETURN_TYPES = ("IMAGE", "mAI_TILE_LAYOUT")
    RETURN_NAMES = ("tiles", "tile_layout")
    FUNCTION = "split"
    CATEGORY = "mAI/tiling"

    def split(self, image, tile_width, tile_height, overlap):
        B, H, W, C = image.shape
        
        # Ensure stride is at least 1 pixel
        stride_y = max(1, tile_height - overlap)
        stride_x = max(1, tile_width - overlap)

        # Pad the image if the requested tile size is actually larger than the image itself
        pad_y = max(0, tile_height - H)
        pad_x = max(0, tile_width - W)

        if pad_y > 0 or pad_x > 0:
            # PyTorch padding expects (B, C, H, W)
            image_permuted = image.permute(0, 3, 1, 2)
            padded = F.pad(image_permuted, (0, pad_x, 0, pad_y), mode='reflect')
            padded_image = padded.permute(0, 2, 3, 1)
        else:
            padded_image = image

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "mAI/tiling"

    def merge(self, tiles, tile_layout, blend_mode):
        # Read the blueprint
        B, H, W = tile_layout["B"], tile_layout["H"], tile_layout["W"]
        pad_y, pad_x = tile_layout["pad_y"], tile_layout["pad_x"]
        tile_width, tile_height = tile_layout["tile_width"], tile_layout["tile_height"]
        y_coords, x_coords = tile_layout["y_coords"], tile_layout["x_coords"]

        P_H, P_W = H + pad_y, W + pad_x
        C = tiles.shape[3]

        # Create empty canvases for the reconstructed image and the blending weights
        reconstructed = torch.zeros((B, P_H, P_W, C), dtype=torch.float32, device=tiles.device)
        weights = torch.zeros((B, P_H, P_W, 1), dtype=torch.float32, device=tiles.device)

        tile_idx = 0
        for b in range(B):
            for y in y_coords:
                for x in x_coords:
                    tile = tiles[tile_idx:tile_idx+1]

                    # Gradient feathering ensures seams are blended softly instead of sharp lines
                    if blend_mode == "gradient_feathering":
                        mask_y = torch.ones((tile_height, 1), device=tiles.device)
                        mask_x = torch.ones((1, tile_width), device=tiles.device)

                        # Create soft linear ramps for the edges
                        mask_y[:tile_height//2, 0] = torch.linspace(0, 1, tile_height//2, device=tiles.device)
                        mask_y[tile_height//2:, 0] = torch.linspace(1, 0, tile_height - tile_height//2, device=tiles.device)
                        
                        mask_x[0, :tile_width//2] = torch.linspace(0, 1, tile_width//2, device=tiles.device)
                        mask_x[0, tile_width//2:] = torch.linspace(1, 0, tile_width - tile_width//2, device=tiles.device)

                        # Do not fade the absolute outer edges of the whole image
                        if y == y_coords[0]: mask_y[:tile_height//2, 0] = 1.0
                        if y == y_coords[-1]: mask_y[tile_height//2:, 0] = 1.0
                        if x == x_coords[0]: mask_x[0, :tile_width//2] = 1.0
                        if x == x_coords[-1]: mask_x[0, tile_width//2:] = 1.0

                        mask = (mask_y * mask_x).unsqueeze(0).unsqueeze(3)
                    else:
                        mask = torch.ones((1, tile_height, tile_width, 1), device=tiles.device)

                    # Add the tile to the canvas and add its mask to the weight tracker
                    reconstructed[b:b+1, y:y+tile_height, x:x+tile_width, :] += tile * mask
                    weights[b:b+1, y:y+tile_height, x:x+tile_width, :] += mask
                    tile_idx += 1

        # Normalize the canvas by dividing the pixels by the weight map (this guarantees 100% perfect math)
        weights = torch.clamp(weights, min=1e-8)
        reconstructed /= weights

        # Crop out any padding that was added at the start to perfectly match the original input
        final_image = reconstructed[:, :H, :W, :]
        final_image = torch.clamp(final_image, 0.0, 1.0)
        
        return (final_image,)