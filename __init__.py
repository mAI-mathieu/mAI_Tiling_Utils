from .mai_tiling import mAI_ImageToTiles, mAI_TilesToImage

NODE_CLASS_MAPPINGS = {
    "mAI_ImageToTiles": mAI_ImageToTiles,
    "mAI_TilesToImage": mAI_TilesToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mAI_ImageToTiles": "mAI Image To Tiles",
    "mAI_TilesToImage": "mAI Tiles To Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
