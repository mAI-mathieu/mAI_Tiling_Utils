# mAI Tiling Utils

ComfyUI custom nodes for splitting images into overlapping tiles and stitching
processed tiles back into a full image.

## Nodes

```text
mAI Image To Tiles
mAI Tiles To Image
```

Both nodes are registered under:

```text
mAI/tiling
```

## Scaled Tile Stitching

`mAI Tiles To Image` supports processed tiles that are larger or smaller than
the original tiles produced by `mAI Image To Tiles`.

Example 4x workflow:

```text
Input Image
-> mAI Image To Tiles, tile size 512x512, overlap 64
-> custom PiD/upscale workflow, output tile size 2048x2048
-> mAI Tiles To Image, scale_mode auto
-> final stitched 4x image
```

Recommended settings:

```text
tile_width: 512
tile_height: 512
overlap: 64 or 128
tile_divisor: 8
minimize_tiles: false
blend_mode: gradient_feathering
scale_mode: auto
```

`tile_divisor` requires the effective tile width and height to be divisible by
the selected number. For example, with `tile_divisor: 8`, a tile size of
`512x512` is valid, while `510x512` is rejected.

When `minimize_tiles` is enabled, the node adjusts each tile dimension in a
bounded range around the requested size (up to one overlap width in either
direction). It chooses divisible dimensions that minimize the number of tile
rows and columns, while preferring dimensions closest to the requested values.
If no divisible size exists in that range, it rounds up to the next valid
divisible size. The selected dimensions are stored in `tile_layout` and used
automatically by the stitcher.

The stitcher reads the original tile size from `tile_layout` and the processed
tile size from the incoming `tiles` tensor. In `auto` mode it detects the scale
factor from those sizes. In `manual` mode it uses `manual_scale` and validates
that the processed tile dimensions match.

The stitcher does not resize, crop, or stretch tiles. It only places and blends
the processed tiles it receives.

## Requirements

Processed tiles must keep the same aspect ratio as the original tiles. For
example, tiles split at `512x512` can be stitched after processing to
`2048x2048`, but not to `2048x1024`.

The scaled output dimensions must be exact integers. If the tile scale would
produce fractional canvas, crop, or tile-coordinate sizes, the stitcher raises a
clear error instead of guessing.

Overlap is strongly recommended to reduce seams. Use `gradient_feathering` for
soft blending or `hard_average` for direct averaging in overlapping regions.
