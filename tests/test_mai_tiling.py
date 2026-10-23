import unittest

import torch

from mai_tiling import mAI_ImageToTiles, mAI_TilesToImage


def _split(
    image,
    tile_width=512,
    tile_height=512,
    overlap=64,
    tile_divisor=8,
    minimize_tiles=False,
):
    return mAI_ImageToTiles().split(
        image,
        tile_width,
        tile_height,
        overlap,
        tile_divisor,
        minimize_tiles,
    )


def _fake_scaled_tiles(tiles, scale, fill_value=0.0):
    return torch.full(
        (
            tiles.shape[0],
            tiles.shape[1] * scale,
            tiles.shape[2] * scale,
            tiles.shape[3],
        ),
        fill_value,
        dtype=tiles.dtype,
        device=tiles.device,
    )


class TestMAITiling(unittest.TestCase):
    def test_tile_dimensions_must_be_divisible_by_tile_divisor(self):
        image = torch.zeros((1, 768, 768, 1), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "tile_width=500.*tile_divisor=64"):
            _split(image, tile_width=500, tile_height=512, tile_divisor=64)

    def test_minimize_tiles_adjusts_dimensions_to_reduce_tile_count(self):
        image = torch.zeros((1, 1536, 1536, 1), dtype=torch.float32)

        tiles, layout = _split(
            image,
            tile_width=512,
            tile_height=512,
            overlap=64,
            tile_divisor=64,
            minimize_tiles=True,
        )

        self.assertEqual(layout["tile_width"], 576)
        self.assertEqual(layout["tile_height"], 576)
        self.assertEqual(layout["x_coords"], [0, 512, 960])
        self.assertEqual(layout["y_coords"], [0, 512, 960])
        self.assertEqual(tiles.shape, (9, 576, 576, 1))

    def test_minimize_tiles_optimizes_each_axis_independently(self):
        image = torch.zeros((1, 768, 1536, 1), dtype=torch.float32)

        _, layout = _split(
            image,
            tile_width=512,
            tile_height=512,
            overlap=64,
            tile_divisor=64,
            minimize_tiles=True,
        )

        self.assertEqual(layout["tile_width"], 576)
        self.assertEqual(layout["tile_height"], 512)

    def test_512_tile_stitches_at_1x(self):
        image = torch.linspace(0.0, 1.0, 512 * 512, dtype=torch.float32).reshape(
            1, 512, 512, 1
        )
        tiles, layout = _split(image)

        result = mAI_TilesToImage().merge(tiles, layout, "hard_average")[0]

        self.assertEqual(result.shape, image.shape)
        torch.testing.assert_close(result, image, rtol=0, atol=0)

    def test_512_tiles_stitch_after_fake_4x_upscale_auto(self):
        image = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
        tiles, layout = _split(image)
        processed_tiles = _fake_scaled_tiles(tiles, 4, fill_value=1.0)

        result = mAI_TilesToImage().merge(
            processed_tiles, layout, "hard_average", "auto", 4.0
        )[0]

        self.assertEqual(result.shape, (1, 2048, 2048, 1))
        self.assertTrue(torch.all(result == 1.0))

    def test_non_square_image_fake_4x_tiles_has_scaled_final_size(self):
        image = torch.zeros((1, 768, 1280, 1), dtype=torch.float32)
        tiles, layout = _split(image)
        processed_tiles = _fake_scaled_tiles(tiles, 4)

        result = mAI_TilesToImage().merge(
            processed_tiles, layout, "gradient_feathering", "auto", 4.0
        )[0]

        self.assertEqual(result.shape, (1, 3072, 5120, 1))
        self.assertTrue(torch.isfinite(result).all())

    def test_image_smaller_than_tile_fake_4x_has_scaled_original_size(self):
        image = torch.zeros((1, 200, 300, 1), dtype=torch.float32)
        tiles, layout = _split(image)
        processed_tiles = _fake_scaled_tiles(tiles, 4)

        result = mAI_TilesToImage().merge(
            processed_tiles, layout, "hard_average", "auto", 4.0
        )[0]

        self.assertEqual(result.shape, (1, 800, 1200, 1))

    def test_tile_count_mismatch_raises_clear_error(self):
        image = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
        tiles, layout = _split(image)

        with self.assertRaisesRegex(ValueError, "Tile count mismatch"):
            mAI_TilesToImage().merge(
                tiles[:0], layout, "hard_average", "auto", 4.0
            )

    def test_non_proportional_processed_tile_size_raises_clear_error(self):
        image = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
        _, layout = _split(image)
        bad_tiles = torch.zeros((1, 1024, 2048, 1), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "not proportional"):
            mAI_TilesToImage().merge(
                bad_tiles, layout, "hard_average", "auto", 4.0
            )

    def test_manual_scale_validates_processed_tile_size(self):
        image = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
        _, layout = _split(image)
        bad_tiles = torch.zeros((1, 1024, 1024, 1), dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError, "expects processed tile size 2048x2048"
        ):
            mAI_TilesToImage().merge(
                bad_tiles, layout, "hard_average", "manual", 4.0
            )
