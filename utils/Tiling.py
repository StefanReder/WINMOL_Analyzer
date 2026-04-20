from __future__ import annotations

from dataclasses import dataclass
from math import ceil


import rasterio
from rasterio.windows import Window


@dataclass
class TileJob:
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    hx0: int
    hy0: int
    hx1: int
    hy1: int

    @property
    def inner_window(self) -> Window:
        return Window(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)

    @property
    def halo_window(self) -> Window:
        return Window(
            self.hx0, self.hy0, self.hx1 - self.hx0, self.hy1 - self.hy0)


def meters_to_pixels(tile_overlap_m, pixel_size_x, pixel_size_y) -> int:
    px = max(abs(pixel_size_x) or 0.0, abs(pixel_size_y) or 0.0)
    if px <= 0.0:
        return 0
    return int(ceil(tile_overlap_m / px))


def build_tile_grid(
    width: int, height: int, tile_inner_px: int, halo_px: int
) -> list[TileJob]:

    tile_inner_px = max(1, int(tile_inner_px))
    halo_px = max(0, int(halo_px))
    jobs: list[TileJob] = []
    row = 0
    for y0 in range(0, int(height), tile_inner_px):
        col = 0
        y1 = min(y0 + tile_inner_px, int(height))
        for x0 in range(0, int(width), tile_inner_px):
            x1 = min(x0 + tile_inner_px, int(width))
            hx0 = max(0, x0 - halo_px)
            hy0 = max(0, y0 - halo_px)
            hx1 = min(int(width), x1 + halo_px)
            hy1 = min(int(height), y1 + halo_px)
            jobs.append(
                TileJob(
                    tile_id=f"raster_r{row:05d}_c{col:05d}",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    hx0=hx0,
                    hy0=hy0,
                    hx1=hx1,
                    hy1=hy1,
                )
            )
            col += 1
        row += 1
    return jobs


def tile_profile_from_parent(profile, window):
    out = profile.copy()
    out['width'] = int(window.width)
    out['height'] = int(window.height)
    out['transform'] = rasterio.windows.transform(window, profile['transform'])
    return out
