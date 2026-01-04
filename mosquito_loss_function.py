
import math
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import rasterio


@dataclass
class GaussianMuResult:
    x_bar: float
    y_bar: float
    s_x: float
    s_y: float
    A: float
    G_max: float


def _normal_pdf(z: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """1D Normal PDF evaluated at z."""
    coef = 1.0 / (math.sqrt(2.0 * math.pi) * sd)
    return coef * np.exp(-0.5 * ((z - mean) / sd) ** 2)


def build_mu_gaussian_from_population(
    pop_tif: str,
    out_mu_tif: str,
    mu0: float = 0.0015,
    *,
    center_mode: Literal["weighted_mean", "max_cell"] = "weighted_mean",
    A: Optional[float] = None,
    hotspot_multiplier: float = 1.02,
    coord_scale: float = 1.0,
    min_sd: float = 1e-6,
) -> GaussianMuResult:
    """
    Build a μ(x,y) raster from a population density raster using a separable Gaussian hotspot.
    """

    if mu0 <= 0:
        raise ValueError("mu0 must be positive.")
    if A is None and hotspot_multiplier <= 1.0:
        raise ValueError("hotspot_multiplier must be > 1.0 when A is not provided.")

    with rasterio.open(pop_tif) as src:
        pop_ma = src.read(1, masked=True).astype("float64")
        profile = src.profile.copy()
        transform = src.transform

        pop = pop_ma.filled(np.nan)
        valid = np.isfinite(pop)
        if not np.any(valid):
            raise ValueError("No valid pixels found (check nodata/mask).")

        w = np.where(valid, np.maximum(pop, 0.0), 0.0)
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("Sum of population weights is zero; cannot compute hotspot center/spread.")

        height, width = pop.shape
        cols = np.arange(width)
        rows = np.arange(height)
        x_centers = (transform.c + (cols + 0.5) * transform.a) * coord_scale
        y_centers = (transform.f + (rows + 0.5) * transform.e) * coord_scale  # transform.e often negative

        X = x_centers[None, :]  # (1, W)
        Y = y_centers[:, None]  # (H, 1)

        # Choose hotspot center
        if center_mode == "weighted_mean":
            x_bar = float((w * X).sum() / w_sum)
            y_bar = float((w * Y).sum() / w_sum)
        else:  # "max_cell"
            r, c = np.unravel_index(np.nanargmax(np.where(valid, pop, np.nan)), pop.shape)
            x_bar = float(x_centers[c])
            y_bar = float(y_centers[r])

        # Weighted standard deviations (spread)
        s_x = math.sqrt(float((w * (X - x_bar) ** 2).sum() / w_sum))
        s_y = math.sqrt(float((w * (Y - y_bar) ** 2).sum() / w_sum))

        # Prevent zero/near-zero SD (would break PDF)
        s_x = max(s_x, min_sd)
        s_y = max(s_y, min_sd)

        gx = _normal_pdf(x_centers, x_bar, s_x)  # (W,)
        gy = _normal_pdf(y_centers, y_bar, s_y)  # (H,)
        G = gy[:, None] * gx[None, :]            # (H, W)

        G = np.where(valid, G, np.nan)
        G_max = float(np.nanmax(G))
        if not np.isfinite(G_max) or G_max <= 0:
            raise ValueError("G_max is not finite/positive; check raster, coord_scale, and min_sd.")

        if A is None:
            A = float((hotspot_multiplier - 1.0) / G_max)
        else:
            A = float(A)

        mu = mu0 * (1.0 + A * G)

        out_nodata = src.nodata if src.nodata is not None else -99999.0
        mu_out = np.where(valid, mu, out_nodata).astype("float32")

        profile.update(dtype="float32", count=1, nodata=out_nodata)

        with rasterio.open(out_mu_tif, "w", **profile) as dst:
            dst.write(mu_out, 1)

    return GaussianMuResult(
        x_bar=x_bar,
        y_bar=y_bar,
        s_x=float(s_x),
        s_y=float(s_y),
        A=float(A),
        G_max=float(G_max),
    )
res = build_mu_gaussian_from_population(
    pop_tif="martinique_projected.tif",
    out_mu_tif="mu_control_hotspot_1.tif",
    mu0=0.0015,
    center_mode = "max_cell",
    A = 100,
    coord_scale=0.001,
)

print(res)
