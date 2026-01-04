
from __future__ import annotations
import math
import csv
import datetime as dt
from dataclasses import dataclass
import numpy as np
from fipy import (
    CellVariable,
    Grid2D,
    TransientTerm,
    DiffusionTerm,
    ImplicitSourceTerm,
)
from pathlib import Path

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, transform as crs_transform
    from rasterio.transform import rowcol
except Exception as e:
    raise SystemExit(
        "This script requires 'rasterio'. Install with:\n"
        "  pip install rasterio\n\n"
        f"Import error: {e}"
    )

@dataclass
class Params:
    delta1: float = 0.2
    delta2: float = 0.2
    lam: float = 1.0
    mu1: float = 0.5
    sigma1: float = 5e-6
    sigma2: float = 0.6
    beta_scale: float = 300.0
    beta_mu: float = 6 
    beta_sigma: float = 10 
    beta_lambda: float = 0.2
    output_dir: str = "outputs"

    weeks_to_simulate: int = 50
    dt_week: float = 0.05
    picard_sweeps: int = 20
    sweep_tol: float = 1e-6

    Hi0_total: float = 57.0

    init_mode: str = "point_seed"

    gaussian_sigma_km: float = 2.0
    gaussian_center_xy_km: tuple[float, float] | None = None

    seed_coord: tuple[float, float] = (709053.0, 1617343.0)
    seed_crs: str | None = None
    seed_cases: float | None = None  
    seed_smooth: bool = True
    seed_sigma_km: float = 2

    mosquito_init_mode: str = "ratio_people"
    mosquitoes_per_person: float = 2.0
    Vi0_multiplier: float = 10.0 
    use_avg_beta: bool = True 
    min_mosquito_density: float = 50.0 
    mask_sink: float = 1e6

    export_weeks: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28)
    export_prefix: str = "Hi_week"
    export_dtype: str = "float32"
    export_nodata: float = -9999.0

def emg_beta(t_week: float, p: Params) -> float:
    """Exponentially modified Gaussian for seasonal breeding rate β(t)."""
    lam = p.beta_lambda
    mu = p.beta_mu
    sig = p.beta_sigma
    z = (lam * sig * sig + mu - t_week) / (math.sqrt(2.0) * sig)
    emg = (lam / 2.0) * math.exp((lam / 2.0) * (2.0 * mu + lam * sig * sig - 2.0 * t_week)) * math.erfc(z)
    return p.beta_scale * emg

def read_tif(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
        crs = src.crs
    return arr, profile, transform, nodata, crs

def reproject_to_match(src_arr, src_profile, dst_profile):
    """Reproject src_arr onto the grid described by dst_profile."""
    dst_nodata = dst_profile.get("nodata", None)
    if dst_nodata is None:
        dst_nodata = src_profile.get("nodata", None)
    if dst_nodata is None:
        dst_nodata = 0.0

    dst_arr = np.full((dst_profile["height"], dst_profile["width"]), dst_nodata, dtype="float64")

    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.nearest,
        src_nodata=src_profile.get("nodata", None),
        dst_nodata=dst_nodata,
    )
    return dst_arr

def write_geotiff(path: str | Path, arr_topdown: np.ndarray, ref_profile: dict, nodata_val: float, dtype: str = "float32"):
    prof = ref_profile.copy()
    prof.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata_val, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr_topdown.astype(dtype), 1)



def assert_seed_on_land(row: int, col: int, landmask_topdown: np.ndarray, x_crs: float, y_crs: float):
    H, W = landmask_topdown.shape

    if row < 0 or row >= H or col < 0 or col >= W:
        raise ValueError(
            f"Seed coordinate is outside the raster extent. "
            f"(x, y)=({x_crs:.3f}, {y_crs:.3f}) maps to (row, col)=({row}, {col}), "
            f"but raster shape is (H, W)=({H}, {W})."
        )

    # 2) Strict land check
    if landmask_topdown[row, col] <= 0.5:
        raise ValueError(
            f"Seed coordinate is not on land (landmask<=0.5). "
            f"(x, y)=({x_crs:.3f}, {y_crs:.3f}) -> (row, col)=({row}, {col}). "
            f"Choose a seed point inside the island."
        )

def build_initial_Hi_point_seed(
    Hu_transform,
    Hu_crs,
    Hu_shape,
    landmask_topdown: np.ndarray,
    landmask_int: np.ndarray,
    cell_area_km2: float,
    mesh: Grid2D,
    p: Params,
    export_path: str | None = None,
    export_profile: dict | None = None,
    export_nodata: float = -9999.0,
    export_dtype: str = "float32",
):
    """Place initial cases at a user-specified coordinate; optionally export Hi(0) as a GeoTIFF."""
    sx, sy = p.seed_coord
    src_crs = p.seed_crs if p.seed_crs is not None else Hu_crs
    dst_crs = Hu_crs

    if str(src_crs) != str(dst_crs):
        xs, ys = crs_transform(src_crs, dst_crs, [sx], [sy])
        x_crs, y_crs = xs[0], ys[0]
    else:
        x_crs, y_crs = sx, sy

    r, c = rowcol(Hu_transform, x_crs, y_crs)
    r, c = int(r), int(c)

    assert_seed_on_land(r, c, landmask_topdown, x_crs, y_crs)

    total_cases = float(p.Hi0_total if p.seed_cases is None else p.seed_cases)

    Ny, Nx = Hu_shape

    if not p.seed_smooth:
        Hi0_top = np.zeros((Ny, Nx), dtype="float64")
        Hi0_top[r, c] = total_cases / cell_area_km2  # density (cases / km^2)
        Hi0_int = np.flipud(Hi0_top)

    else:
        r_int = (Ny - 1) - r
        c_int = c

        dx_km = float(mesh.dx)
        dy_km = float(mesh.dy)
        x0 = (c_int + 0.5) * dx_km
        y0 = (r_int + 0.5) * dy_km

        x, y = mesh.cellCenters[0], mesh.cellCenters[1]
        g = np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * p.seed_sigma_km * p.seed_sigma_km)))

        w = np.array(g).reshape((Ny, Nx)) * landmask_int
        denom = np.sum(w * cell_area_km2)
        if denom <= 0:
            raise ValueError("Seed smoothing: weights sum to zero; check landmask / sigma / seed location.")

        Hi0_int = total_cases * w / denom

        Hi0_top = np.flipud(Hi0_int)

    if export_path is not None:
        Hi0_top_out = np.where(landmask_topdown > 0.5, Hi0_top, export_nodata)

        if export_profile is None:
            export_profile = {
                "driver": "GTiff",
                "height": Ny,
                "width": Nx,
                "count": 1,
                "dtype": export_dtype,
                "crs": Hu_crs,
                "transform": Hu_transform,
                "nodata": export_nodata,
                "compress": "lzw",
            }
        else:
            export_profile = export_profile.copy()
            export_profile.update(
                driver="GTiff",
                count=1,
                dtype=export_dtype,
                nodata=export_nodata,
                compress="lzw",
            )

        with rasterio.open(export_path, "w", **export_profile) as dst:
            dst.write(Hi0_top_out.astype(export_dtype), 1)
    return Hi0_int



def clean_mu(mu_top: np.ndarray, mu_nodata: float | None, land_top: np.ndarray) -> tuple[np.ndarray, float]:
    mu = mu_top.copy()
    if mu_nodata is not None:
        mu[np.isclose(mu, mu_nodata)] = np.nan
    mu[~np.isfinite(mu)] = np.nan
    valid_land = np.isfinite(mu) & (land_top > 0.5) & (mu > 0.0)
    if np.count_nonzero(valid_land) == 0:
        raise ValueError("μ raster has no valid positive values on land after masking nodata.")
    mu_floor = float(max(1e-6, np.nanpercentile(mu[valid_land], 1)))
    mu = np.where(np.isfinite(mu) & (mu > 0.0), mu, mu_floor)
    mu = np.maximum(mu, mu_floor)
    return mu, mu_floor


def main():
    p = Params()
    Hu_path = "Dataset/martinique_1km.tif"
    mu_path = "Dataset/mu_control_hotspot_1km.tif"
    Hu_top, Hu_prof, Hu_transform, Hu_nodata, Hu_crs = read_tif(Hu_path)
    mu_top, mu_prof, mu_transform, mu_nodata, mu_crs = read_tif(mu_path)
    out_dir = Path(p.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (mu_top.shape != Hu_top.shape) or (mu_crs != Hu_crs) or (mu_transform != Hu_transform):
        dst_prof = {
            "height": Hu_top.shape[0],
            "width": Hu_top.shape[1],
            "crs": Hu_crs,
            "transform": Hu_transform,
            "nodata": mu_nodata if mu_nodata is not None else -99999.0,
        }
        src_prof = {
            "crs": mu_crs,
            "transform": mu_transform,
            "nodata": mu_nodata,
        }
        mu_top = reproject_to_match(mu_top, src_prof, dst_prof)
        mu_nodata = dst_prof["nodata"]

    land_top = np.ones_like(Hu_top, dtype="float64")
    if Hu_nodata is not None:
        land_top[np.isclose(Hu_top, Hu_nodata)] = 0.0
    land_top[Hu_top <= 0] = 0.0

    mu_top_clean, mu_floor = clean_mu(mu_top, mu_nodata, land_top)

    Hu_int = np.flipud(Hu_top)
    mu_int = np.flipud(mu_top_clean)
    land_int = np.flipud(land_top)

    dx_m = abs(Hu_transform.a)
    dy_m = abs(Hu_transform.e)
    dx_km = dx_m / 1000.0
    dy_km = dy_m / 1000.0
    cell_area_km2 = (dx_m * dy_m) / 1e6

    Ny, Nx = Hu_top.shape
    mesh = Grid2D(nx=Nx, ny=Ny, dx=dx_km, dy=dy_km)

    Hu_f = Hu_int.ravel()
    mu_f = mu_int.ravel()
    land_f = land_int.ravel()
    non_land = np.count_nonzero(land_f < 0.5)   # or land_f == 0
    total = land_f.size
    print(f"Non-land cells: {non_land}/{total} ({100*non_land/total:.2f}%)")

    Hi = CellVariable(name="Hi", mesh=mesh, hasOld=True, value=0.0)
    Vu = CellVariable(name="Vu", mesh=mesh, hasOld=True, value=0.0)
    Vi = CellVariable(name="Vi", mesh=mesh, hasOld=True, value=0.0)

    meshBnd = mesh.facesLeft | mesh.facesRight | mesh.facesBottom | mesh.facesTop
    for var in (Hi, Vu, Vi):
        var.faceGrad.constrain(0.0, meshBnd)

    Hi0_int = build_initial_Hi_point_seed(
        Hu_transform=Hu_transform,
        Hu_crs=Hu_crs,
        Hu_shape=Hu_top.shape,
        landmask_topdown=land_top,
        landmask_int=land_int,
        cell_area_km2=cell_area_km2,
        mesh=mesh,
        p=p,
    )
    Hi.value = Hi0_int.ravel()
    total_people = np.sum((Hu_f * land_f) * cell_area_km2)
    total_area = np.sum(land_f) * cell_area_km2
    total_mosq = p.mosquitoes_per_person * total_people
    M0_uniform = total_mosq / max(total_area, 1e-9)
    M0 = M0_uniform * land_f
    Vi0 = np.minimum(p.Vi0_multiplier * Hi.value, 0.9 * M0)
    Vu0 = np.maximum(M0 - Vi0, 0.0)

    Vu.value = Vu0
    Vi.value = Vi0

    HuVar = CellVariable(name="Hu", mesh=mesh, value=Hu_f)
    muVar = CellVariable(name="mu", mesh=mesh, value=mu_f)
    landVar = CellVariable(name="landmask", mesh=mesh, value=land_f)
    oceanSinkCoeff = p.mask_sink * (1.0 - landVar)

    if 0 in p.export_weeks:
        Hi0_out_top = np.flipud(np.array(Hi.value).reshape((Ny, Nx)))
        Hi0_out_top = np.where(land_top > 0.5, Hi0_out_top, p.export_nodata)
        write_geotiff(out_dir / f"{p.export_prefix}{0:03d}.tif", Hi0_out_top, Hu_prof, p.export_nodata, p.export_dtype)

    t0 = dt.date(2015, 12, 20)
    n_steps = int(round(p.weeks_to_simulate / p.dt_week))
    steps_per_week = int(round(1.0 / p.dt_week))

    out_rows = [["week_index", "date", "t_week", "total_Hi", "weekly_incidence_proxy", "beta_t"]]
    inc_this_week = 0.0
    t_week_steps = []
    beta_steps = []
    solver = None
    for step in range(1, n_steps + 1):
        t_week = step * p.dt_week
        beta_t = emg_beta(t_week, p)
        t_week_steps.append(t_week)
        beta_steps.append(beta_t)

        Hi.updateOld()
        Vu.updateOld()
        Vi.updateOld()

        M_old = Vu.old + Vi.old

        eqHi = (TransientTerm(var=Hi)
                == DiffusionTerm(coeff=p.delta1, var=Hi)
                + ImplicitSourceTerm(coeff=-(p.lam + oceanSinkCoeff), var=Hi)
                + (p.sigma1 * HuVar * Vi.old))

        coeff_Vu = (beta_t - p.mu1
                    - p.sigma2 * Hi.old
                    - muVar * M_old
                    - oceanSinkCoeff)
        eqVu = (TransientTerm(var=Vu)
                == DiffusionTerm(coeff=p.delta2, var=Vu)
                + ImplicitSourceTerm(coeff=coeff_Vu, var=Vu)
                + (beta_t * Vi.old))

        coeff_Vi = -(p.mu1 + muVar * M_old + oceanSinkCoeff)
        eqVi = (TransientTerm(var=Vi)
                == DiffusionTerm(coeff=p.delta2, var=Vi)
                + ImplicitSourceTerm(coeff=coeff_Vi, var=Vi)
                + (p.sigma2 * Vu * Hi.old))
        eqHi.sweep(dt=p.dt_week, solver=solver)
        eqVu.sweep(dt=p.dt_week, solver=solver)
        eqVi.sweep(dt=p.dt_week, solver=solver)

        Hi.value = np.maximum(Hi.value, 0.0)
        Vu.value = np.maximum(Vu.value, 0.0)
        Vi.value = np.maximum(Vi.value, 0.0)

        if (not np.all(np.isfinite(Hi.value))) or (not np.all(np.isfinite(Vu.value))) or (not np.all(np.isfinite(Vi.value))):
            raise FloatingPointError("Non-finite values detected (NaN/inf). Check μ cleaning and timestep.")

        inc_dt = float(np.sum((p.sigma1 * Hu_f * np.array(Vi.value)) * land_f) * cell_area_km2 * p.dt_week)
        inc_this_week += inc_dt

        if step % steps_per_week == 0:
            week_index = step // steps_per_week
            date = t0 + dt.timedelta(days=7 * week_index)
            total_Hi = float(np.sum(np.array(Hi.value) * land_f) * cell_area_km2)

            out_rows.append([week_index, str(date), week_index, total_Hi, inc_this_week, beta_t])
            inc_this_week = 0.0

            print(f"week {week_index:02d}: total_Hi={total_Hi:.6f}, beta={beta_t:.2f}")

            if week_index in p.export_weeks:
                Hi_out_top = np.flipud(np.array(Hi.value).reshape((Ny, Nx)))
                Hi_out_top = np.where(land_top > 0.5, Hi_out_top, p.export_nodata)
                write_geotiff(out_dir / f"{p.export_prefix}{week_index:03d}.tif", Hi_out_top, Hu_prof, p.export_nodata, p.export_dtype)

    out_csv = out_dir / f"martinique_predicted_cases_{p.weeks_to_simulate}w.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
        print(f"\nWrote {out_csv}")


    plot_postrun_outputs(
        out_rows=out_rows,
        t0=t0,
        t_week_steps=np.asarray(t_week_steps, dtype=float),
        beta_steps=np.asarray(beta_steps, dtype=float),
        prefix=str(out_dir / "martinique"),
        dpi=200,
        show=False,
    )

if __name__ == "__main__":
    main()
