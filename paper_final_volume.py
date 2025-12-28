#### IMPORTING ####
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Callable, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

try:
    import polars as pl
    _HAS_POLARS = True
except Exception:
    _HAS_POLARS = False
###########################

###### GLOBAL VARIABLES ######
H_MIN, H_MAX, H_MESH = 0.01, 0.499, 0.001
WINDOWS = [12, 24, 36, 48, 60, 120, 180, 240, 360]
N_LAGS_QV = 3             
SUBSAMPLING_SECONDS = 5   

DATA_DIR = Path(__file__).resolve().parent / "clean2"
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
#############################

###### REMOVING DATES #########
FOMC = [
    "2012-01-25",
    "2012-03-13",
    "2012-04-25",
    "2012-06-20",
    "2012-08-01",
    "2012-09-13",
    "2012-10-24",
    "2012-12-12",

    "2013-01-30",
    "2013-03-20",
    "2013-05-01",
    "2013-06-19",
    "2013-07-31",
    "2013-09-18",
    "2013-10-30",
    "2013-12-18",

    "2014-01-29",
    "2014-03-19",
    "2014-04-30",
    "2014-06-18",
    "2014-07-30",
    "2014-09-17",
    "2014-10-29",
    "2014-12-17",

    "2015-01-28",
    "2015-03-18",
    "2015-04-29",
    "2015-06-17",
    "2015-07-29",
    "2015-09-17",
    "2015-10-28",
    "2015-12-16",

    "2016-01-27",
    "2016-03-16",
    "2016-04-27",
    "2016-06-15",
    "2016-07-27",
    "2016-09-21",
    "2016-11-02",
    "2016-12-14",

    "2017-02-01",
    "2017-03-15",
    "2017-05-03",
    "2017-06-14",
    "2017-07-26",
    "2017-09-20",
    "2017-11-01",
    "2017-12-13",

    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",

    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",

    "2020-01-29",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",

    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",

    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
]

TRADING_HALT = [
    '2013-07-03', 
    '2013-11-29', 
    '2013-12-24',

    '2014-07-03', 
    '2014-10-30', 
    '2014-11-28', 
    '2014-12-24', 
    
    '2015-11-27', 
    '2015-12-24', 
    
    '2016-11-25', 
    
    '2017-07-03', 
    '2017-11-24', 
    
    '2018-07-03', 
    '2018-11-23', 
    '2018-12-24',

    "2019-07-03",
    "2019-08-12",
    "2019-11-29",
    "2019-12-24",

    "2020-03-09",
    "2020-03-12",
    "2020-03-16",
    "2020-03-18",
    "2020-11-27",
    "2020-12-24",

    "2021-05-05",
    "2022-11-26",
    
    "2022-11-25"
]
###################################


#### WORKFLOW #######
def Phi_Hl(l: int, H: float) -> float:
    """
    params:
        l: integer lag index (0, 1, 2, ...)
        H: Hurst parameter (typically in (0, 0.5))
    Returns:
        Floating-point Phi_{H,l} coefficient.
    """
    num = (abs(l + 2) ** (2 * H + 2) - 4 * abs(l + 1) ** (2 * H + 2) +
           6 * abs(l) ** (2 * H + 2) - 4 * abs(l - 1) ** (2 * H + 2) +
           abs(l - 2) ** (2 * H + 2))
    den = 2 * (2 * H + 1) * (2 * H + 2)
    return num / den

def estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H_min=H_MIN, H_max=H_MAX, mesh=H_MESH):
    """
    Performing a grid search over H in [H_min, H_max) with step
    `mesh`. For each candidate H it evaluates the theoretical moment vector
    Psi(H) via `Psi_func`, forms the quadratic GMM objective using the
    weighting matrix W and empirical moments V, and selects the H that
    minimizes the objective F. After selecting H_hat it computes R_hat by
    projecting V onto Psi(H_hat) under W.
    params:
        W: weighting matrix (m×m numpy array)
        V: empirical moment column vector (m×1 numpy array)
        Psi_func: callable H -> Psi(H) returning (m×1) numpy array
        H_min, H_max, mesh: grid parameters controlling H search range and resolution
    Returns:
        Tuple (H_hat, R_hat) as floats
    """
    H_values = np.arange(H_min, H_max, mesh)
    best_H, best_F = None, np.inf
    for H in H_values:
        Psi = Psi_func(H)
        term0 = V.T @ W @ V
        term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
        term2 = Psi.T @ W @ Psi
        term0, term1, term2 = term0.item(), term1.item(), term2.item()
        if term2 == 0:
            continue
        R = term1 / term2 / 2
        F = term0 - R * term1 + term2 * R * R
        if np.isnan(F):
            continue
        if F < best_F:
            best_F = F
            best_H = H
    if best_H is None:
        return np.nan, np.nan
    Psi_star = Psi_func(best_H)
    term1 = (Psi_star.T @ W @ V).item()
    term2 = (Psi_star.T @ W @ Psi_star).item()
    R_hat = term1 / term2 / 2 if term2 != 0 else 0.0
    return float(best_H), float(R_hat)

def estimation_ratio(V: np.ndarray, Psi_func: Callable[[float], np.ndarray], H_min=H_MIN, H_max=H_MAX, mesh=H_MESH):
    """
    Finds H that minimizes |Psi[1]/Psi[0] - V[1]/V[0]| over a grid of H.
    Returns: (H_hat, R_hat)
    """
    if V.size < 2 or V[0] == 0:
        return np.nan, np.nan
    emp_ratio = V[1] / V[0]
    H_values = np.arange(H_min, H_max, mesh)
    best_H = None
    best_err = np.inf
    for H in H_values:
        Psi = Psi_func(H).flatten()
        if Psi[0] == 0:
            continue
        psi_ratio = Psi[1] / Psi[0]
        err = abs(psi_ratio - emp_ratio)
        if np.isnan(err):
            continue
        if err < best_err:
            best_err = err
            best_H = H
    if best_H is None:
        return np.nan, np.nan
    Psi_star = Psi_func(best_H)
    term1 = (Psi_star.T @ V.reshape(-1, 1)).item()
    term2 = (Psi_star.T @ Psi_star).item()
    R_hat = term1 / term2 / 2 if term2 != 0 else 0.0
    return float(best_H), float(R_hat)

def plot_psi_ratio(Psi_func: Callable[[float], np.ndarray], V: np.ndarray, K: int, out_dir: Optional[Path] = None):
    """
    Plot Psi[1]/Psi[0] as a function of H and mark the empirical V[1]/V[0].
    """
    if V.size < 2 or V[0] == 0:
        return None
    if out_dir is None:
        out_dir = PROJECT_ROOT / "debug_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    H_values = np.arange(H_MIN, H_MAX, H_MESH)
    psi_ratios = []
    for H in H_values:
        Psi = Psi_func(H).flatten()
        if Psi[0] == 0:
            psi_ratios.append(np.nan)
        else:
            psi_ratios.append(Psi[1] / Psi[0])
    psi_ratios = np.array(psi_ratios)
    emp_ratio = V[1] / V[0]
    plt.figure(figsize=(6, 3))
    plt.plot(H_values, psi_ratios, label="Psi[1]/Psi[0]")
    plt.axhline(emp_ratio, color="red", linestyle="--", label=f"emp ratio={emp_ratio:.4f}")
    plt.xlabel("H")
    plt.ylabel("Psi[1]/Psi[0]")
    plt.title(f"Psi ratio vs H (K={K})")
    plt.legend()
    fname = out_dir / f"psi_ratio_K{K}.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def build_Psi_function(lags: np.ndarray, window_steps: int):
    """
    The returned callable Psi(H) produces the theoretical moment vector
    for the requested lags and window size. The vector has shape (m, 1),
    where m == len(lags). The factor window_steps ** (2*H) captures the
    scaling with the window length
    params:
        lags: Array-like of integer lag indices to include (e.g., [1, 2]).
        window_steps: Integer number of subsampled steps in the window (K).
    Returns:
        Callable that maps H (float) to an (m, 1) numpy array of moments
    """

    def Psi(H):
        factor = (window_steps ** (2 * H))
        out = []
        for lag in lags:
            if lag == 1:
                out.append(factor * (Phi_Hl(0, H) + 2 * Phi_Hl(1, H)))
            else:
                out.append(factor * Phi_Hl(int(lag), H))
        return np.array(out).reshape(-1, 1)

    return Psi

def std_truncation(arr: np.ndarray, mult: float) -> np.ndarray:
    """
    params:
        arr: 1D numpy array of increments or residuals.
        mult: multiplier for the standard deviation used as threshold.
    Returns:
        A copy of `arr` where values with |value| > mult * std(arr) are set to 0.
        If `arr` is empty or has zero std, it is returned unchanged.
    """
    if arr.size == 0:
        return arr
    sd = np.std(arr)
    if sd == 0:
        return arr
    thr = mult * sd
    arr = arr.copy()
    arr[np.abs(arr) > thr] = 0
    return arr

def realized_variance(series: np.ndarray, window_steps: int, delta: float) -> np.ndarray:
    """
    params:
        series: 1D numpy array of increments (e.g. price or count differences).
        window_steps: number of subsampled steps (K) used in the rolling window.
        delta: time step length (seconds) corresponding to one subsampled step.
    Returns:
        1D numpy array of realized-variance values; empty array if input too short.
    """
    if series.size < window_steps + 1:
        return np.array([])
    rv = np.concatenate([[0.0], np.cumsum(series ** 2)])
    vol = (rv[window_steps:] - rv[:-window_steps]) / (delta * window_steps)
    return vol

def build_pattern(vols: List[np.ndarray]) -> np.ndarray:
    """
    params:
        vols: list of 1D numpy arrays (realized-variance series for each day)
    Returns:
        1D numpy array representing the averaged normalized pattern; returns
        an empty array when input is empty or normalization is impossible.
    """
    vols = [v for v in vols if len(v) > 0]
    if not vols:
        return np.array([])
    min_len = min(len(v) for v in vols)
    if min_len == 0:
        return np.array([])
    acc = np.zeros(min_len)
    for v in vols:
        sub = v[:min_len]
        m = sub.mean() if sub.size else 1.0
        if m == 0:
            return np.array([])
        acc += sub / m
    return acc / len(vols)

def quadratic_covariations(
    vol: np.ndarray,
    pattern: np.ndarray,
    window_steps: int,
    n_lags: int,
    trunc_mult: float = 3.0,
    adjust_lag1: bool = True,
) -> np.ndarray:
    """
    1) Normalize the day's realized-variance by the intraday pattern and mean.
    2) Form increments at horizon window_steps
    3) Apply std-based truncation to suppress outliers
    4) Compute empirical covariances for lags 0..(n_lags-1).
    5) Apply the lag-1 adjustment (if enabled) and return lags 1..(n_lags-1).
    params:
        vol: Realized-variance series for the day (1D numpy array).
        pattern: Averaged normalized pattern (1D numpy array).
        window_steps: Integer K (number of subsampled steps).
        n_lags: Number of lag entries to compute (including lag 0).
        trunc_mult: Multiplier for std truncation on increments.
        adjust_lag1: Whether to apply the lag-1 adjustment used by the protocol.
    Returns:
        1D numpy array of length n_lags-1 with covariations for lags 1..(n_lags-1),
        or an empty array when computation is not possible.
    """
    max_len = min(len(vol), len(pattern))
    if max_len <= window_steps:
        return np.array([])
    vol = vol[:max_len]
    pattern = pattern[:max_len]
    mean_vol = vol.mean() if vol.size else 1.0
    norm = vol / pattern / mean_vol
    norm = vol / pattern / mean_vol
    inc = norm[window_steps:] - norm[:-window_steps]
    inc = std_truncation(inc, 3.0)
    if inc.size == 0:
        return np.array([])
    cov = []
    for lag in range(n_lags):
        if lag == 0:
            cov.append(np.mean(inc ** 2))
        else:
            m = len(inc) - lag * window_steps
            if m <= 0:
                cov.append(np.nan)
                continue
            cov.append(np.mean(inc[:m] * inc[lag * window_steps:]))
    if n_lags > 1 and adjust_lag1:
        cov[1] = cov[0] + 2 * cov[1]
    return np.array(cov[1:])  # drop lag0

#Tryied Laplacian smoothing without improvement
def laplacian_smoothing(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply Laplacian smoothing to the series.
    x[t] = (1-alpha)*x[t] + alpha*(x[t-1] + x[t+1])/2
    """
    if len(x) < 3:
        return x
    s = x.copy()
    # Vectorized update for internal points
    s[1:-1] = (1 - alpha) * x[1:-1] + (alpha / 2) * (x[:-2] + x[2:])
    return s

def load_clean2(metric: str, progress_cb: Optional[Callable[[int, int], None]] = None) -> List[pd.Series]:
    """
    - Read each CSV (Polars or pandas, Polars is useful for fast reading)
    - Remove FOMC and TRADING_HALTS dates.
    - Parse timestamps and resample to SUBSAMPLING_SECONDS.
    - For orders: count entries per bin; for size: sum SIZE per bin.
    - Normalize each day by its L2 norm and cumulatively sum.
    params:
        metric: Either "orders" or "size".
        progress_cb: Optional callback invoked as progress_cb(i, total).
    Returns:
        List of pandas Series (one per day), indexed by datetime.
    """
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")
    exclude_dates = set(FOMC) | set(TRADING_HALT)
    series_list = []
    total = len(files)
    for idx, f in enumerate(tqdm(files, desc="Loading clean2")):
        if _HAS_POLARS:
            df_pl = pl.read_csv(f, columns=["DATE", "TIME_M", "SIZE"])
            df_pl = df_pl.filter(~pl.col("DATE").is_in(exclude_dates))
            if df_pl.height == 0:
                if progress_cb:
                    progress_cb(idx + 1, total)
                continue
            df = df_pl.to_pandas()
        else:
            df = pd.read_csv(f, usecols=["DATE", "TIME_M", "SIZE"])
            df = df[~df["DATE"].isin(exclude_dates)]
            if df.empty:
                if progress_cb:
                    progress_cb(idx + 1, total)
                continue
        df["DT"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"], errors="coerce")
        df = df.dropna(subset=["DT"])
        df = df.sort_values("DT")
        if metric == "orders":
            df = df.drop_duplicates(subset=["TIME_M"])
            ser = df.set_index("DT")["TIME_M"]
            ser = ser.resample(f"{SUBSAMPLING_SECONDS}s").count().astype(float)
            if len(ser) > 0:
                l2 = np.linalg.norm(ser.values.astype(float))
                if l2 != 0:
                    ser = ser / l2
        else:
            ser = df.set_index("DT")["SIZE"]
            ser = pd.to_numeric(ser, errors="coerce")
            ser = ser.resample(f"{SUBSAMPLING_SECONDS}s").sum()
            if len(ser) > 0:
                l2 = np.linalg.norm(ser.values.astype(float))
                if l2 != 0:
                    ser = ser / l2
        
        ser = ser.fillna(0)
        ser = ser.cumsum()
            
        if not ser.empty:
            series_list.append(ser)
        if progress_cb:
            progress_cb(idx + 1, total)
    if not series_list:
        raise RuntimeError("No positive values after loading.")
    return series_list

def plot_size_column(sample_path: Path) -> None:
    """
    Plot the raw SIZE column for a single clean2 CSV file.
    """
    if _HAS_POLARS:
        df_pl = pl.read_csv(sample_path, columns=["DATE", "TIME_M", "SIZE"])
        df = df_pl.to_pandas()
    else:
        df = pd.read_csv(sample_path, usecols=["DATE", "TIME_M", "SIZE"])
    df["DT"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"], errors="coerce")
    df = df.dropna(subset=["DT"])
    df = df.sort_values("DT")
    df["SIZE"] = pd.to_numeric(df["SIZE"], errors="coerce")
    df = df.dropna(subset=["SIZE"])
    if df.empty:
        print(f"No SIZE data to plot for {sample_path.name}")
        return
    plt.figure(figsize=(8, 3))
    plt.plot(df["DT"], df["SIZE"], linewidth=0.7)
    plt.title(f"Raw SIZE series: {sample_path.name}")
    plt.xlabel("Time")
    plt.ylabel("SIZE")
    plt.tight_layout()
    out_dir = PROJECT_ROOT / "debug_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"size_series_{sample_path.stem}.png"
    plt.savefig(out_path)
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    else:
        print(f"Saved SIZE plot to {out_path}")

    plt.figure(figsize=(6, 3))
    plt.hist(df["SIZE"].to_numpy(), bins=50, edgecolor="black", linewidth=0.3)
    plt.title(f"SIZE histogram: {sample_path.name}")
    plt.xlabel("SIZE")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    hist_path = out_dir / f"size_hist_{sample_path.stem}.png"
    plt.savefig(hist_path)
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    else:
        print(f"Saved SIZE histogram to {hist_path}")

def plot_size_histogram_all(bins: int = 50) -> None:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")
    exclude_dates = set(FOMC) | set(TRADING_HALT)
    all_sizes = []
    for f in files:
        if _HAS_POLARS:
            df_pl = pl.read_csv(f, columns=["DATE", "SIZE"])
            df_pl = df_pl.filter(~pl.col("DATE").is_in(exclude_dates))
            if df_pl.height == 0:
                continue
            sizes = df_pl["SIZE"].to_pandas()
        else:
            df = pd.read_csv(f, usecols=["DATE", "SIZE"])
            df = df[~df["DATE"].isin(exclude_dates)]
            if df.empty:
                continue
            sizes = df["SIZE"]
        sizes = pd.to_numeric(sizes, errors="coerce").dropna()
        if not sizes.empty:
            all_sizes.append(sizes.to_numpy())
    if not all_sizes:
        raise RuntimeError("No SIZE data available after filtering.")
    all_sizes = np.concatenate(all_sizes)
    plt.figure(figsize=(6, 3))
    plt.hist(all_sizes, bins=bins, edgecolor="black", linewidth=0.3)
    plt.title("SIZE histogram: all data")
    plt.xlabel("SIZE")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    out_dir = PROJECT_ROOT / "debug_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "size_hist_all.png"
    plt.savefig(hist_path)
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    else:
        print(f"Saved SIZE histogram to {hist_path}")

def compute_H_series(
    series_list: List[pd.Series],
    tail_mode: str = "lag1-off",
    tail_trunc_mult: float = 2.0,
    ) -> Dict[int, Dict]:
    """
    For each K:
    - Computes increments for each day's subsampled series.
    - Applies std-based truncation to increments.
    - Computes realized-variance series (windowed).
    - Builds the normalized intraday pattern across days.
    - Computes quadratic covariations per day relative to the pattern.
    - Averages empirical covariations and runs the GMM grid search.
    params:
        series_list: List of pandas Series indexed by datetime and resampled to
            SUBSAMPLING_SECONDS (values are counts or sizes).
        tail_mode: Rule for handling tail behavior in covariations.
        tail_trunc_mult: Truncation multiplier used in tail handling.
    Returns:
        Dictionary keyed by K with entries containing:
        - H_hat, R_hat
        - V_avg (averaged empirical moments)
        - lags used for estimation
    """
    results: Dict[int, Dict] = {}
    # delta = SUBSAMPLING_SECONDS
    for K in WINDOWS:
        vols = []
        inc_lens = []
        vol_lens = []
        for ser in series_list:
            vals = ser.values.astype(float)
            inc = np.diff(vals)
            inc = inc[~np.isnan(inc) & ~np.isinf(inc)]
            inc_lens.append(len(inc))
            
            if inc.size < K:
                continue
            csum = np.concatenate([[0.0], np.cumsum(inc)])
            vol_series = (csum[K:] - csum[:-K])
            
            if vol_series.size == 0:
                continue
            vol_lens.append(len(vol_series))
            vols.append(vol_series)
        if vols:
            print(
                f"K={K}: total_days={len(series_list)} usable_days={len(vols)} "
                f"mean_inc_len={np.mean(inc_lens):.1f} mean_vol_len={np.mean(vol_lens):.1f}"
            )
        else:
            print(f"K={K}: no usable days (total_days={len(series_list)})")
        pattern = build_pattern(vols)
        if pattern.size == 0:
            results[K] = {"H": np.nan, "R": np.nan, "V_avg": []}
            continue
        #before the V_days 
        trunc_mult = 3.0
        if tail_mode == "trunc" and K >= 180:
            trunc_mult = tail_trunc_mult
        V_days = []
        for v in vols:
            V_day = quadratic_covariations(v, pattern, K, N_LAGS_QV, trunc_mult=trunc_mult)
            if V_day.size:
                V_days.append(V_day)
        if not V_days:
            results[K] = {"H": np.nan, "R": np.nan, "V_avg": []}
            continue
        V_matrix = np.vstack(V_days)
        V_avg = np.nanmean(V_matrix, axis=0)
        lags = np.arange(1, N_LAGS_QV)
        Psi = build_Psi_function(lags, K)
        W = np.eye(len(lags))
        H_hat, R_hat = np.nan, np.nan
        use_ratio = len(lags) == 2 and K >= 180 and tail_mode != "gmm-only"
        if use_ratio:
            #tryed ratio estimation
            H_ratio, R_ratio = estimation_ratio(V_avg, Psi)
            try:
                saved = plot_psi_ratio(Psi, V_avg, K)
                if saved is not None:
                    print(f"Saved psi-ratio plot: {saved}")
            except Exception:
                pass
            if not np.isnan(H_ratio) and H_ratio > H_MIN + 1e-12:
                H_hat, R_hat = H_ratio, R_ratio
            else:
                H_hat, R_hat = estimation_GMM(W, V_avg.reshape(-1, 1), Psi)
        else:
            H_hat, R_hat = estimation_GMM(W, V_avg.reshape(-1, 1), Psi)
        if use_ratio and V_avg.size == 2 and V_avg[0] != 0:
            emp_ratio = V_avg[1] / V_avg[0]
            H_values = np.arange(H_MIN, H_MAX, H_MESH)
            psi_ratios = []
            for H in H_values:
                Psi_vals = Psi(H).flatten()
                if Psi_vals[0] == 0:
                    continue
                psi_ratios.append(Psi_vals[1] / Psi_vals[0])
            if psi_ratios:
                psi_min, psi_max = float(np.min(psi_ratios)), float(np.max(psi_ratios))
                out_of_bounds = emp_ratio < psi_min or emp_ratio > psi_max
                results.setdefault(K, {})
                results[K]["ratio_bounds"] = (emp_ratio, psi_min, psi_max)
                if out_of_bounds and tail_mode == "lag1-off":
                    V_days_alt = []
                    for v in vols:
                        V_day = quadratic_covariations(
                            v, pattern, K, N_LAGS_QV, trunc_mult=trunc_mult, adjust_lag1=False
                        )
                        if V_day.size:
                            V_days_alt.append(V_day)
                    if V_days_alt:
                        V_matrix_alt = np.vstack(V_days_alt)
                        V_avg_alt = np.nanmean(V_matrix_alt, axis=0)
                        if V_avg_alt.size == V_avg.size and not np.isnan(V_avg_alt).any():
                            V_avg = V_avg_alt
                            Psi = build_Psi_function(lags, K)
                            H_hat, R_hat = estimation_ratio(V_avg, Psi)
                            if np.isnan(H_hat) or H_hat <= H_MIN + 1e-12:
                                H_hat, R_hat = estimation_GMM(W, V_avg.reshape(-1, 1), Psi)
                            print(f"K={K}: empirical ratio out of bounds; recomputed without lag1 adjustment.")
                elif out_of_bounds and tail_mode == "clip":
                    clip_ratio = min(max(emp_ratio, psi_min), psi_max)
                    best_idx = None
                    best_err = np.inf
                    for idx, H in enumerate(H_values):
                        Psi_vals = Psi(H).flatten()
                        if Psi_vals[0] == 0:
                            continue
                        psi_ratio = Psi_vals[1] / Psi_vals[0]
                        err = abs(psi_ratio - clip_ratio)
                        if err < best_err:
                            best_err = err
                            best_idx = idx
                    if best_idx is not None:
                        H_hat = float(H_values[best_idx])
                        Psi_star = Psi(H_hat)
                        term1 = (Psi_star.T @ V_avg.reshape(-1, 1)).item()
                        term2 = (Psi_star.T @ Psi_star).item()
                        R_hat = term1 / term2 / 2 if term2 != 0 else 0.0
                        print(f"K={K}: empirical ratio out of bounds; clipped into bounds.")
        results[K] = {"H": H_hat, "R": R_hat, "V_avg": V_avg, "lags": lags}
    return results

def plot_results(results: Dict[int, Dict]):
    pattern_K = 120 if 120 in results else WINDOWS[0]
    res0 = results.get(pattern_K, {})
    if res0 and "V_avg" in res0 and len(res0.get("V_avg", [])) > 0:
        plt.figure(figsize=(6, 3))
        lags = res0.get("lags", np.arange(1, len(res0["V_avg"]) + 1))
        V_avg = np.array(res0["V_avg"])
        plt.plot(lags, V_avg, "o-", label=f"V_avg (K={pattern_K})")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.title("Volatility correction (V_avg)")
        plt.xlabel("lag")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    for K, res in results.items():
        V_avg = np.array(res.get("V_avg", []))
        lags = res.get("lags", np.arange(1, len(V_avg) + 1))
        H_hat = res.get("H")
        if V_avg.size == 0 or np.isnan(V_avg).any() or H_hat is None or np.isnan(H_hat):
            continue
        Psi_func = build_Psi_function(lags, K)
        Psi_vals = Psi_func(H_hat).flatten()
        plt.figure(figsize=(6, 3))
        plt.plot(lags, V_avg, "o-", label="V_avg")
        plt.plot(lags, Psi_vals, "s--", label=f"Psi(H={H_hat:.3f})")
        plt.title(f"Diagnostic: V vs Psi (K={K})")
        plt.xlabel("lag")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Hurst estimation on clean2 data (orders or volume).")
    parser.add_argument("--plot-size-only", action="store_true", help="plot raw SIZE for the first CSV and exit (volume only)")
    parser.add_argument(
        "--tail-mode",
        choices=["lag1-off", "clip", "trunc", "gmm-only"],
        default="lag1-off",
        help="handling for tail windows (K>=180): lag1-off, clip, trunc, gmm-only",
    )
    parser.add_argument(
        "--tail-trunc-mult",
        type=float,
        default=2.0,
        help="std truncation multiplier for tail windows when --tail-mode=trunc",
    )
    parser.add_argument(
        "--print-ratio-bounds",
        action="store_true",
        help="print empirical ratio and theoretical bounds for K>=180",
    )
    args = parser.parse_args()

    metric = "volume"
    files = sorted(DATA_DIR.glob("*.csv"))
    if files:
        plot_size_column(files[0])
        plot_size_histogram_all()
        if args.plot_size_only:
            return
    else:
        print(f"No CSV files found in {DATA_DIR} to plot SIZE.")

    series_list = load_clean2(metric=metric)
    print(f"Loaded {len(series_list)} days from clean2 using metric {metric}")
    print(f"Tail mode: {args.tail_mode}")
    results = compute_H_series(
        series_list,
        tail_mode=args.tail_mode,
        tail_trunc_mult=args.tail_trunc_mult,
    )
    if args.print_ratio_bounds:
        for K in WINDOWS:
            if K < 180:
                continue
            ratio_info = results.get(K, {}).get("ratio_bounds")
            if ratio_info:
                emp_ratio, psi_min, psi_max = ratio_info
                print(
                    f"K={K} r_emp={emp_ratio:.6f} r_theo_min={psi_min:.6f} r_theo_max={psi_max:.6f}"
                )
    for K, res in results.items():
        print(f"K={K} H={res.get('H')} R={res.get('R')} V_avg={res.get('V_avg')}")
    plot_results(results)
    labels = []
    H_vals = []
    for K, res in results.items():
        labels.append(f"{int(K/12)}T" if K % 12 == 0 else f"{K} bins")
        H_vals.append(res.get("H"))
    plt.figure(figsize=(6, 3))
    plt.bar(labels, H_vals)
    plt.xlabel("Frequency (T)")
    plt.ylabel("Hurst exponent")
    plt.title("H vs frequency (5s bins)")
    plt.tight_layout()
    plt.show()
    print(f"Total runtime: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
