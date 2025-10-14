from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

FOMC_announcement = [
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
]
trading_halt = [
    "2013-07-03","2013-11-29","2013-12-24",
    "2014-07-03","2014-10-30","2014-11-28","2014-12-24",
    "2015-11-27","2015-12-24",
    "2016-11-25",
    "2017-07-03","2017-11-24",
    "2018-07-03","2018-11-23","2018-12-24",
    "2019-07-03","2019-08-12","2019-11-29","2019-12-24",
    "2020-03-09","2020-03-12","2020-03-16","2020-03-18","2020-11-27","2020-12-24",
    "2021-05-05","2022-11-26","2022-11-25",
]
EXCLUDE_DATES_DEFAULT: Set[pd.Timestamp] = set(
    pd.to_datetime(FOMC_announcement + trading_halt, errors="coerce").dropna().normalize()
)

RTH_OPEN = (9, 30)   # 09:30
RTH_CLOSE = (16, 0)  # 16:00
RTH_SECONDS = int(6.5 * 3600)  # 23400


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["empirical_plus_mc_from_data", "paper_mc"], default="empirical_plus_mc_from_data")
    ap.add_argument("--csv", type=str, required=False, help="CSV with columns DT,Price (timestamps parseable by pandas).")
    ap.add_argument("--dt", type=int, default=5, help="Grid step in seconds (default 5). Use 1 to compare 1s.")
    ap.add_argument("--n-mc", type=int, default=200, help="Monte Carlo runs.")
    ap.add_argument("--days", type=int, default=252, help="[paper_mc] trading days per MC run.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument("--exclude-dates-csv", type=str, default=None, help="Optional CSV of dates to exclude (col name irrelevant).")
    ap.add_argument("--jump-sigma", type=float, default=6.0, help="Robust sigma multiple to clip returns (price jumps removal).")
    ap.add_argument("--verbose", action="store_true")
    # paper_mc params
    ap.add_argument("--H", type=float, default=0.2, help="[paper_mc] Hurst for rough/Heston; here we just do Heston QE baseline.")
    return ap.parse_args()


def load_excludes(path: Optional[str]) -> Set[pd.Timestamp]:
    if path is None:
        return set(EXCLUDE_DATES_DEFAULT)
    p = Path(path)
    if not p.exists():
        return set(EXCLUDE_DATES_DEFAULT)
    try:
        df = pd.read_csv(p, nrows=10_000)
        col = df.columns[0]
        ts = pd.to_datetime(df[col], errors="coerce").dropna().normalize()
        out = set(ts)
        return set(out) | set(EXCLUDE_DATES_DEFAULT)
    except Exception:
        return set(EXCLUDE_DATES_DEFAULT)


def _align_one_df(df_in: pd.DataFrame, dt_sec: int, exclude_days: Set[pd.Timestamp], verbose: bool) -> Optional[pd.DataFrame]:
    df = df_in.dropna(subset=["DT", "Price"]).copy()
    df = df.sort_values("DT")
    df["date"] = df["DT"].dt.normalize()

    n_kept, n_skipped = 0, 0
    out = []
    for d, g in df.groupby("date", sort=True):
        if d in exclude_days:
            n_skipped += 1
            continue
        start = d + pd.Timedelta(hours=RTH_OPEN[0], minutes=RTH_OPEN[1])
        end   = d + pd.Timedelta(hours=RTH_CLOSE[0], minutes=RTH_CLOSE[1])

        gi = g.set_index("DT").sort_index()
        gi_buf = gi.loc[(gi.index >= start - pd.Timedelta(minutes=2)) & (gi.index <= end)]
        if gi_buf.empty:
            n_skipped += 1
            continue

        grid = pd.date_range(start, end, freq=f"{dt_sec}s")
        gi2 = gi_buf.reindex(grid)
        gi2["Price"] = gi2["Price"].ffill().bfill()
        if gi.loc[(gi.index >= start) & (gi.index <= end)].empty:
            n_skipped += 1
            continue

        gi2 = gi2.rename_axis("DT").reset_index()
        gi2["date"] = d
        out.append(gi2[["DT", "date", "Price"]])
        n_kept += 1

    if verbose:
        print(f"[align] dt={dt_sec}s → kept days: {n_kept} | skipped: {n_skipped}")
    if not out:
        return None
    return pd.concat(out, ignore_index=True)

def load_and_align(path_csv: Path, dt_sec: int, exclude_days: Set[pd.Timestamp], verbose: bool=True) -> Optional[pd.DataFrame]:
    raw = pd.read_csv(path_csv, parse_dates=["DT"]).dropna(subset=["DT","Price"])

    aligned = _align_one_df(raw, dt_sec, exclude_days, verbose)
    if aligned is not None and not aligned.empty:
        return aligned

    raw2 = raw.copy()
    dt_parsed = pd.to_datetime(raw2["DT"], utc=True, errors="coerce")
    raw2["DT"] = dt_parsed.dt.tz_convert("America/New_York").dt.tz_localize(None)
    aligned2 = _align_one_df(raw2, dt_sec, exclude_days, verbose)
    if aligned2 is not None and not aligned2.empty:
        if verbose:
            print("[align] Interpreted timestamps as UTC and converted to New York.")
        return aligned2

    return None


def clip_price_jumps_inplace(rets: np.ndarray, sigma_mult: float) -> None:
    """Robustly clip extreme intraday returns (remove price jumps).
    MAD-based scale; clip to +- sigma_mult * 1.4826 * MAD."""
    if rets.size == 0:
        return
    med = np.median(rets)
    mad = np.median(np.abs(rets - med))
    scale = 1.4826 * max(mad, 1e-12)
    thr = sigma_mult * scale
    np.clip(rets, -thr, thr, out=rets)

# ============================== Empirical H from RV-ACF =========================

def realized_variance_from_returns(r: np.ndarray, k_steps: int) -> np.ndarray:
    if k_steps <= 0 or r.size < k_steps:
        return np.empty(0, dtype=float)
    sq = r * r
    # rolling sum via convolution; length = n - k + 1
    return np.convolve(sq, np.ones(k_steps, float), mode="valid")

def autocovariances(x: np.ndarray, lags: List[int]) -> Dict[int, float]:
    if x.size == 0:
        return {ell: np.nan for ell in lags}
    xc = x - x.mean()
    n = xc.size
    out: Dict[int, float] = {}
    for ell in lags:
        if ell >= n:
            out[ell] = np.nan
            continue
        out[ell] = float(np.dot(xc[:n-ell], xc[ell:]) / n)
    return out

def estimate_H_from_autocov(gammas: Dict[int, float]) -> Tuple[float, float]:
    # Power law: gamma(ell) ≈ C * ell^alpha → H = 1 + alpha/2
    items = [(ell, g) for ell, g in gammas.items() if (g is not None) and np.isfinite(g) and g > 0]
    if len(items) < 2:
        return float("nan"), float("nan")
    l = np.array([i[0] for i in items], float)
    g = np.array([i[1] for i in items], float)
    X = np.vstack([np.ones_like(l), np.log(l)]).T
    beta, *_ = np.linalg.lstsq(X, np.log(g), rcond=None)
    alpha = float(beta[1])
    pred = X @ beta
    resid = np.log(g) - pred
    dof = max(len(g) - 2, 1)
    s2 = float((resid @ resid) / dof)
    cov_beta = s2 * np.linalg.inv(X.T @ X)
    se_H = 0.5 * float(np.sqrt(max(cov_beta[1, 1], 0.0)))
    return 1.0 + 0.5 * alpha, se_H

def to_daily_returns(aligned: pd.DataFrame, dt_sec: int, jump_sigma: float) -> List[np.ndarray]:
    """Extract daily log-return arrays; keep modal-length days; clip jumps per day."""
    rets: List[np.ndarray] = []
    lengths = []
    for d, g in aligned.groupby("date", sort=True):
        p = g["Price"].to_numpy(dtype=float)
        if p.size < 2:
            continue
        r = np.diff(np.log(p))
        clip_price_jumps_inplace(r, jump_sigma)
        rets.append(r)
        lengths.append(r.size)
    if not rets:
        raise RuntimeError("No usable days after alignment.")
    mode_len = int(pd.Series(lengths).mode().iloc[0])
    rets = [r for r in rets if r.size == mode_len]
    if not rets:
        raise RuntimeError("All days were filtered out by length consistency.")
    exp_steps = RTH_SECONDS // dt_sec
    if mode_len != exp_steps:
        if abs(mode_len - exp_steps) > 1:
            print(f"[warn] modal steps/day={mode_len}, expected ~{exp_steps} for dt={dt_sec}s")
    return rets

def estimate_H_on_returns(day_rets: List[np.ndarray], dt_sec: int) -> Dict[str, object]:
    # Choose k so that the *time* window matches 10m and 15m:
    k10 = int(round(600 / dt_sec))
    k15 = int(round(900 / dt_sec))
    l10 = [1,2,3,4,5,6]
    l15 = [1,2,3,4]
    ac10 = {ell: 0.0 for ell in l10}
    ac15 = {ell: 0.0 for ell in l15}
    n_days = 0
    for r in day_rets:
        rv10 = realized_variance_from_returns(r, k10)
        rv15 = realized_variance_from_returns(r, k15)
        if rv10.size == 0 or rv15.size == 0:
            continue
        a10 = autocovariances(rv10, l10)
        a15 = autocovariances(rv15, l15)
        for ell in l10:
            ac10[ell] += (0.0 if not np.isfinite(a10[ell]) else a10[ell])
        for ell in l15:
            ac15[ell] += (0.0 if not np.isfinite(a15[ell]) else a15[ell])
        n_days += 1
    if n_days == 0:
        return {"ac10": ac10, "ac15": ac15, "H": float("nan"), "H_se": float("nan"), "days": 0}
    ac10 = {k: v / n_days for k, v in ac10.items()}
    ac15 = {k: v / n_days for k, v in ac15.items()}
    pooled: Dict[int, float] = {}
    pooled.update(ac10)
    for ell, g in ac15.items():
        pooled[ell] = pooled.get(ell, 0.0) + g
    H, seH = estimate_H_from_autocov(pooled)
    return {"ac10": ac10, "ac15": ac15, "H": H, "H_se": seH, "days": n_days}

# ============================== Robust Heston calibration ========================

@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    s0: float

def calibrate_heston_from_returns_robust(all_rets: List[np.ndarray], dt_years: float, s0: float,
                                         xi_cap: float = 5.0, rho_bounds: Tuple[float,float]=(-0.2,0.0)
                                         ) -> HestonParams:
    r = np.concatenate(all_rets)
    v = (r**2) / dt_years
    v_t, v_tp = v[:-1], v[1:]
    X = np.vstack([np.ones_like(v_t), v_t]).T
    beta, *_ = np.linalg.lstsq(X, v_tp, rcond=None)
    c, phi = float(beta[0]), float(beta[1])
    phi = float(np.clip(phi, -0.99, 0.999999))
    kappa = max((1.0 - phi) / dt_years, 1e-8)
    theta = max(c / (1.0 - phi), 1e-12)

    e = v_tp - (c + phi * v_t)
    denom = float(np.sum(v_t * dt_years))
    xi2 = float(np.sum(e**2) / max(denom, 1e-16))
    xi_raw = float(np.sqrt(max(xi2, 1e-16)))
    xi = float(min(xi_raw, xi_cap))  # cap vol-of-vol to avoid pathological fits

    vt_dt = np.maximum(v_t * dt_years, 1e-16)
    eta = (r[1:] + 0.5 * v_t * dt_years) / np.sqrt(vt_dt)
    zeta = e / (xi * np.sqrt(vt_dt))
    rho_est = float(np.corrcoef(eta, zeta)[0, 1])
    rho = float(np.clip(rho_est, rho_bounds[0], rho_bounds[1]))
    v0 = float(theta)

    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, s0=float(s0))

# ============================== Heston QE simulator =============================

def heston_qe_step(s, v, dt, p: HestonParams, z1, z2):
    z_v = z1
    z_s = p.rho * z1 + np.sqrt(max(1.0 - p.rho**2, 0.0)) * z2
    # Andersen QE moments
    m = p.theta + (v - p.theta) * math.exp(-p.kappa * dt)
    s2 = (v * p.xi**2 * math.exp(-p.kappa * dt) / p.kappa) * (1.0 - math.exp(-p.kappa * dt)) \
         + (p.theta * p.xi**2 / (2.0 * p.kappa)) * (1.0 - math.exp(-p.kappa * dt))**2
    psi = s2 / (m**2 + 1e-16)
    if psi <= 1.5:
        b2 = 2.0 / psi - 1.0 + math.sqrt(max(2.0/psi, 0.0)) * math.sqrt(max(2.0/psi - 1.0, 0.0))
        a = m / (1.0 + b2)
        v_next = a * (math.sqrt(max(b2, 0.0)) * z_v + 1.0)**2
    else:
        p_qe = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p_qe) / (m + 1e-16)
        # convert z_v ~ N to U ~ (0,1) using Φ
        U = 0.5 * (1.0 + math.erf(z_v / math.sqrt(2.0)))
        v_next = 0.0 if U <= p_qe else -math.log((1.0 - p_qe) / (1.0 - U + 1e-16)) / (beta + 1e-16)
    v_bar = max(0.5 * (v + v_next), 0.0)
    s_next = s * math.exp(-0.5 * v_bar * dt + math.sqrt(max(v_bar, 0.0) * dt) * z_s)
    return s_next, max(v_next, 0.0)

def simulate_heston_day(p: HestonParams, dt_years: float, steps: int, rng: np.random.Generator):
    s = np.empty(steps + 1); v = np.empty(steps + 1)
    s[0], v[0] = p.s0, p.v0
    z = rng.standard_normal(size=(2, steps))
    for i in range(steps):
        s[i+1], v[i+1] = heston_qe_step(s[i], v[i], dt_years, p, z[0, i], z[1, i])
    return s, v

# ============================== MC driver =======================================

def run_mc_from_data(aligned_df: pd.DataFrame, dt_sec: int, n_mc: int, seed: int,
                     jump_sigma: float, verbose: bool=False):
    steps_per_day = int(RTH_SECONDS // dt_sec)
    dt_years = dt_sec / (252.0 * RTH_SECONDS)

    day_rets = to_daily_returns(aligned_df, dt_sec, jump_sigma)
    real_est = estimate_H_on_returns(day_rets, dt_sec)

    first_day = min(aligned_df["date"])
    s0 = float(aligned_df.loc[aligned_df["date"] == first_day, "Price"].iloc[0])
    params = calibrate_heston_from_returns_robust(day_rets, dt_years, s0, xi_cap=5.0, rho_bounds=(-0.2, 0.0))

    rng = np.random.default_rng(seed)
    H_list: List[float] = []
    valid = 0
    for _ in range(n_mc):
        mc_rets: List[np.ndarray] = []
        p_today = params
        for _day in range(len(day_rets)):
            s_path, v_path = simulate_heston_day(p_today, dt_years, steps_per_day, rng)
            r = np.diff(np.log(s_path))
            clip_price_jumps_inplace(r, jump_sigma)  # mirror the empirical filtering
            mc_rets.append(r)
            # roll state
            p_today = HestonParams(
                kappa=params.kappa, theta=params.theta, xi=params.xi, rho=params.rho,
                v0=float(v_path[-1]), s0=float(s_path[-1])
            )
        est = estimate_H_on_returns(mc_rets, dt_sec)
        H_est = float(est["H"])
        if np.isfinite(H_est):
            H_list.append(H_est)
            valid += 1

    H_arr = np.array(H_list, dtype=float)
    if verbose:
        print(f"valid runs: {valid}/{n_mc}")
    return {
        "empirical_H": float(real_est["H"]),
        "empirical_H_se": float(real_est["H_se"]),
        "days_used": int(real_est["days"]),
        "ac10_empirical": real_est["ac10"],
        "ac15_empirical": real_est["ac15"],
        "calibrated_params": params.__dict__,
        "dt_sec": dt_sec,
        "mc_runs": int(n_mc),
        "valid_runs": int(valid),
        "H_mc_mean": float(np.nanmean(H_arr)) if H_arr.size else float("nan"),
        "H_mc_std": float(np.nanstd(H_arr, ddof=1)) if H_arr.size > 1 else float("nan"),
        "H_mc_p5": float(np.nanpercentile(H_arr, 5)) if H_arr.size else float("nan"),
        "H_mc_p95": float(np.nanpercentile(H_arr, 95)) if H_arr.size else float("nan"),
    }

# ============================== Main ============================================

def main():
    args = parse_args()
    if args.mode == "paper_mc":
        dt_sec = args.dt
        steps_per_day = int(RTH_SECONDS // dt_sec)
        dt_years = dt_sec / (252.0 * RTH_SECONDS)
        p = HestonParams(kappa=6.0, theta=0.02, xi=0.4, rho=-0.5, v0=0.02, s0=100.0)
        rng = np.random.default_rng(args.seed)
        s_path, v_path = simulate_heston_day(p, dt_years, steps_per_day, rng)
        print(f"[paper_mc] Simulated one day with steps={steps_per_day}, dt={dt_sec}s. H label={args.H}")
        return

    if not args.csv:
        raise SystemExit("--csv is required for empirical_plus_mc_from_data")
    exclude_days = load_excludes(args.exclude_dates_csv)
    aligned = load_and_align(Path(args.csv), args.dt, exclude_days, verbose=args.verbose)
    if aligned is None or aligned.empty:
        raise SystemExit("No aligned data. Check CSV columns [DT, Price], timezone, or frequency.")
    summary = run_mc_from_data(aligned, dt_sec=args.dt, n_mc=args.n_mc,
                               seed=args.seed, jump_sigma=args.jump_sigma,
                               verbose=args.verbose)

    print("\n========== EMPIRICAL ==========")
    print(f"Grid: {summary['dt_sec']}s | Days used: {summary['days_used']}")
    print(f"H (empirical): {summary['empirical_H']:.3f}  (SE ≈ {summary['empirical_H_se']:.3f})")

    print("\n===== CALIBRATED HESTON (robust, no jumps/noise) =====")
    for k, v in summary["calibrated_params"].items():
        if isinstance(v, float):
            print(f"  {k:6s}: {v:.6g}")
        else:
            print(f"  {k:6s}: {v}")

    print("\n========== MONTE CARLO ==========")
    print(f"valid runs: {summary['valid_runs']}/{summary['mc_runs']}")
    print(f"Runs: {summary['mc_runs']}  |  H_mean={summary['H_mc_mean']:.3f}  "
          f"H_sd={summary['H_mc_std']:.3f}  "
          f"[p5, p95]=[{summary['H_mc_p5']:.3f}, {summary['H_mc_p95']:.3f}]")

if __name__ == "__main__":
    main()
