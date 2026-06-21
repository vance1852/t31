from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    D: float
    alpha: float
    r2: float
    lag_start_s: float
    lag_end_s: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    fit_points: Optional[list[dict]] = None
    params: dict = field(default_factory=dict)
    aic: float = np.inf
    bic: float = np.inf
    residuals: Optional[np.ndarray] = None


@dataclass
class ModelDiscrimination:
    model_type: str
    reason: str
    scores: dict = field(default_factory=dict)


def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _aic_bic(n: int, k: int, ss_res: float) -> tuple[float, float]:
    if n <= k or ss_res <= 0:
        return np.inf, np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    return float(aic), float(bic)


def _brownian_model(t: np.ndarray, D: float, alpha: float) -> np.ndarray:
    return 4.0 * D * (t ** alpha)


def _confined_model(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
    return A * (1.0 - np.exp(-t / tau)) + B * t


def _directed_model(t: np.ndarray, D: float, v: float) -> np.ndarray:
    return 4.0 * D * t + (v * t) ** 2


def fit_brownian(
    msd_df: pd.DataFrame,
    min_lags: int = 5,
    max_ratio: float = 0.4,
    confidence: float = 0.95,
) -> FitResult:
    empty_result = FitResult(
        D=np.nan,
        alpha=np.nan,
        r2=0.0,
        lag_start_s=0.0,
        lag_end_s=0.0,
    )

    if msd_df.empty or len(msd_df) < min_lags:
        return empty_result

    data = msd_df.copy().sort_values("lag_time_s").reset_index(drop=True)
    times = data["lag_time_s"].values.astype(float)
    msds = data["msd"].values.astype(float)

    valid_mask = (times > 0) & (msds > 0) & (~np.isnan(times)) & (~np.isnan(msds))
    times = times[valid_mask]
    msds = msds[valid_mask]

    if len(times) < min_lags:
        return empty_result

    n_points = len(times)
    max_end = max(min_lags, int(n_points * max_ratio))
    if max_end > n_points:
        max_end = n_points
    if max_end < min_lags:
        max_end = n_points

    best_r2 = -np.inf
    best_start = 0
    best_end = max_end
    best_popt: Optional[np.ndarray] = None
    best_pcov: Optional[np.ndarray] = None

    for start in range(0, max(n_points - min_lags + 1, 1)):
        for end in range(start + min_lags, min(start + max_end + 1, n_points + 1)):
            if end - start < min_lags:
                continue

            t_sub = times[start:end]
            m_sub = msds[start:end]

            if len(t_sub) < 2:
                continue

            try:
                log_t = np.log(t_sub)
                log_m = np.log(m_sub)
                slope, intercept = np.polyfit(log_t, log_m, 1)
                alpha_init = max(0.1, min(2.0, slope))
                D_init = np.exp(intercept) / 4.0
                D_init = max(1e-6, D_init)

                bounds = ([1e-12, 0.05], [1e6, 3.0])
                popt, pcov = curve_fit(
                    _brownian_model,
                    t_sub,
                    m_sub,
                    p0=[D_init, alpha_init],
                    bounds=bounds,
                    maxfev=10000,
                )

                m_pred = _brownian_model(t_sub, *popt)
                r2 = _calculate_r2(m_sub, m_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_start = start
                    best_end = end
                    best_popt = popt
                    best_pcov = pcov
            except Exception:
                continue

    if best_popt is None:
        return empty_result

    t_fit = times[best_start:best_end]
    m_fit = msds[best_start:best_end]
    m_pred = _brownian_model(t_fit, *best_popt)
    residuals = m_fit - m_pred
    ss_res = np.sum(residuals ** 2)
    aic, bic = _aic_bic(len(t_fit), 2, ss_res)

    D_val = float(best_popt[0])
    alpha_val = float(best_popt[1])

    ci_low, ci_high = None, None
    if best_pcov is not None:
        try:
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - confidence) / 2)
            perr = np.sqrt(np.diag(best_pcov))
            D_err = perr[0] * z
            ci_low = max(0.0, D_val - D_err)
            ci_high = D_val + D_err
        except Exception:
            pass

    fit_points = [
        {"lag_time_s": float(t), "msd": float(m), "msd_predicted": float(mp)}
        for t, m, mp in zip(t_fit, m_fit, m_pred)
    ]

    return FitResult(
        D=D_val,
        alpha=alpha_val,
        r2=float(max(0.0, best_r2)),
        lag_start_s=float(t_fit[0]),
        lag_end_s=float(t_fit[-1]),
        ci_low=ci_low,
        ci_high=ci_high,
        fit_points=fit_points,
        aic=float(aic),
        bic=float(bic),
        residuals=residuals,
    )


def fit_confined(
    msd_df: pd.DataFrame,
    min_lags: int = 5,
    max_ratio: float = 0.8,
    confidence: float = 0.95,
) -> FitResult:
    empty_result = FitResult(
        D=np.nan,
        alpha=np.nan,
        r2=0.0,
        lag_start_s=0.0,
        lag_end_s=0.0,
    )

    if msd_df.empty or len(msd_df) < min_lags:
        return empty_result

    data = msd_df.copy().sort_values("lag_time_s").reset_index(drop=True)
    times = data["lag_time_s"].values.astype(float)
    msds = data["msd"].values.astype(float)

    valid_mask = (times > 0) & (msds > 0) & (~np.isnan(times)) & (~np.isnan(msds))
    times = times[valid_mask]
    msds = msds[valid_mask]

    if len(times) < min_lags:
        return empty_result

    n_points = len(times)
    use_end = max(min_lags, int(n_points * max_ratio))
    t_fit = times[:use_end]
    m_fit = msds[:use_end]

    if len(t_fit) < min_lags:
        return empty_result

    try:
        A_init = max(m_fit[-1] * 0.8, m_fit.max() * 0.5)
        tau_init = t_fit[len(t_fit) // 2]
        slope = (m_fit[-1] - m_fit[-5]) / (t_fit[-1] - t_fit[-5]) if len(t_fit) >= 5 else 0.0
        B_init = max(0.0, slope)
        if B_init <= 0:
            B_init = 1e-3

        bounds = ([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(
            _confined_model,
            t_fit,
            m_fit,
            p0=[A_init, tau_init, B_init],
            bounds=bounds,
            maxfev=20000,
        )

        m_pred = _confined_model(t_fit, *popt)
        r2 = _calculate_r2(m_fit, m_pred)
        residuals = m_fit - m_pred
        ss_res = np.sum(residuals ** 2)
        aic, bic = _aic_bic(len(t_fit), 3, ss_res)

        A_val, tau_val, B_val = float(popt[0]), float(popt[1]), float(popt[2])
        D_eq = B_val / 4.0 if B_val > 0 else 0.0

        fit_points = [
            {"lag_time_s": float(t), "msd": float(m), "msd_predicted": float(mp)}
            for t, m, mp in zip(t_fit, m_fit, m_pred)
        ]

        return FitResult(
            D=D_eq,
            alpha=0.5,
            r2=float(max(0.0, r2)),
            lag_start_s=float(t_fit[0]),
            lag_end_s=float(t_fit[-1]),
            fit_points=fit_points,
            params={"A": A_val, "tau": tau_val, "B": B_val},
            aic=float(aic),
            bic=float(bic),
            residuals=residuals,
        )
    except Exception:
        return empty_result


def fit_directed(
    msd_df: pd.DataFrame,
    min_lags: int = 5,
    max_ratio: float = 0.6,
    confidence: float = 0.95,
) -> FitResult:
    empty_result = FitResult(
        D=np.nan,
        alpha=np.nan,
        r2=0.0,
        lag_start_s=0.0,
        lag_end_s=0.0,
    )

    if msd_df.empty or len(msd_df) < min_lags:
        return empty_result

    data = msd_df.copy().sort_values("lag_time_s").reset_index(drop=True)
    times = data["lag_time_s"].values.astype(float)
    msds = data["msd"].values.astype(float)

    valid_mask = (times > 0) & (msds > 0) & (~np.isnan(times)) & (~np.isnan(msds))
    times = times[valid_mask]
    msds = msds[valid_mask]

    if len(times) < min_lags:
        return empty_result

    n_points = len(times)
    use_end = max(min_lags, int(n_points * max_ratio))
    t_fit = times[:use_end]
    m_fit = msds[:use_end]

    if len(t_fit) < min_lags:
        return empty_result

    try:
        initial_slope = m_fit[-1] / t_fit[-1] if t_fit[-1] > 0 else 1.0
        D_init = max(1e-6, initial_slope / 8.0)
        v_init = np.sqrt(max(0.0, initial_slope / 2.0 - 2.0 * D_init))
        if np.isnan(v_init) or v_init < 0:
            v_init = 0.01

        bounds = ([0.0, 0.0], [np.inf, np.inf])
        popt, pcov = curve_fit(
            _directed_model,
            t_fit,
            m_fit,
            p0=[D_init, v_init],
            bounds=bounds,
            maxfev=15000,
        )

        m_pred = _directed_model(t_fit, *popt)
        r2 = _calculate_r2(m_fit, m_pred)
        residuals = m_fit - m_pred
        ss_res = np.sum(residuals ** 2)
        aic, bic = _aic_bic(len(t_fit), 2, ss_res)

        D_val, v_val = float(popt[0]), float(popt[1])

        fit_points = [
            {"lag_time_s": float(t), "msd": float(m), "msd_predicted": float(mp)}
            for t, m, mp in zip(t_fit, m_fit, m_pred)
        ]

        return FitResult(
            D=D_val,
            alpha=2.0,
            r2=float(max(0.0, r2)),
            lag_start_s=float(t_fit[0]),
            lag_end_s=float(t_fit[-1]),
            fit_points=fit_points,
            params={"v": v_val},
            aic=float(aic),
            bic=float(bic),
            residuals=residuals,
        )
    except Exception:
        return empty_result


def discriminate_model(
    msd_df: pd.DataFrame,
    results_brownian: FitResult,
    results_confined: Optional[FitResult] = None,
    results_directed: Optional[FitResult] = None,
) -> ModelDiscrimination:
    scores: dict[str, float] = {}
    scores["brownian_r2"] = float(results_brownian.r2)
    scores["brownian_aic"] = float(results_brownian.aic)
    scores["brownian_bic"] = float(results_brownian.bic)
    scores["alpha"] = float(results_brownian.alpha)

    has_confined = results_confined is not None and not np.isnan(results_confined.D)
    has_directed = results_directed is not None and not np.isnan(results_directed.D)

    if has_confined and results_confined is not None:
        scores["confined_r2"] = float(results_confined.r2)
        scores["confined_aic"] = float(results_confined.aic)
        scores["confined_bic"] = float(results_confined.bic)
    if has_directed and results_directed is not None:
        scores["directed_r2"] = float(results_directed.r2)
        scores["directed_aic"] = float(results_directed.aic)
        scores["directed_bic"] = float(results_directed.bic)
        scores["directed_v"] = float(results_directed.params.get("v", 0.0))

    alpha = results_brownian.alpha
    r2_brownian = results_brownian.r2

    directed_v_threshold = 0.5
    r2_good_threshold = 0.9
    r2_poor_threshold = 0.7

    if has_directed and results_directed is not None:
        v_val = results_directed.params.get("v", 0.0)
        if v_val > directed_v_threshold and results_directed.r2 > r2_brownian - 0.05:
            return ModelDiscrimination(
                model_type="directed",
                reason=f"主动漂移: v={v_val:.3f} um/s 超过阈值 {directed_v_threshold}",
                scores=scores,
            )

    if alpha > 1.2:
        if has_directed and results_directed is not None:
            return ModelDiscrimination(
                model_type="directed",
                reason=f"alpha={alpha:.2f} > 1.2, 可能存在主动漂移或残留漂移",
                scores=scores,
            )
        return ModelDiscrimination(
            model_type="superdiffusive",
            reason=f"alpha={alpha:.2f} > 1.2, 超扩散行为(可能是残留漂移)",
            scores=scores,
        )

    if alpha < 0.8 and r2_brownian < r2_good_threshold:
        if has_confined and results_confined is not None:
            if results_confined.aic < results_brownian.aic:
                return ModelDiscrimination(
                    model_type="confined",
                    reason=f"alpha={alpha:.2f} < 0.8 且受限扩散 AIC 更优",
                    scores=scores,
                )
        return ModelDiscrimination(
            model_type="subdiffusive",
            reason=f"alpha={alpha:.2f} < 0.8, 次扩散/受限行为",
            scores=scores,
        )

    if 0.9 <= alpha <= 1.1 and r2_brownian >= r2_good_threshold:
        return ModelDiscrimination(
            model_type="brownian",
            reason=f"alpha={alpha:.2f} 在 0.9-1.1 且 R2={r2_brownian:.3f} >= 0.9",
            scores=scores,
        )

    if r2_brownian >= r2_poor_threshold:
        return ModelDiscrimination(
            model_type="brownian",
            reason=f"布朗扩散(边缘): alpha={alpha:.2f}, R2={r2_brownian:.3f}",
            scores=scores,
        )

    return ModelDiscrimination(
        model_type="anomalous",
        reason=f"无法明确分类: alpha={alpha:.2f}, R2={r2_brownian:.3f}",
        scores=scores,
    )
