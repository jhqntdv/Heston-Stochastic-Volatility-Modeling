"""
Heston stochastic-volatility model: pricing, simulation, and calibration.

Model convention
----------------
dS_t / S_t = (r - q) dt + sqrt(v_t) dW^S_t
dv_t       = kappa(theta - v_t) dt + xi sqrt(v_t) dW^v_t
dW^S dW^v = rho dt

The module deliberately separates:
    1. model parameters / validation
    2. analytical variance-swap quantities
    3. Heston characteristic-function pricing
    4. Black-Scholes / implied volatility
    5. Monte Carlo variance schemes (FT and Andersen QE)
    6. price correction choices (None, EMS, Andersen QE martingale correction)
    7. market calibration

Important design choice
-----------------------
EMS and Andersen's QE martingale correction are alternative price-correction
methods. They should not be stacked. The realized-variance simulation does
not expose a price-correction switch because RV must be computed from the
raw Heston log returns, not from an EMS-adjusted spot process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, root_scalar
from scipy.stats import norm


# =============================================================================
# 1. Model parameters and configuration
# =============================================================================


@dataclass(frozen=True)
class HestonParameters:
    """Validated container for the five Heston parameters.""" 
    kappa: float
    theta: float
    xi: float
    rho: float
    v_0: float

    def __post_init__(self) -> None:
        values = {
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v_0": self.v_0,
        }
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("All Heston parameters must be finite.")
        if self.kappa <= 0:
            raise ValueError("kappa must be strictly positive.")
        if self.theta <= 0:
            raise ValueError("theta must be strictly positive.")
        if self.xi <= 0:
            raise ValueError("xi must be strictly positive.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie within [-1, 1].")
        if self.v_0 < 0:
            raise ValueError("v_0 must be non-negative.")

    @property
    def feller_ratio(self) -> float:
        """Return 2*kappa*theta/xi^2."""
        return 2.0 * self.kappa * self.theta / self.xi**2

    @property
    def is_feller_satisfied(self) -> bool:
        """Whether the classical CIR Feller condition is satisfied."""
        return self.feller_ratio >= 1.0

    def to_array(self) -> np.ndarray:
        """Return optimizer ordering [rho, kappa, theta, v_0, xi]."""
        return np.array(
            [self.rho, self.kappa, self.theta, self.v_0, self.xi],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "HestonParameters":
        """Construct from optimizer ordering [rho, kappa, theta, v_0, xi]."""
        if len(arr) != 5:
            raise ValueError("Heston parameter array must have length 5.")
        rho, kappa, theta, v_0, xi = arr
        return cls(
            kappa=float(kappa),
            theta=float(theta),
            xi=float(xi),
            rho=float(rho),
            v_0=float(v_0),
        )

    def summary(self) -> pd.DataFrame:
        """Return a compact parameter table."""
        return pd.DataFrame(
            {
                "Parameter": [
                    "Mean Reversion Speed (kappa)",
                    "Long-Term Variance (theta)",
                    "Vol of Vol (xi)",
                    "Asset-Variance Correlation (rho)",
                    "Initial Variance (v_0)",
                ],
                "Symbol": ["κ", "θ", "ξ", "ρ", "v₀"],
                "Value": [self.kappa, self.theta, self.xi, self.rho, self.v_0],
            }
        )


class PriceCorrection(str, Enum):
    """Alternative corrections for the simulated spot-price process."""

    NONE = "none"
    EMS = "ems"
    ANDERSEN = "andersen"


@dataclass(frozen=True)
class HestonSimulationConfig:
    """Common simulation configuration and input validation."""

    n_steps: int = 252
    n_sims: int = 10_000
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive.")


@dataclass(frozen=True)
class HestonCalibrationConfig:
    """Calibration settings; market-specific assumptions live here.

    Attributes:
        bounds: Parameter search bounds for [rho, kappa, theta, v_0, xi].
        feller_penalty_weight: Weight for penalizing Feller condition violation:
            penalty = weight * max(0, xi^2 - 2*kappa*theta)^2. Note: in "msre"
            mode (where pricing loss is relative and small, ~1e-4), smaller weights
            provide gentle regularization, whereas larger weights strictly prioritize Feller.
        maxiter_global: Maximum iterations for Differential Evolution stage.
        maxiter_local: Maximum iterations for L-BFGS-B stage.
        population_size: Population size multiplier for Differential Evolution.
        seed: Random seed for Differential Evolution.
        price_floor_for_mape: Minimum price denominator for relative error and MAPE.
        loss_type: Objective loss formulation: 'sse' (price sum of squared errors)
            or 'msre' (mean squared relative error weighted by 1 / price^2).
    """

    # Optimizer ordering: [rho, kappa, theta, v_0, xi]
    bounds: Tuple[Tuple[float, float], ...] = (
        (-0.95, 0.20),
        (0.20, 5.00),
        (0.01, 0.40),
        (0.01, 0.40),
        (0.05, 1.50),
    )
    feller_penalty_weight: float = 0.0
    maxiter_global: int = 35
    maxiter_local: int = 200
    population_size: int = 10
    seed: int = 42
    price_floor_for_mape: float = 0.01
    loss_type: str = "sse"

    def __post_init__(self) -> None:
        if len(self.bounds) != 5:
            raise ValueError("Calibration bounds must contain five parameter ranges.")
        for low, high in self.bounds:
            if not low < high:
                raise ValueError("Every calibration bound must satisfy low < high.")
        if self.feller_penalty_weight < 0:
            raise ValueError("feller_penalty_weight must be non-negative.")
        if self.maxiter_global <= 0 or self.maxiter_local <= 0:
            raise ValueError("Optimizer iteration limits must be positive.")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.price_floor_for_mape <= 0:
            raise ValueError("price_floor_for_mape must be positive.")
        if self.loss_type not in ("sse", "msre"):
            raise ValueError("loss_type must be either 'sse' or 'msre'.")


# =============================================================================
# 2. Analytical variance-swap quantities
# =============================================================================


class AnalyticalVarianceSwapPricer:
    """Closed-form quantities for continuously monitored Heston variance."""

    @staticmethod
    def fair_variance_strike(params: HestonParameters, T: float) -> float:
        """
        Return K_var = E[(1/T) integral_0^T v_t dt].

        Under Heston/CIR:
            K_var = theta + (v_0-theta) * (1-exp(-kappa*T))/(kappa*T).
        """
        if T < 0:
            raise ValueError("T must be non-negative.")
        if T == 0:
            return float(params.v_0)

        x = params.kappa * T
        # expm1 improves cancellation for small x:
        factor = -np.expm1(-x) / x if abs(x) > 1e-8 else 1.0 - x / 2.0
        return float(params.theta + (params.v_0 - params.theta) * factor)

    @staticmethod
    def fair_volatility_strike(params: HestonParameters, T: float) -> float:
        """Return sqrt of the continuous fair variance strike."""
        return float(np.sqrt(AnalyticalVarianceSwapPricer.fair_variance_strike(params, T)))


# =============================================================================
# 3. Heston analytical European option pricer
# =============================================================================

@dataclass(frozen=True)
class HestonGreeks:
    """Analytical Greeks and sensitivities for Heston European options."""
    price: Union[float, np.ndarray]
    delta: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    rho: Union[float, np.ndarray]
    theta: Union[float, np.ndarray]
    vega: Union[float, np.ndarray]
    dkappa: Union[float, np.ndarray]
    dtheta: Union[float, np.ndarray]
    dxi: Union[float, np.ndarray]
    drho_param: Union[float, np.ndarray]



class HestonPricer:
    """
    Vectorized Heston European-option pricer using Gauss-Legendre quadrature.

    The characteristic function uses the stable 'little trap' formulation.
    The implementation supports a continuous dividend yield q while keeping
    q=0 as the backward-compatible default.
    """

    def __init__(self, n_quad_points: int = 256, u_max: float = 512.0) -> None:
        if n_quad_points <= 0:
            raise ValueError("n_quad_points must be positive.")
        if u_max <= 0:
            raise ValueError("u_max must be positive.")

        self.n_quad = int(n_quad_points)
        self.u_max = float(u_max)
        nodes, weights = np.polynomial.legendre.leggauss(self.n_quad)
        self.nodes = 0.5 * (nodes + 1.0) * self.u_max
        self.weights = 0.5 * weights * self.u_max

    @staticmethod
    def _validate_market_inputs(S0: float, T: float, r: float, q: float) -> None:
        if S0 <= 0 or not np.isfinite(S0):
            raise ValueError("S0 must be strictly positive and finite.")
        if T < 0 or not np.isfinite(T):
            raise ValueError("T must be non-negative and finite.")
        if not np.isfinite(r) or not np.isfinite(q):
            raise ValueError("r and q must be finite.")

    @staticmethod
    def char_func(
        S0: float,
        T: float,
        r: float,
        params: HestonParameters,
        w: np.ndarray,
        q: float = 0.0,
    ) -> np.ndarray:
        """Characteristic function of log(S_T) under the risk-neutral Heston model."""
        HestonPricer._validate_market_inputs(S0, T, r, q)
        w = np.asarray(w, dtype=np.complex128)

        if T == 0:
            return np.exp(1j * w * np.log(S0))

        kappa, theta, xi, rho, v_0 = (
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
            params.v_0,
        )

        alpha = -0.5 * (w**2) - 0.5j * w
        beta = kappa - rho * xi * 1j * w
        gamma = 0.5 * xi**2

        d = np.sqrt(beta**2 - 4.0 * alpha * gamma)
        r_plus = (beta + d) / xi**2
        r_minus = (beta - d) / xi**2
        g = r_minus / r_plus

        exp_dT = np.exp(-d * T)
        log_term = np.log((1.0 - g * exp_dT) / (1.0 - g))

        C = kappa * (r_minus * T - (2.0 / xi**2) * log_term)
        D = r_minus * (1.0 - exp_dT) / (1.0 - g * exp_dT)

        return np.exp(
            C * theta
            + D * v_0
            + 1j * w * (np.log(S0) + (r - q) * T)
        )

    def _pricing_engine(
        self,
        S0: float,
        K: Union[float, np.ndarray],
        T: float,
        r: float,
        params: HestonParameters,
        option_type: str,
        q: float,
        return_greeks: bool = False,
    ) -> Union[float, np.ndarray, HestonGreeks]:
        """Core unified engine for price and Greeks computation."""
        self._validate_market_inputs(S0, T, r, q)
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        if np.any(~np.isfinite(K_arr)) or np.any(K_arr <= 0):
            raise ValueError("All strikes must be strictly positive and finite.")

        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        if T == 0:
            if return_greeks:
                raise ValueError("Analytical Greeks not defined for T=0.")
            call = np.maximum(S0 - K_arr, 0.0)
            result = call if option_type == "call" else np.maximum(K_arr - S0, 0.0)
            return float(result[0]) if np.isscalar(K) else result

        w = self.nodes
        wt = self.weights

        logK = np.log(K_arr)[:, None]
        w_row = w[None, :]
        exp_kernel = np.exp(-1j * w_row * logK)
        forward_factor = np.exp((r - q) * T)

        if not return_greeks:
            cf1 = self.char_func(S0, T, r, params, w - 1j, q=q)
            cf2 = self.char_func(S0, T, r, params, w, q=q)
        else:
            cf1, dcf1 = self._heston_cf_and_derivs(
                S0, T, r, q, params.kappa, params.theta, params.xi, params.rho, params.v_0, w - 1j
            )
            cf2, dcf2 = self._heston_cf_and_derivs(
                S0, T, r, q, params.kappa, params.theta, params.xi, params.rho, params.v_0, w
            )

        i1 = np.real(exp_kernel * cf1[None, :] / (1j * w_row * S0 * forward_factor))
        i2 = np.real(exp_kernel * cf2[None, :] / (1j * w_row))

        P1 = 0.5 + np.sum(i1 * wt[None, :], axis=1) / np.pi
        P2 = 0.5 + np.sum(i2 * wt[None, :], axis=1) / np.pi

        call = np.maximum(0.0, S0 * np.exp(-q * T) * P1 - K_arr * np.exp(-r * T) * P2)
        if option_type == "call":
            price = call
        else:
            price = np.maximum(0.0, call + K_arr * np.exp(-r * T) - S0 * np.exp(-q * T))

        if not return_greeks:
            return float(price[0]) if np.isscalar(K) else price

        # Calculate Greeks
        delta = np.exp(-q * T) * P1 if option_type == "call" else np.exp(-q * T) * (P1 - 1.0)
        rho_rate = K_arr * T * np.exp(-r * T) * P2
        if option_type == "put":
            rho_rate -= K_arr * T * np.exp(-r * T)

        h1 = cf1 / S0
        dh1_dx = h1 * (1j * (w - 1j) - 1.0)
        di1_dx = np.real(exp_kernel * dh1_dx[None, :] / (1j * w_row * forward_factor))
        dP1_dx = np.sum(di1_dx * wt[None, :], axis=1) / np.pi
        gamma = np.exp(-q * T) * (dP1_dx / S0)
        
        dcf1_dT_eff = dcf1["T"] - cf1 * (r - q)
        di1_dT = np.real(exp_kernel * dcf1_dT_eff[None, :] / (1j * w_row * S0 * forward_factor))
        di2_dT = np.real(exp_kernel * dcf2["T"][None, :] / (1j * w_row))
        dP1_dT = np.sum(di1_dT * wt[None, :], axis=1) / np.pi
        dP2_dT = np.sum(di2_dT * wt[None, :], axis=1) / np.pi
        
        dcall_dT = (-q * S0 * np.exp(-q * T) * P1 + r * K_arr * np.exp(-r * T) * P2
                    + S0 * np.exp(-q * T) * dP1_dT - K_arr * np.exp(-r * T) * dP2_dT)
        theta_val = -dcall_dT / 365.0
        if option_type == "put":
            theta_val = -(dcall_dT + q * S0 * np.exp(-q * T) - r * K_arr * np.exp(-r * T)) / 365.0
            
        def _param_sens(key: str) -> np.ndarray:
            di1_p = np.real(exp_kernel * dcf1[key][None, :] / (1j * w_row * S0 * forward_factor))
            di2_p = np.real(exp_kernel * dcf2[key][None, :] / (1j * w_row))
            dP1_p = np.sum(di1_p * wt[None, :], axis=1) / np.pi
            dP2_p = np.sum(di2_p * wt[None, :], axis=1) / np.pi
            return S0 * np.exp(-q * T) * dP1_p - K_arr * np.exp(-r * T) * dP2_p

        return HestonGreeks(
            price=float(price[0]) if np.isscalar(K) else price,
            delta=float(delta[0]) if np.isscalar(K) else delta,
            gamma=float(gamma[0]) if np.isscalar(K) else gamma,
            rho=float(rho_rate[0]) if np.isscalar(K) else rho_rate,
            theta=float(theta_val[0]) if np.isscalar(K) else theta_val,
            vega=float(_param_sens("v0")[0]) if np.isscalar(K) else _param_sens("v0"),
            dkappa=float(_param_sens("kappa")[0]) if np.isscalar(K) else _param_sens("kappa"),
            dtheta=float(_param_sens("theta")[0]) if np.isscalar(K) else _param_sens("theta"),
            dxi=float(_param_sens("xi")[0]) if np.isscalar(K) else _param_sens("xi"),
            drho_param=float(_param_sens("rho")[0]) if np.isscalar(K) else _param_sens("rho"),
        )

    def price_options_fast(
        self,
        S0: float,
        K: Union[float, np.ndarray],
        T: float,
        r: float,
        params: HestonParameters,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """Price one or many European calls/puts using Fourier inversion."""
        return self._pricing_engine(S0, K, T, r, params, option_type, q, return_greeks=False)

    @staticmethod
    def _heston_cf_and_derivs(
        S0: float, T: float, r: float, q: float, 
        kappa: float, theta: float, xi: float, rho: float, v0: float, w: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute characteristic function and its analytical derivatives."""
        w = np.asarray(w, dtype=np.complex128)
        alpha = -0.5 * (w ** 2) - 0.5j * w
        beta = kappa - rho * xi * 1j * w
        gamma = 0.5 * xi ** 2
        xi2 = xi ** 2

        d = np.sqrt(beta ** 2 - 4.0 * alpha * gamma)
        r_plus = (beta + d) / xi2
        r_minus = (beta - d) / xi2
        g = r_minus / r_plus

        exp_dT = np.exp(-d * T)
        one_minus_g = 1.0 - g
        one_minus_g_exp = 1.0 - g * exp_dT
        log_term = np.log(one_minus_g_exp / one_minus_g)

        C = kappa * (r_minus * T - (2.0 / xi2) * log_term)
        D = r_minus * (1.0 - exp_dT) / one_minus_g_exp

        cf = np.exp(C * theta + D * v0 + 1j * w * (np.log(S0) + (r - q) * T))

        # Flattened derivatives (no dicts)
        dbeta_dkappa = np.ones_like(w)
        dbeta_drho = -1j * xi * w
        dbeta_dxi = -1j * rho * w
        dgamma_dxi = xi

        du_dkappa = 2.0 * beta * dbeta_dkappa
        du_drho = 2.0 * beta * dbeta_drho
        du_dxi = 2.0 * beta * dbeta_dxi - 4.0 * alpha * dgamma_dxi
        
        dd_dkappa = du_dkappa / (2.0 * d)
        dd_drho = du_drho / (2.0 * d)
        dd_dxi = du_dxi / (2.0 * d)

        dr_plus_dkappa = (dbeta_dkappa + dd_dkappa) / xi2
        dr_minus_dkappa = (dbeta_dkappa - dd_dkappa) / xi2
        
        dr_plus_drho = (dbeta_drho + dd_drho) / xi2
        dr_minus_drho = (dbeta_drho - dd_drho) / xi2

        dr_plus_dxi = (dbeta_dxi + dd_dxi) / xi2 - 2.0 * (beta + d) / (xi ** 3)
        dr_minus_dxi = (dbeta_dxi - dd_dxi) / xi2 - 2.0 * (beta - d) / (xi ** 3)

        dg_dkappa = (dr_minus_dkappa * r_plus - r_minus * dr_plus_dkappa) / (r_plus ** 2)
        dg_drho = (dr_minus_drho * r_plus - r_minus * dr_plus_drho) / (r_plus ** 2)
        dg_dxi = (dr_minus_dxi * r_plus - r_minus * dr_plus_dxi) / (r_plus ** 2)

        dexp_dT_dkappa = exp_dT * (-T * dd_dkappa)
        dexp_dT_drho = exp_dT * (-T * dd_drho)
        dexp_dT_dxi = exp_dT * (-T * dd_dxi)

        def dlog_term_fn(dg_val, dexp_dT_val):
            dnum = -(dg_val * exp_dT + g * dexp_dT_val)
            dden = -dg_val
            return dnum / one_minus_g_exp - dden / one_minus_g

        dlog_term_dkappa = dlog_term_fn(dg_dkappa, dexp_dT_dkappa)
        dlog_term_drho = dlog_term_fn(dg_drho, dexp_dT_drho)
        dlog_term_dxi = dlog_term_fn(dg_dxi, dexp_dT_dxi)

        dC_dkappa = (r_minus * T - (2.0 / xi2) * log_term) + kappa * (dr_minus_dkappa * T - (2.0 / xi2) * dlog_term_dkappa)
        dC_drho = kappa * (dr_minus_drho * T - (2.0 / xi2) * dlog_term_drho)
        dC_dxi = kappa * (dr_minus_dxi * T + (4.0 / xi**3) * log_term - (2.0 / xi2) * dlog_term_dxi)

        N = r_minus * (1.0 - exp_dT)
        Den = one_minus_g_exp
        
        def dD_fn(dr_minus_val, dexp_dT_val, dg_val):
            dN = dr_minus_val * (1.0 - exp_dT) - r_minus * dexp_dT_val
            dDen = -(dg_val * exp_dT + g * dexp_dT_val)
            return (dN * Den - N * dDen) / (Den ** 2)

        dD_dkappa = dD_fn(dr_minus_dkappa, dexp_dT_dkappa, dg_dkappa)
        dD_drho = dD_fn(dr_minus_drho, dexp_dT_drho, dg_drho)
        dD_dxi = dD_fn(dr_minus_dxi, dexp_dT_dxi, dg_dxi)

        dcf = {
            "kappa": cf * (dC_dkappa * theta + dD_dkappa * v0),
            "rho": cf * (dC_drho * theta + dD_drho * v0),
            "xi": cf * (dC_dxi * theta + dD_dxi * v0),
            "theta": cf * C,
            "v0": cf * D,
        }
        
        dlogterm_dT = g * d * exp_dT / one_minus_g_exp
        dC_dT = kappa * (r_minus - (2.0 / xi2) * dlogterm_dT)
        dD_dT = r_minus * d * exp_dT * one_minus_g / (one_minus_g_exp ** 2)
        dcf["T"] = cf * (dC_dT * theta + dD_dT * v0 + 1j * w * (r - q))

        return cf, dcf

    def price_and_greeks_fast(
        self,
        S0: float,
        K: Union[float, np.ndarray],
        T: float,
        r: float,
        params: HestonParameters,
        option_type: str = "call",
        q: float = 0.0,
    ) -> HestonGreeks:
        """Compute price and analytical Greeks in a single pass."""
        return self._pricing_engine(S0, K, T, r, params, option_type, q, return_greeks=True)


# =============================================================================
# 4. Black-Scholes pricing and implied volatility
# =============================================================================


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Analytical Greeks for Black-Scholes European options."""
    price: float
    delta: float
    gamma: float
    rho: float
    theta: float
    vega: float


class BlackScholesPricer:
    """Standard Black-Scholes European option pricer."""

    @staticmethod
    def price(
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        if not np.isfinite([S0, K, T, r, q, sigma]).all():
            raise ValueError("All Black-Scholes inputs must be finite.")
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be strictly positive.")
        if T < 0:
            raise ValueError("T must be non-negative.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")

        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        if T == 0 or sigma <= 1e-10:
            if option_type == "call":
                return float(max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T)))
            return float(max(0.0, K * np.exp(-r * T) - S0 * np.exp(-q * T)))

        sqrt_T = np.sqrt(T)
        d1 = (
            np.log(S0 / K)
            + (r - q + 0.5 * sigma**2) * T
        ) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        call_val = (
            S0 * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

        if option_type == "call":
            return float(call_val)
        return float(call_val + K * np.exp(-r * T) - S0 * np.exp(-q * T))

    @staticmethod
    def price_and_greeks(
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> BlackScholesGreeks:
        """Compute price and analytical Greeks for Black-Scholes."""
        if not np.isfinite([S0, K, T, r, q, sigma]).all():
            raise ValueError("All Black-Scholes inputs must be finite.")
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be strictly positive.")
        if T < 0:
            raise ValueError("T must be non-negative.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")

        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        if T == 0 or sigma <= 1e-10:
            # Greeks are largely degenerate or singular at T=0
            if option_type == "call":
                price = float(max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T)))
                delta = float(np.exp(-q * T) if S0 > K else 0.0)
            else:
                price = float(max(0.0, K * np.exp(-r * T) - S0 * np.exp(-q * T)))
                delta = float(-np.exp(-q * T) if K > S0 else 0.0)
            return BlackScholesGreeks(price=price, delta=delta, gamma=0.0, rho=0.0, theta=0.0, vega=0.0)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_minus_d1 = norm.cdf(-d1)
        cdf_minus_d2 = norm.cdf(-d2)

        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)

        # Base price
        if option_type == "call":
            price = S0 * exp_qT * cdf_d1 - K * exp_rT * cdf_d2
        else:
            price = K * exp_rT * cdf_minus_d2 - S0 * exp_qT * cdf_minus_d1

        # Greeks
        if option_type == "call":
            delta = exp_qT * cdf_d1
            rho = K * T * exp_rT * cdf_d2
            theta = -(S0 * exp_qT * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * K * exp_rT * cdf_d2 + q * S0 * exp_qT * cdf_d1
        else:
            delta = exp_qT * (cdf_d1 - 1.0)
            rho = -K * T * exp_rT * cdf_minus_d2
            theta = -(S0 * exp_qT * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * K * exp_rT * cdf_minus_d2 - q * S0 * exp_qT * cdf_minus_d1

        gamma = (exp_qT * pdf_d1) / (S0 * sigma * sqrt_T)
        vega = S0 * exp_qT * pdf_d1 * sqrt_T

        return BlackScholesGreeks(
            price=float(price),
            delta=float(delta),
            gamma=float(gamma),
            rho=float(rho),
            theta=float(theta),
            vega=float(vega)
        )


class ImpliedVolatilitySolver:
    """Brent root finder for Black-Scholes implied volatility."""

    @staticmethod
    def bs_call_price(
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        return BlackScholesPricer.price(S0, K, T, r, sigma, "call", q=q)

    @classmethod
    def calculate_implied_vol(
        cls,
        price: float,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        sigma_upper: float = 8.0,
    ) -> float:
        """Return call IV, or NaN for an invalid/no-solution price."""
        if not np.isfinite([price, S0, K, T, r, q]).all():
            return np.nan
        if S0 <= 0 or K <= 0 or T < 0 or price < 0:
            return np.nan

        pv_spot = S0 * np.exp(-q * T)
        pv_strike = K * np.exp(-r * T)
        intrinsic = max(0.0, pv_spot - pv_strike)
        upper_bound = pv_spot
        tol = 1e-10 * max(1.0, S0)

        if price < intrinsic - tol or price > upper_bound + tol:
            return np.nan
        if abs(price - intrinsic) <= tol:
            return 0.0
        if T == 0:
            return np.nan

        def objective(sigma: float) -> float:
            return cls.bs_call_price(S0, K, T, r, sigma, q=q) - price

        low = 1e-10
        high = sigma_upper
        f_low = objective(low)
        f_high = objective(high)

        # Expand the upper bracket instead of silently declaring failure.
        while f_high < 0.0 and high < 100.0:
            high *= 2.0
            f_high = objective(high)

        if f_low > 0.0 or f_high < 0.0:
            return np.nan

        try:
            sol = root_scalar(
                objective,
                bracket=[low, high],
                method="brentq",
                xtol=1e-10,
                rtol=1e-10,
            )
            return float(sol.root) if sol.converged else np.nan
        except (ValueError, RuntimeError):
            return np.nan

    @classmethod
    def calculate_surface(
        cls,
        prices: np.ndarray,
        S0: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        q: float = 0.0,
    ) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        strikes = np.asarray(strikes, dtype=np.float64)
        if prices.shape != strikes.shape:
            raise ValueError("prices and strikes must have the same shape.")
        vec_iv = np.vectorize(cls.calculate_implied_vol, otypes=[float])
        return vec_iv(prices, S0, strikes, T, r, q)


# =============================================================================
# 5. Monte Carlo utilities
# =============================================================================


class _HestonMonteCarloBase:
    """Shared input validation and EMS utility for Heston simulators."""

    def __init__(self, params: HestonParameters, seed: Optional[int] = 42) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_sim_inputs(
        S0: float,
        T: float,
        r: float,
        q: float,
        n_steps: int,
        n_sims: int,
    ) -> None:
        if S0 <= 0 or not np.isfinite(S0):
            raise ValueError("S0 must be strictly positive and finite.")
        if T <= 0 or not np.isfinite(T):
            raise ValueError("T must be strictly positive and finite.")
        if not np.isfinite(r) or not np.isfinite(q):
            raise ValueError("r and q must be finite.")
        if n_steps <= 0 or n_sims <= 0:
            raise ValueError("n_steps and n_sims must be positive.")

    @staticmethod
    def _apply_ems(
        S_raw: np.ndarray,
        S0: float,
        r: float,
        q: float,
        t: float,
    ) -> Tuple[np.ndarray, float]:
        """Scale the cross-sectional spot mean to the theoretical forward."""
        mean_raw = float(np.mean(S_raw))
        if not np.isfinite(mean_raw) or mean_raw <= 0:
            raise FloatingPointError("EMS failed because the simulated mean spot is invalid.")
        target_forward = S0 * np.exp((r - q) * t)
        correction = target_forward / mean_raw
        return S_raw * correction, correction


# =============================================================================
# 6. Full-Truncation Euler simulator
# =============================================================================


class HestonSimulatorFT(_HestonMonteCarloBase):
    """
    Full-Truncation Euler simulator.

    Variance update:
        V_raw = V + kappa(theta - V+)dt + xi sqrt(V+) dW_v
        V_next = max(V_raw, 0)

    EMS is optional for simulate_paths(). It is intentionally not part of
    simulate_realized_variance(), because EMS would change the log returns
    whose squared increments define the discrete realized variance.
    """

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        n_steps: int = 252,
        n_sims: int = 10_000,
        q: float = 0.0,
        price_correction: Union[str, PriceCorrection] = PriceCorrection.NONE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._validate_sim_inputs(S0, T, r, q, n_steps, n_sims)
        correction = PriceCorrection(price_correction)
        if correction == PriceCorrection.ANDERSEN:
            raise ValueError(
                "Andersen QE martingale correction is defined for the QE price scheme, "
                "not for the Full-Truncation Euler price scheme."
            )

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho, kappa, theta, xi = (
            self.params.rho,
            self.params.kappa,
            self.params.theta,
            self.params.xi,
        )
        sqrt_1_rho2 = np.sqrt(max(0.0, 1.0 - rho**2))

        prices = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
        variances = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
        time_grid = np.linspace(0.0, T, n_steps + 1)
        prices[:, 0] = S0
        variances[:, 0] = self.params.v_0

        for step in range(n_steps):
            Zs = self.rng.standard_normal(n_sims)
            Zv_ind = self.rng.standard_normal(n_sims)

            v_prev = variances[:, step]
            S_prev = prices[:, step]
            v_pos = np.maximum(v_prev, 0.0)

            dW_s = Zs * sqrt_dt
            dW_v = (rho * Zs + sqrt_1_rho2 * Zv_ind) * sqrt_dt

            log_ret = (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_s
            S_raw = S_prev * np.exp(log_ret)

            v_next_raw = (
                v_prev
                + kappa * (theta - v_pos) * dt
                + xi * np.sqrt(v_pos) * dW_v
            )
            v_next = np.maximum(v_next_raw, 0.0)

            if correction == PriceCorrection.EMS:
                S_next, _ = self._apply_ems(
                    S_raw, S0, r, q, (step + 1) * dt
                )
            else:
                S_next = S_raw

            prices[:, step + 1] = S_next
            variances[:, step + 1] = v_next

        return time_grid, prices, variances

    def simulate_realized_variance(
        self,
        S0: float,
        T: float,
        r: float,
        n_steps: int = 252,
        n_sims: int = 10_000,
        q: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate integrated variance and discrete realized variance.

        Returns
        -------
        integrated_variance:
            Trapezoidal approximation to (1/T) integral v_t dt.
        realized_variance:
            (1/T) sum_i (Delta log S_i)^2 using the raw Heston increments.
        """
        self._validate_sim_inputs(S0, T, r, q, n_steps, n_sims)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho, kappa, theta, xi = (
            self.params.rho,
            self.params.kappa,
            self.params.theta,
            self.params.xi,
        )
        sqrt_1_rho2 = np.sqrt(max(0.0, 1.0 - rho**2))

        v = np.full(n_sims, self.params.v_0, dtype=np.float64)
        integrated_v = np.zeros(n_sims, dtype=np.float64)
        sum_log_ret_sq = np.zeros(n_sims, dtype=np.float64)

        for _ in range(n_steps):
            Zs = self.rng.standard_normal(n_sims)
            Z2 = self.rng.standard_normal(n_sims)
            v_pos = np.maximum(v, 0.0)

            dW_s = Zs * sqrt_dt
            dW_v = (rho * Zs + sqrt_1_rho2 * Z2) * sqrt_dt

            log_ret = (
                (r - q - 0.5 * v_pos) * dt
                + np.sqrt(v_pos) * dW_s
            )
            sum_log_ret_sq += log_ret**2

            v_next_raw = (
                v
                + kappa * (theta - v_pos) * dt
                + xi * np.sqrt(v_pos) * dW_v
            )
            v_next = np.maximum(v_next_raw, 0.0)

            integrated_v += 0.5 * (v_pos + v_next) * dt
            v = v_next

        return integrated_v / T, sum_log_ret_sq / T


# Backward-compatible alias.
HestonSimulator = HestonSimulatorFT


# =============================================================================
# 7. Andersen (2008) QE simulator
# =============================================================================


class HestonSimulatorQE(_HestonMonteCarloBase):
    """
    Andersen (2008) Quadratic-Exponential variance simulator.

    Price correction choices:
        NONE      : use the uncorrected QE log-price discretization.
        EMS       : apply a cross-sectional martingale normalization.
        ANDERSEN  : use Andersen's conditional QE martingale correction.

    NONE / EMS use the same base QE log-price construction. ANDERSEN uses the
    K0-K4 price discretization for which the correction is derived.
    """

    def __init__(
        self,
        params: HestonParameters,
        seed: Optional[int] = 42,
        psi_crit: float = 1.5,
        gamma1: float = 0.5,
        gamma2: float = 0.5,
    ) -> None:
        super().__init__(params, seed=seed)
        if psi_crit <= 1.0:
            raise ValueError("psi_crit should be greater than 1.0 for standard QE branching.")
        if gamma1 < 0 or gamma2 < 0:
            raise ValueError("gamma1 and gamma2 must be non-negative.")
        self.psi_crit = float(psi_crit)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)

    def _qe_variance_step(
        self,
        v_curr: np.ndarray,
        dt: float,
        c1: float,
        c2: float,
        c3: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate V_next and return branch diagnostics needed by QE-M."""
        exp_kdt = np.exp(-self.params.kappa * dt)
        m = c1 + v_curr * exp_kdt
        s2 = v_curr * c2 + c3
        psi = s2 / np.maximum(m, 1e-300) ** 2

        Zv = self.rng.standard_normal(v_curr.shape[0])
        Uv = self.rng.random(v_curr.shape[0])
        v_next = np.zeros_like(v_curr)

        low_mask = psi <= self.psi_crit
        high_mask = ~low_mask

        # QE branch 1: quadratic transformation matching the first two moments.
        b2 = np.full_like(v_curr, np.nan)
        a = np.full_like(v_curr, np.nan)
        if np.any(low_mask):
            psi_low = psi[low_mask]
            inv_psi = 1.0 / np.maximum(psi_low, 1e-300)
            term = np.maximum(2.0 * inv_psi - 1.0, 0.0)
            b2_low = (
                2.0 * inv_psi
                - 1.0
                + np.sqrt(2.0 * inv_psi) * np.sqrt(term)
            )
            a_low = m[low_mask] / (1.0 + b2_low)
            v_next[low_mask] = a_low * (
                np.sqrt(np.maximum(b2_low, 0.0)) + Zv[low_mask]
            ) ** 2
            b2[low_mask] = b2_low
            a[low_mask] = a_low

        # QE branch 2: exponential distribution with an atom at zero.
        p = np.full_like(v_curr, np.nan)
        beta = np.full_like(v_curr, np.nan)
        if np.any(high_mask):
            psi_high = psi[high_mask]
            p_high = (psi_high - 1.0) / (psi_high + 1.0)
            beta_high = (1.0 - p_high) / np.maximum(m[high_mask], 1e-300)
            u_high = Uv[high_mask]
            positive = u_high > p_high
            values = np.zeros_like(u_high)
            values[positive] = (
                np.log(
                    (1.0 - p_high[positive])
                    / np.maximum(1.0 - u_high[positive], 1e-300)
                )
                / beta_high[positive]
            )
            v_next[high_mask] = values
            p[high_mask] = p_high
            beta[high_mask] = beta_high

        return v_next, low_mask, b2, a, p, beta

    def _qe_price_coefficients(self, dt: float) -> Tuple[float, float, float, float, float]:
        """Return Andersen central-scheme K0...K4 coefficients."""
        kappa, theta, xi, rho = (
            self.params.kappa,
            self.params.theta,
            self.params.xi,
            self.params.rho,
        )
        K0 = -(kappa * rho * theta / xi) * dt
        K1 = (kappa * rho / xi - 0.5) * self.gamma1 * dt - rho / xi
        K2 = (kappa * rho / xi - 0.5) * self.gamma2 * dt + rho / xi
        K3 = (1.0 - rho**2) * self.gamma1 * dt
        K4 = (1.0 - rho**2) * self.gamma2 * dt
        return K0, K1, K2, K3, K4

    def _andersen_martingale_corrected_k0(
        self,
        v_curr: np.ndarray,
        low_mask: np.ndarray,
        b2: np.ndarray,
        a: np.ndarray,
        p: np.ndarray,
        beta: np.ndarray,
        K1: float,
        K2: float,
        K3: float,
        K4: float,
    ) -> np.ndarray:
        """
        Compute pathwise K0* for Andersen's QE martingale correction.

        For branch 1, V_next = a(b + Z)^2 and
            E[exp(A V_next)]
              = exp(A a b^2/(1-2Aa)) / sqrt(1-2Aa).

        For branch 2, V_next has an atom p at zero and an exponential positive
        tail with rate beta, so
            E[exp(A V_next)] = p + (1-p) beta/(beta-A).

        Here A = K2 + 0.5*K4 and the correction enforces the conditional
        martingale condition for the QE price discretization.
        """
        A = K2 + 0.5 * K4
        base = (K1 + 0.5 * K3) * v_curr
        K0_star = np.full_like(v_curr, np.nan)

        if np.any(low_mask):
            idx = low_mask
            a_low = a[idx]
            b2_low = b2[idx]
            denom = 1.0 - 2.0 * A * a_low
            valid = denom > 1e-12
            if not np.all(valid):
                raise FloatingPointError(
                    "Andersen QE martingale correction is undefined for some "
                    "quadratic-branch paths (1 - 2*A*a <= 0). Reduce dt or "
                    "use a different price correction."
                )
            log_mgf = (
                A * b2_low * a_low / denom
                - 0.5 * np.log(denom)
            )
            K0_star[idx] = -base[idx] - log_mgf

        high_mask = ~low_mask
        if np.any(high_mask):
            idx = high_mask
            p_high = p[idx]
            beta_high = beta[idx]
            beta_minus_A = beta_high - A
            valid = beta_minus_A > 1e-12
            if not np.all(valid):
                raise FloatingPointError(
                    "Andersen QE martingale correction is undefined for some "
                    "exponential-branch paths (beta-A <= 0). Reduce dt or "
                    "use a different price correction."
                )
            mgf = p_high + (1.0 - p_high) * beta_high / beta_minus_A
            if np.any(~np.isfinite(mgf)) or np.any(mgf <= 0):
                raise FloatingPointError("Invalid exponential-branch QE martingale correction.")
            K0_star[idx] = -base[idx] - np.log(mgf)

        if np.any(~np.isfinite(K0_star)):
            raise FloatingPointError("Invalid K0* generated by Andersen QE martingale correction.")
        return K0_star

    def _simulate_one_step(
        self,
        v_curr: np.ndarray,
        lnS_curr: np.ndarray,
        dt: float,
        r: float,
        q: float,
        correction: PriceCorrection,
        c1: float,
        c2: float,
        c3: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one QE variance step and one corresponding log-price step."""
        kappa = self.params.kappa
        theta = self.params.theta
        rho = self.params.rho
        xi = self.params.xi
        sqrt_1_rho2 = np.sqrt(max(0.0, 1.0 - rho**2))

        v_next, low_mask, b2, a, p, beta = self._qe_variance_step(
            v_curr, dt, c1, c2, c3
        )

        if correction == PriceCorrection.ANDERSEN:
            K0, K1, K2, K3, K4 = self._qe_price_coefficients(dt)
            K0_star = self._andersen_martingale_corrected_k0(
                v_curr, low_mask, b2, a, p, beta, K1, K2, K3, K4
            )
            Zs = self.rng.standard_normal(v_curr.shape[0])
            log_ret = (
                (r - q) * dt
                + K0_star
                + K1 * v_curr
                + K2 * v_next
                + np.sqrt(
                    np.maximum(K3 * v_curr + K4 * v_next, 0.0)
                ) * Zs
            )
        else:
            # Same integrated-variance-based price approximation used by the
            # uncorrected and EMS variants. No sample normalization is applied here.
            integral_step = 0.5 * (v_curr + v_next) * dt
            int_sqrt_v_dWv = (
                v_next
                - v_curr
                - kappa * theta * dt
                + kappa * integral_step
            ) / xi
            Zs = self.rng.standard_normal(v_curr.shape[0])
            log_ret = (
                (r - q) * dt
                - 0.5 * integral_step
                + rho * int_sqrt_v_dWv
                + sqrt_1_rho2 * np.sqrt(np.maximum(integral_step, 0.0)) * Zs
            )

        return v_next, log_ret

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        n_steps: int = 252,
        n_sims: int = 10_000,
        q: float = 0.0,
        price_correction: Union[str, PriceCorrection] = PriceCorrection.NONE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._validate_sim_inputs(S0, T, r, q, n_steps, n_sims)
        correction = PriceCorrection(price_correction)
        dt = T / n_steps
        exp_kdt = np.exp(-self.params.kappa * dt)

        c1 = self.params.theta * (1.0 - exp_kdt)
        c2 = (
            self.params.xi**2
            * exp_kdt
            * (1.0 - exp_kdt)
            / self.params.kappa
        )
        c3 = (
            self.params.theta
            * self.params.xi**2
            * (1.0 - exp_kdt) ** 2
            / (2.0 * self.params.kappa)
        )

        time_grid = np.linspace(0.0, T, n_steps + 1)
        lnS = np.full(n_sims, np.log(S0), dtype=np.float64)
        v = np.full(n_sims, self.params.v_0, dtype=np.float64)
        lnS_paths = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
        v_paths = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
        lnS_paths[:, 0] = lnS
        v_paths[:, 0] = v

        for step in range(n_steps):
            v, raw_log_ret = self._simulate_one_step(
                v, lnS, dt, r, q, correction, c1, c2, c3
            )
            lnS_raw = lnS + raw_log_ret

            if correction == PriceCorrection.EMS:
                S_raw = np.exp(lnS_raw)
                S_next, _ = self._apply_ems(
                    S_raw, S0, r, q, (step + 1) * dt
                )
                lnS = np.log(S_next)
            else:
                lnS = lnS_raw

            lnS_paths[:, step + 1] = lnS
            v_paths[:, step + 1] = v

        return time_grid, np.exp(lnS_paths), v_paths

    def simulate_realized_variance(
        self,
        S0: float,
        T: float,
        r: float,
        n_steps: int = 252,
        n_sims: int = 10_000,
        q: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate integrated variance and discrete realized variance from the
        uncorrected QE log-price increments.
        """
        self._validate_sim_inputs(S0, T, r, q, n_steps, n_sims)
        dt = T / n_steps
        exp_kdt = np.exp(-self.params.kappa * dt)
        c1 = self.params.theta * (1.0 - exp_kdt)
        c2 = (
            self.params.xi**2
            * exp_kdt
            * (1.0 - exp_kdt)
            / self.params.kappa
        )
        c3 = (
            self.params.theta
            * self.params.xi**2
            * (1.0 - exp_kdt) ** 2
            / (2.0 * self.params.kappa)
        )

        v = np.full(n_sims, self.params.v_0, dtype=np.float64)
        integrated_v = np.zeros(n_sims, dtype=np.float64)
        sum_log_ret_sq = np.zeros(n_sims, dtype=np.float64)

        for _ in range(n_steps):
            v_next, log_ret = self._simulate_one_step(
                v, np.zeros_like(v), dt, r, q, PriceCorrection.NONE, c1, c2, c3
            )
            integral_step = 0.5 * (v + v_next) * dt
            integrated_v += integral_step
            sum_log_ret_sq += log_ret**2
            v = v_next

        return integrated_v / T, sum_log_ret_sq / T


# =============================================================================
# 8. Heston calibration
# =============================================================================


class HestonCalibrator:
    """Two-stage Heston surface calibration: Differential Evolution + L-BFGS-B."""

    def __init__(self, pricer: HestonPricer) -> None:
        self.pricer = pricer

    @staticmethod
    def _validate_quotes(market_quotes: List[Dict[str, Any]]) -> None:
        if not market_quotes:
            raise ValueError("market_quotes cannot be empty.")
        for quote in market_quotes:
            for key in ("strikes", "market_prices", "T", "r"):
                if key not in quote:
                    raise ValueError(f"Missing quote field: {key}")
            strikes = np.asarray(quote["strikes"], dtype=np.float64)
            prices = np.asarray(quote["market_prices"], dtype=np.float64)
            if strikes.shape != prices.shape or strikes.ndim != 1:
                raise ValueError("strikes and market_prices must be 1-D arrays of equal shape.")
            if np.any(strikes <= 0) or np.any(~np.isfinite(strikes)):
                raise ValueError("All strikes must be positive and finite.")
            if np.any(~np.isfinite(prices)) or np.any(prices < 0):
                raise ValueError("Market prices must be finite and non-negative.")
            if quote["T"] <= 0 or not np.isfinite(quote["T"]):
                raise ValueError("Each quote maturity T must be positive and finite.")
            if not np.isfinite(quote["r"]):
                raise ValueError("Each quote rate r must be finite.")
            if "q" in quote and not np.isfinite(quote["q"]):
                raise ValueError("Each quote dividend yield q must be finite.")

    def calibrate(
        self,
        S0: float,
        market_quotes: List[Dict[str, Any]],
        initial_guess: Optional[HestonParameters] = None,
        config: Optional[HestonCalibrationConfig] = None,
    ) -> Tuple[HestonParameters, Dict[str, Any]]:
        self._validate_quotes(market_quotes)
        if S0 <= 0 or not np.isfinite(S0):
            raise ValueError("S0 must be strictly positive and finite.")
        config = config or HestonCalibrationConfig()

        def loss_function(p_vec: np.ndarray) -> float:
            try:
                current_p = HestonParameters.from_array(p_vec)
            except ValueError:
                return np.inf

            total_loss = 0.0
            total_count = 0
            for quote in market_quotes:
                strikes = np.asarray(quote["strikes"], dtype=np.float64)
                market_prices = np.asarray(quote["market_prices"], dtype=np.float64)
                q = float(quote.get("q", 0.0))
                model_prices = self.pricer.price_options_fast(
                    S0,
                    strikes,
                    float(quote["T"]),
                    float(quote["r"]),
                    current_p,
                    option_type=quote.get("option_type", "call"),
                    q=q,
                )
                if np.any(~np.isfinite(model_prices)):
                    return np.inf
                
                diff = market_prices - model_prices
                if config.loss_type == "msre":
                    weights = 1.0 / np.maximum(market_prices, config.price_floor_for_mape)
                    total_loss += float(np.sum((diff * weights) ** 2))
                else:
                    total_loss += float(np.sum(diff ** 2))
                
                total_count += len(strikes)

            penalty = 0.0
            if config.feller_penalty_weight > 0:
                violation = max(
                    0.0,
                    current_p.xi**2 - 2.0 * current_p.kappa * current_p.theta,
                )
                penalty = config.feller_penalty_weight * violation**2

            return total_loss / total_count + penalty

        t_start = time.time()
        x0 = (
            initial_guess.to_array()
            if initial_guess is not None
            else np.array([-0.1, 1.0, 0.1, 0.1, 0.5], dtype=np.float64)
        )

        # Let scipy handle population initialization while supplying x0 as a
        # candidate. This avoids coupling calibration behavior to a hard-coded point.
        res_global = differential_evolution(
            loss_function,
            bounds=config.bounds,
            x0=x0,
            seed=config.seed,
            maxiter=config.maxiter_global,
            popsize=config.population_size,
            polish=False,
        )

        res_local = minimize(
            loss_function,
            res_global.x,
            bounds=config.bounds,
            method="L-BFGS-B",
            options={
                "maxiter": config.maxiter_local,
                "ftol": 1e-9,
            },
        )

        elapsed = time.time() - t_start
        calibrated_params = HestonParameters.from_array(res_local.x)

        total_se = 0.0
        total_ae = 0.0
        total_ape = 0.0
        max_error = 0.0
        n_obs = 0

        for quote in market_quotes:
            strikes = np.asarray(quote["strikes"], dtype=np.float64)
            market_prices = np.asarray(quote["market_prices"], dtype=np.float64)
            q = float(quote.get("q", 0.0))
            model_prices = self.pricer.price_options_fast(
                S0,
                strikes,
                float(quote["T"]),
                float(quote["r"]),
                calibrated_params,
                option_type=quote.get("option_type", "call"),
                q=q,
            )
            diff = np.abs(market_prices - model_prices)
            total_se += float(np.sum(diff**2))
            total_ae += float(np.sum(diff))
            total_ape += float(
                np.sum(diff / np.maximum(market_prices, config.price_floor_for_mape))
            )
            max_error = max(max_error, float(np.max(diff)))
            n_obs += len(strikes)

        diagnostics = {
            "Elapsed Time (s)": round(elapsed, 2),
            "RMSE": float(np.sqrt(total_se / n_obs)),
            "MAE": float(total_ae / n_obs),
            "MAPE (%)": float(total_ape / n_obs * 100.0),
            "Max Abs Error": float(max_error),
            "Objective": float(res_local.fun),
            "Feller Ratio": float(calibrated_params.feller_ratio),
            "Feller Satisfied": calibrated_params.is_feller_satisfied,
            "Global Success": bool(res_global.success),
            "Global Message": str(res_global.message),
            "Local Success": bool(res_local.success),
            "Local Message": str(res_local.message),
            "Optimizer Function Evaluations": int(res_local.nfev),
        }
        return calibrated_params, diagnostics


__all__ = [
    "HestonParameters",
    "PriceCorrection",
    "HestonSimulationConfig",
    "HestonCalibrationConfig",
    "AnalyticalVarianceSwapPricer",
    "HestonPricer",
    "BlackScholesPricer",
    "ImpliedVolatilitySolver",
    "HestonSimulatorFT",
    "HestonSimulatorQE",
    "HestonSimulator",
    "HestonCalibrator",
    "HestonGreeks",
    "BlackScholesGreeks",
]
