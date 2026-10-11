"""Black-Scholes-Merton pricing, Greeks, implied volatility and a CRR tree.

Every scalar function keeps its original signature. The new arguments (q, american, digits) are optional and come
last, so existing calls and the reference tests behave exactly as before.

q is a continuous dividend yield (q = 0 gives plain Black-Scholes).
Greeks are returned in model units: delta per 1.00 of spot, gamma per 1.00 of spot squared, vega per 1.00 of vol,
theta per year, rho per 1.00 of rate. The app converts them to desk units (per vol point, per day, per 1% rate).
"""
import math

import numpy as np
from scipy.special import ndtr


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _intrinsic(S: float, K: float, option_type: str) -> float:
    return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0
) -> float:
    """Calculates European option price via analytical Black-Scholes(-Merton) formula."""
    if T <= 0.0:
        return _intrinsic(S, K, option_type)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    fwd_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)

    if option_type.lower() == "call":
        return fwd_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
    elif option_type.lower() == "put":
        return disc_K * norm_cdf(-d2) - fwd_S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
    q: float = 0.0, digits=4,
) -> dict:
    """Calculates first and second-order analytical Greeks. digits=None returns unrounded values."""
    if T <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm_pdf(d1)
    is_call = option_type.lower() == "call"
    dq = math.exp(-q * T)
    dr = math.exp(-r * T)

    delta = dq * norm_cdf(d1) if is_call else dq * (norm_cdf(d1) - 1.0)
    gamma = dq * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * dq * pdf_d1 * sqrt_T
    decay = -(S * dq * sigma * pdf_d1) / (2.0 * sqrt_T)

    if is_call:
        theta = decay - r * K * dr * norm_cdf(d2) + q * S * dq * norm_cdf(d1)
        rho = K * T * dr * norm_cdf(d2)
    else:
        theta = decay + r * K * dr * norm_cdf(-d2) - q * S * dq * norm_cdf(-d1)
        rho = -K * T * dr * norm_cdf(-d2)

    out = {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
    return {k: (float(v) if digits is None else round(float(v), digits)) for k, v in out.items()}


def numerical_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    eps_s: float = 0.01,
    eps_vol: float = 0.0001,
    eps_t: float = 1e-6,
    eps_r: float = 0.0001,
    q: float = 0.0,
) -> dict:
    """Calculates Greeks via finite-difference numerical approximations."""
    base_px = black_scholes_price(S, K, T, r, sigma, option_type, q)

    px_up = black_scholes_price(S + eps_s, K, T, r, sigma, option_type, q)
    px_down = black_scholes_price(S - eps_s, K, T, r, sigma, option_type, q)
    num_delta = (px_up - px_down) / (2.0 * eps_s)
    num_gamma = (px_up - 2.0 * base_px + px_down) / (eps_s**2)

    px_vol_up = black_scholes_price(S, K, T, r, sigma + eps_vol, option_type, q)
    px_vol_down = black_scholes_price(S, K, T, r, sigma - eps_vol, option_type, q)
    num_vega = (px_vol_up - px_vol_down) / (2.0 * eps_vol)

    px_t_up = black_scholes_price(S, K, T + eps_t, r, sigma, option_type, q)
    px_t_down = black_scholes_price(S, K, max(1e-8, T - eps_t), r, sigma, option_type, q)
    num_theta = -(px_t_up - px_t_down) / (2.0 * eps_t)

    px_r_up = black_scholes_price(S, K, T, r + eps_r, sigma, option_type, q)
    px_r_down = black_scholes_price(S, K, T, max(0.0, r - eps_r), sigma, option_type, q)
    num_rho = (px_r_up - px_r_down) / (2.0 * eps_r)

    return {
        "num_delta": round(float(num_delta), 4),
        "num_gamma": round(float(num_gamma), 4),
        "num_theta": round(float(num_theta), 4),
        "num_vega": round(float(num_vega), 4),
        "num_rho": round(float(num_rho), 4),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
    q: float = 0.0,
) -> float:
    """Computes Implied Volatility using Newton-Raphson with bisection fallback and no-arbitrage boundary checks."""
    disc_strike = K * math.exp(-r * T)
    fwd_spot = S * math.exp(-q * T)
    is_call = option_type.lower() == "call"

    if is_call:
        lower = max(0.0, fwd_spot - disc_strike)
        upper = float(fwd_spot)
    elif option_type.lower() == "put":
        lower = max(0.0, disc_strike - fwd_spot)
        upper = float(disc_strike)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    if not (lower < market_price < upper):
        raise ValueError(
            f"Price {market_price:.4f} violates no-arbitrage boundary conditions ({lower:.4f}, {upper:.4f})"
        )

    # Initial volatility estimate
    sigma = math.sqrt(2.0 * math.pi / T) * (market_price / S)
    sigma = max(0.05, min(sigma, 3.0))

    # Newton-Raphson iteration
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price

        if abs(diff) < tol:
            return round(sigma, 6)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = fwd_spot * norm_pdf(d1) * math.sqrt(T)

        if vega < 1e-8:
            break

        sigma -= diff / vega
        if sigma < 0.001 or sigma > 5.0:
            break

    # Fallback to bisection on [0.001, 5.0]
    low, high = 0.001, 5.0
    p_low = black_scholes_price(S, K, T, r, low, option_type, q)
    p_high = black_scholes_price(S, K, T, r, high, option_type, q)

    if market_price < p_low or market_price > p_high:
        raise ValueError(
            f"Implied volatility has no root within sane range [0.001, 5.0] (pricing bounds: [{p_low:.4f}, {p_high:.4f}])"
        )

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        p = black_scholes_price(S, K, T, r, mid, option_type, q)
        if abs(p - market_price) < tol or (high - low) < 1e-7:
            return round(mid, 6)
        if p > market_price:
            high = mid
        else:
            low = mid

    return round((low + high) / 2.0, 6)


def crr_binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "call",
    q: float = 0.0,
    american: bool = False,
    digits=4,
) -> float:
    """Prices an option on a Cox-Ross-Rubinstein binomial tree.

    american=False gives the European value (converges to Black-Scholes), american=True allows early exercise
    at every node. digits=None returns the unrounded value (used for convergence studies).
    """
    if T <= 0.0:
        return _intrinsic(S, K, option_type)

    dt = T / N
    step = sigma * math.sqrt(dt)
    u = math.exp(step)
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Tree is unstable for these inputs (risk neutral probability outside 0 to 1). Increase N.")
    discount = math.exp(-r * dt)
    sign = 1.0 if option_type.lower() == "call" else -1.0

    j = np.arange(N + 1)
    # Terminal payoffs at step N. Node j has (N - j) up moves and j down moves.
    spots = S * np.exp((N - 2 * j) * step)
    values = np.maximum(sign * (spots - K), 0.0)

    # Backward induction to step 0
    for i in range(N - 1, -1, -1):
        values = discount * (p * values[: i + 1] + (1.0 - p) * values[1: i + 2])
        if american:
            spots_i = S * np.exp((i - 2 * j[: i + 1]) * step)
            values = np.maximum(values, sign * (spots_i - K))

    out = float(values[0])
    return out if digits is None else round(out, digits)


# ---------------------------------------------------------------------------------------------- vectorised
def bs_np(S, K, T, r, sigma, is_call=True, q=0.0):
    """Black-Scholes-Merton price and Greeks on numpy arrays (any broadcastable shapes), unrounded.

    Used for grids (scenario heatmaps, curves, option chains). Matches the scalar functions to machine precision.
    Returns a dict of arrays: price, delta, gamma, vega, theta (per year), rho.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    S, K, T, r, sigma, q = np.broadcast_arrays(S, K, T, r, sigma, q)
    call = bool(is_call)
    alive = T > 0.0
    Ts = np.where(alive, T, 1.0)
    sig = np.where(sigma > 0.0, sigma, 1e-12)
    sqrt_T = np.sqrt(Ts)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * Ts) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    dq, dr = np.exp(-q * Ts), np.exp(-r * Ts)
    pdf = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    decay = -(S * dq * sig * pdf) / (2.0 * sqrt_T)

    if call:
        price = S * dq * ndtr(d1) - K * dr * ndtr(d2)
        delta = dq * ndtr(d1)
        theta = decay - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1)
        rho = K * Ts * dr * ndtr(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        price = K * dr * ndtr(-d2) - S * dq * ndtr(-d1)
        delta = dq * (ndtr(d1) - 1.0)
        theta = decay + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1)
        rho = -K * Ts * dr * ndtr(-d2)
        intrinsic = np.maximum(K - S, 0.0)

    gamma = dq * pdf / (S * sig * sqrt_T)
    vega = S * dq * pdf * sqrt_T
    zero = np.zeros_like(price)
    return {
        "price": np.where(alive, price, intrinsic),
        "delta": np.where(alive, delta, zero),
        "gamma": np.where(alive, gamma, zero),
        "vega": np.where(alive, vega, zero),
        "theta": np.where(alive, theta, zero),
        "rho": np.where(alive, rho, zero),
    }


if __name__ == "__main__":
    spot = 100.0
    strike = 100.0
    tenor = 1.0
    rate = 0.05
    vol = 0.20

    # 1. Black-Scholes Pricing
    bs_call = black_scholes_price(spot, strike, tenor, rate, vol, "call")
    bs_put = black_scholes_price(spot, strike, tenor, rate, vol, "put")

    # 2. Greeks
    analytical = bs_greeks(spot, strike, tenor, rate, vol, "call")
    numerical = numerical_greeks(spot, strike, tenor, rate, vol, "call")

    # 3. Binomial Tree Pricing
    crr_call = crr_binomial_tree(spot, strike, tenor, rate, vol, N=250, option_type="call")
    crr_put = crr_binomial_tree(spot, strike, tenor, rate, vol, N=250, option_type="put")
    amer_put = crr_binomial_tree(spot, strike, tenor, rate, vol, N=250, option_type="put", american=True)

    # 4. Implied Volatility Inversion
    inverted_iv = implied_volatility(bs_call, spot, strike, tenor, rate, "call")

    print(f"Call Price (Black-Scholes): {bs_call:.4f} | Binomial CRR: {crr_call:.4f}")
    print(f"Put Price  (Black-Scholes): {bs_put:.4f}  | Binomial CRR: {crr_put:.4f} | American CRR: {amer_put:.4f}")

    print("\n--- GREEKS CROSS-CHECK ---")
    print(f"Delta -> Analytical: {analytical['delta']} | Numerical: {numerical['num_delta']}")
    print(f"Gamma -> Analytical: {analytical['gamma']} | Numerical: {numerical['num_gamma']}")
    print(f"Vega  -> Analytical: {analytical['vega']}  | Numerical: {numerical['num_vega']}")

    print("\n--- IMPLIED VOLATILITY INVERSION ---")
    print(f"Observed Market Price: {bs_call:.4f}")
    print(f"Inverted Implied Volatility: {inverted_iv * 100:.2f}% (Expected: {vol * 100:.2f}%)")
