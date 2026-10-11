"""Delta-hedging engine for an options market maker. Pure numpy, no Streamlit.

The market maker holds a book of European options (qty > 0 means the desk is long, qty < 0 short, in shares of underlying).
Option marks and the hedge ratio come from Black-Scholes at the desk's implied vol. The real price process can differ
(realised vol, jumps), which is exactly where discrete hedging loses or makes money.

Accounting, per path (everything marked to mid, financed at r):

    Pi_t = V_t + B_t + h_t * S_t,   Pi_0 = 0 before the first trade (the desk pays V_0 for the book, so B_0 = -V_0)

V is the option book mark, B the cash account, h the stock hedge in shares. A trade of d shares costs
    |d| * (S * spread_bps / 1e4 + fee) + S * (impact_bps / 1e4) * d^2 / Q
(half spread and per share fee are linear, impact is quadratic and equals impact_bps at a trade the size of the whole book Q).
At the end of the horizon the remaining stock hedge is unwound at the same cost, so the final P&L is fully realised.

    Pi_T = [V_T - V_0] + sum h (S' - S) + financing - costs        (the four pieces shown in the P&L bridge)

The option book, its delta and its gamma depend only on the price path, not on the hedge, so they are computed once per path
and every strategy is run on exactly the same shocks (paired comparison).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

DAYS = 252.0          # trading days per year
G0 = 0.2              # reward / risk unit: 20% of the one step 1-sigma move of the whole book, see risk_unit()


# ------------------------------------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class Market:
    S0: float = 100.0
    sigma_imp: float = 0.25     # vol used by the desk to mark the book and compute delta
    sigma_real: float = 0.25    # vol of the simulated price process
    r: float = 0.03
    mu: float | None = None     # real world drift, None means r
    jump_lambda: float = 0.0    # jumps per year
    jump_mean: float = -0.02    # mean log jump
    jump_sd: float = 0.04
    vol_disp: float = 0.0       # lognormal dispersion of realised vol across paths (0 = constant vol)


@dataclass(frozen=True)
class Costs:
    spread_bps: float = 2.0     # half spread, bps of notional traded
    fee: float = 0.0            # $ per share
    impact_bps: float = 0.0     # extra bps of notional when a single trade is as large as the whole book


@dataclass(frozen=True)
class Horizon:
    horizon_days: int = 30      # how long the desk runs the hedge
    expiry_days: int = 30       # option expiry (>= horizon, if longer the book is marked at the end)
    steps_per_day: int = 1      # hedging opportunities per trading day


@dataclass
class Book:
    K: np.ndarray               # strikes, shape (n,) or (B, n)
    qty: np.ndarray             # signed shares of underlying, same shape
    is_call: np.ndarray         # bool, same shape

    @property
    def gross(self):
        """Q: gross option notional in shares, the book size every hedge quantity is measured against."""
        return np.abs(self.qty).sum(axis=-1)


PRESETS = ("Short ATM straddle", "Short OTM strangle", "Long ATM straddle",
           "Risk reversal (short put, long call)", "Customer flow (random book)")


def make_book(name: str, S0: float, contracts: int = 50, mult: int = 100, seed: int = 0) -> Book:
    n = contracts * mult
    if name == "Short ATM straddle":
        return Book(np.array([S0, S0]), np.array([-n, -n], float), np.array([True, False]))
    if name == "Short OTM strangle":
        return Book(np.array([1.05 * S0, 0.95 * S0]), np.array([-n, -n], float), np.array([True, False]))
    if name == "Long ATM straddle":
        return Book(np.array([S0, S0]), np.array([n, n], float), np.array([True, False]))
    if name.startswith("Risk reversal"):
        return Book(np.array([1.05 * S0, 0.95 * S0]), np.array([n, -n], float), np.array([True, False]))
    return random_book(S0, contracts, mult, np.random.default_rng(seed))


def random_book(S0, contracts, mult, rng, slots=5) -> Book:
    """Customer flow: 2 to 5 positions on a strike grid around spot, random side and size, total about `contracts`."""
    grid = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]) * S0
    k = int(rng.integers(2, slots + 1))
    K = np.zeros(slots)
    qty = np.zeros(slots)
    call = np.zeros(slots, bool)
    K[:k] = rng.choice(grid, size=k)
    w = rng.uniform(0.3, 1.0, size=k)
    sizes = np.maximum(1, np.round(contracts * w / w.sum())).astype(int) * mult
    sides = rng.choice([-1.0, 1.0], size=k)
    if np.all(sides == sides[0]) and k > 1:
        sides[0] = -sides[0]
    qty[:k] = sides * sizes
    call[:k] = rng.random(k) < 0.5
    K[k:] = S0
    return Book(K, qty, call)


# ------------------------------------------------------------------------------------------------ pricing
def _bs_terms(S, K, tau, r, sig):
    """Call price, call delta, gamma for broadcastable inputs. tau <= 0 gives intrinsic value and zero Greeks."""
    alive = tau > 0.0
    t = np.where(alive, tau, 1.0)
    sq = np.sqrt(t)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * t) / (sig * sq)
    d2 = d1 - sig * sq
    disc = np.exp(-r * t)
    call = S * ndtr(d1) - K * disc * ndtr(d2)
    delta = ndtr(d1)
    gamma = np.exp(-0.5 * d1 ** 2) / (np.sqrt(2.0 * np.pi) * S * sig * sq)
    call = np.where(alive, call, np.maximum(S - K, 0.0))
    delta = np.where(alive, delta, 0.0)
    gamma = np.where(alive, gamma, 0.0)
    return call, delta, gamma, disc


def book_state(S, book: Book, tau, r, sigma_imp):
    """Value V, delta D (shares) and gamma G (shares per $) of the book along price paths.

    S: (n, T+1), tau: (T+1,) or (n, T+1) years to expiry, sigma_imp and r: scalar or (n,). Book arrays are (P,) or (n, P).
    """
    S3 = S[:, :, None]
    K = np.atleast_2d(book.K)[:, None, :]
    q = np.atleast_2d(book.qty)[:, None, :]
    isc = np.atleast_2d(book.is_call)[:, None, :]
    sig = np.asarray(sigma_imp, float)
    sig = sig[:, None, None] if sig.ndim == 1 else sig
    rr = np.asarray(r, float)
    rr = rr[:, None, None] if rr.ndim == 1 else rr
    tau = np.asarray(tau, float)
    tau3 = tau[None, :, None] if tau.ndim == 1 else tau[:, :, None]
    call, dcall, gam, disc = _bs_terms(S3, K, tau3, rr, sig)
    put = call - S3 + K * disc
    put = np.where(tau3 > 0.0, put, np.maximum(K - S3, 0.0))
    px = np.where(isc, call, put)
    dl = np.where(isc, dcall, np.where(tau3 > 0.0, dcall - 1.0, 0.0))
    return (px * q).sum(-1), (dl * q).sum(-1), (gam * q).sum(-1)


# ------------------------------------------------------------------------------------------------ simulation
@dataclass
class Paths:
    S: np.ndarray               # (n, T+1)
    V: np.ndarray               # book mark
    D: np.ndarray               # book delta, shares
    G: np.ndarray               # book gamma, shares per $
    tau: np.ndarray             # years to expiry at each step, (T+1,) or (n, T+1)
    Q: np.ndarray               # (n,) gross book size
    sigma_imp: np.ndarray       # (n,)
    r: np.ndarray               # (n,)
    spread_bps: np.ndarray      # (n,)
    fee: np.ndarray
    impact_bps: np.ndarray
    kappa: np.ndarray           # (n,) dimensionless risk aversion
    dt: float                   # years per step
    T: int                      # steps
    spd: int

    @property
    def n(self):
        return self.S.shape[0]

    @property
    def u(self):
        """Dollar risk unit per path, see risk_unit()."""
        return risk_unit(self.Q, self.S[:, 0], self.sigma_imp)

    @property
    def cost_bps(self):
        return self.spread_bps + self.fee / self.S[:, 0] * 1e4


def risk_unit(Q, S0, sigma_imp):
    """One unit of P&L: G0 times the 1-sigma one day $ move of the whole book (Q shares), independent of hedge frequency.

    Risk aversion is dimensionless: kappa_tilde = kappa * u, with kappa the usual exponential utility parameter in 1/$.
    """
    return G0 * Q * S0 * sigma_imp / np.sqrt(DAYS)


def _col(x):
    """Scalars stay scalars, per path arrays become (n, 1) columns."""
    x = np.asarray(x, float)
    return x.reshape(-1, 1) if x.ndim else float(x)


def simulate_prices(mkt: Market, hz: Horizon, n: int, seed: int = 0, antithetic: bool = False, rng=None):
    """Real world price paths: GBM with optional Merton jumps and per path vol dispersion.

    Market fields may be arrays of length n (used when training on random scenarios), then antithetic must be False.
    Antithetic pairs are off by default: hedging P&L is mostly an even function of the shocks (gamma), so mirrored paths
    are positively correlated and would make the Monte Carlo error larger, not smaller.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    T = hz.horizon_days * hz.steps_per_day
    dt = 1.0 / (DAYS * hz.steps_per_day)
    half = n // 2 if antithetic else n
    Z = rng.standard_normal((half, T))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
        if n % 2:
            Z = np.concatenate([Z, rng.standard_normal((1, T))], axis=0)
    xi = rng.standard_normal((Z.shape[0], 1))
    d = _col(mkt.vol_disp)
    sig = _col(mkt.sigma_real) * np.exp(d * xi - 0.5 * d * d)
    r = _col(mkt.r)
    mu = r if mkt.mu is None else _col(mkt.mu)
    lam, jm, js = _col(mkt.jump_lambda), _col(mkt.jump_mean), _col(mkt.jump_sd)
    kbar = np.exp(jm + 0.5 * js ** 2) - 1.0
    ret = (mu - 0.5 * sig ** 2 - lam * kbar) * dt + sig * np.sqrt(dt) * Z
    if np.any(np.asarray(mkt.jump_lambda) > 0.0):
        cnt = rng.poisson(lam * dt, size=Z.shape)
        ret = ret + cnt * jm + np.sqrt(cnt) * js * rng.standard_normal(Z.shape)
    S = _col(mkt.S0) * np.exp(np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(ret, axis=1)], axis=1))
    return S, dt, T


def build_paths(mkt: Market, costs: Costs, hz: Horizon, book: Book, n: int, seed: int = 0, kappa: float = 1.0,
                antithetic: bool = False) -> Paths:
    S, dt, T = simulate_prices(mkt, hz, n, seed, antithetic)
    n = S.shape[0]
    tau = np.maximum(hz.expiry_days / DAYS - np.arange(T + 1) * dt, 0.0)
    V, D, G = book_state(S, book, tau, mkt.r, mkt.sigma_imp)
    one = np.ones(n)
    return Paths(S=S, V=V, D=D, G=G, tau=tau, Q=one * book.gross if np.ndim(book.gross) == 0 else book.gross,
                 sigma_imp=one * mkt.sigma_imp, r=one * mkt.r, spread_bps=one * costs.spread_bps, fee=one * costs.fee,
                 impact_bps=one * costs.impact_bps, kappa=one * kappa, dt=dt, T=T, spd=hz.steps_per_day)


def trade_cost(d, S, Q, spread_bps, fee, impact_bps):
    ad = np.abs(d)
    return ad * (S * spread_bps / 1e4 + fee) + S * (impact_bps / 1e4) * d * d / Q


# ------------------------------------------------------------------------------------------------ running a strategy
@dataclass
class Ctx:
    """What a strategy sees before it trades at step t. All arrays are over paths."""
    t: int
    S: np.ndarray
    D: np.ndarray               # book delta, shares
    G: np.ndarray               # book gamma
    h: np.ndarray               # stock hedge held, shares
    net: np.ndarray             # D + h, the delta the desk is actually running
    P: Paths

    @property
    def tau(self):
        tau = self.P.tau
        return tau[self.t] if tau.ndim == 1 else tau[:, self.t]

    @property
    def steps_left(self):
        return self.P.T - self.t


@dataclass
class Result:
    name: str
    pnl: np.ndarray             # (n,) final P&L after unwinding
    cum: np.ndarray             # (n, T+1) P&L through time, last column is the final P&L
    h: np.ndarray               # (n, T+1) stock hedge after trading at each step (last column: after unwind = 0)
    net: np.ndarray             # (n, T+1) net delta after trading (last column: pre unwind)
    trades: np.ndarray          # (n, T) shares traded
    cost_cum: np.ndarray        # (n, T+1)
    parts: dict                 # book, hedge, financing, costs: each (n,)
    turnover: np.ndarray        # (n,) shares traded / Q
    n_trades: np.ndarray        # (n,)
    avg_abs_net: np.ndarray     # (n,) mean |net delta| / Q


TRADE_EPS = 1e-9


def run_strategy(name, decide, P: Paths) -> Result:
    """Step a hedging rule through every path. decide(ctx) returns the shares to trade (positive = buy)."""
    n, T = P.n, P.T
    h = np.zeros(n)
    B = -P.V[:, 0].copy()
    cum = np.zeros((n, T + 1))
    cost_cum = np.zeros((n, T + 1))
    h_hist = np.zeros((n, T + 1))
    net_hist = np.zeros((n, T + 1))
    trades = np.zeros((n, T))
    book_pnl = P.V[:, -1] - P.V[:, 0]
    hedge_pnl = np.zeros(n)
    fin = np.zeros(n)
    costs = np.zeros(n)
    growth = np.exp(P.r * P.dt)
    pi_prev = np.zeros(n)
    for t in range(T):
        S, S1 = P.S[:, t], P.S[:, t + 1]
        ctx = Ctx(t, S, P.D[:, t], P.G[:, t], h, P.D[:, t] + h, P)
        d = np.asarray(decide(ctx), float)
        d = np.where(np.abs(d) < TRADE_EPS, 0.0, d)
        c = trade_cost(d, S, P.Q, P.spread_bps, P.fee, P.impact_bps)
        h = h + d
        B = B - d * S - c
        B_new = B * growth
        fin += B_new - B
        B = B_new
        hedge_pnl += h * (S1 - S)
        costs += c
        trades[:, t] = d
        h_hist[:, t] = h
        net_hist[:, t] = P.D[:, t] + h
        cost_cum[:, t + 1] = costs
        pi = P.V[:, t + 1] + B + h * S1
        cum[:, t + 1] = pi
        pi_prev = pi
    # unwind what is left
    net_hist[:, T] = P.D[:, T] + h
    c_end = trade_cost(-h, P.S[:, T], P.Q, P.spread_bps, P.fee, P.impact_bps)
    costs += c_end
    cum[:, T] -= c_end
    cost_cum[:, T] = costs
    h_hist[:, T] = 0.0
    pnl = cum[:, T].copy()
    ntr = (np.abs(trades) > TRADE_EPS).sum(axis=1)
    return Result(name=name, pnl=pnl, cum=cum, h=h_hist, net=net_hist, trades=trades, cost_cum=cost_cum,
                  parts={"Option book": book_pnl, "Stock hedge": hedge_pnl, "Cash carry": fin, "Costs": -costs},
                  turnover=np.abs(trades).sum(axis=1) / P.Q, n_trades=ntr,
                  avg_abs_net=np.abs(net_hist[:, :T]).mean(axis=1) / P.Q)


# ------------------------------------------------------------------------------------------------ strategies
def unhedged(ctx):
    return np.zeros_like(ctx.S)


def periodic(every=1):
    """Flatten the net delta every `every` steps."""
    def f(ctx):
        return -ctx.net if ctx.t % every == 0 else np.zeros_like(ctx.S)
    return f


def threshold(band_frac):
    """Flatten the net delta when |net| exceeds band_frac * Q. Hedges all the way back to zero."""
    def f(ctx):
        hit = np.abs(ctx.net) > band_frac * ctx.P.Q
        return np.where(hit, -ctx.net, 0.0)
    return f


def ww_band(S, G, tau, r, cost_bps, kappa, u):
    """Whalley and Wilmott (1997) half width of the no trade band, in shares: (3/2 e^{-r tau} k S Gamma^2 / gamma)^(1/3).

    k is the proportional cost rate (half spread plus fee as a fraction of S), Gamma the book gamma, gamma = kappa / u the
    desk's risk aversion in 1/$.
    """
    inner = 1.5 * np.exp(-r * tau) * (cost_bps / 1e4) * S * G ** 2 / (kappa / u)
    return np.cbrt(np.maximum(inner, 0.0))


def ww_halfwidth(ctx, kappa):
    P = ctx.P
    return ww_band(ctx.S, ctx.G, ctx.tau, P.r, P.cost_bps, kappa, P.u)


def whalley_wilmott(kappa):
    """Trade only when the net delta leaves the band, and then only back to the band edge."""
    def f(ctx):
        H = ww_halfwidth(ctx, kappa)
        over = ctx.net - H
        under = ctx.net + H
        return np.where(ctx.net > H, -over, np.where(ctx.net < -H, -under, 0.0))
    return f


# ------------------------------------------------------------------------------------------------ metrics
def es(x, q=0.05):
    """Expected shortfall: mean of the worst q fraction of outcomes."""
    k = max(1, int(np.floor(q * len(x))))
    return float(np.sort(x)[:k].mean())


def summarize(res: Result, P: Paths, kappa: float | None = None) -> dict:
    p = res.pnl
    u = float(P.u[0])
    kap = float(P.kappa[0]) if kappa is None else kappa
    mean, sd = float(p.mean()), float(p.std(ddof=1))
    return {
        "mean": mean, "std": sd, "es5": es(p), "p5": float(np.percentile(p, 5)),
        "cost": float(np.abs(res.parts["Costs"]).mean()),
        "turnover": float(res.turnover.mean()),
        "trades": float(res.n_trades.mean()),
        "abs_net": float(res.avg_abs_net.mean()),
        # mean minus half kappa variance, the objective the agent is trained on, in dollars
        "objective": mean - 0.5 * (kap / u) * sd ** 2,
        "se_mean": sd / np.sqrt(len(p)),
    }


def objective_diff(pnl_a, pnl_b, kappa_over_u, n_boot=400, seed=0):
    """Paired bootstrap of J(a) - J(b), J = mean - 0.5 * (kappa / u) * var. Returns (difference, standard error)."""
    pnl_a, pnl_b = np.asarray(pnl_a), np.asarray(pnl_b)
    n = len(pnl_a)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))

    def J(x):
        return x.mean(axis=-1) - 0.5 * kappa_over_u * x.var(axis=-1, ddof=1)

    return float(J(pnl_a) - J(pnl_b)), float((J(pnl_a[idx]) - J(pnl_b[idx])).std())
