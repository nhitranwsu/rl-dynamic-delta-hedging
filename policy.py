"""The learned hedging policy: state features and a small MLP evaluated in numpy.

Training lives in train.py (torch). The app only needs numpy, so the weights are exported to policy.npz.

State the agent sees before each trade, all scale free so one network covers different books, vols and costs:
    0  net delta / (5% of book size)
    1  book gamma times the one step 1-sigma spot move, same units (how far the delta will drift next step)
    2  trading cost in bps / 10          (half spread plus fee)
    3  impact bps / 10
    4  log10 of risk aversion kappa
    5  days of hedging left / 60
    6  days to expiry / 60
    7  log2 of the step length in days
    8  book gamma times spot / book size
Action: the share f in [0, 1] of the net delta to trade away, trade = -f * net. f = 0 is "wait".
"""
import math

import numpy as np

N_FEATURES = 9
HIDDEN = 128
DEADZONE = 0.03     # f below this is treated as "do not trade" when the policy is run


def features(xp, net, G, S, sigma, Q, dt, cost_bps, impact_bps, kappa, days_left, exp_days_left):
    """Works on numpy arrays and torch tensors alike (xp is the module). dt is in years."""
    ref = 0.05 * Q
    z = 0.0 * net                                   # broadcasts scalars to one value per path
    step_move = G * S * sigma * (dt ** 0.5)
    cols = [
        xp.clip(net / ref, -10.0, 10.0),
        xp.clip(step_move / ref, -10.0, 10.0),
        cost_bps / 10.0 + z,
        impact_bps / 10.0 + z,
        xp.log10(kappa) + z,
        days_left / 60.0 + z,
        exp_days_left / 60.0 + z,
        math.log2(dt * 252.0) + z,
        xp.clip(G * S / Q, -10.0, 10.0),
    ]
    return xp.stack(cols, -1)


class Policy:
    def __init__(self, W1, b1, W2, b2, W3, b3, meta=None):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = W1, b1, W2, b2, W3, b3
        self.meta = meta or {}

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        return cls(*(z[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")), meta=meta)

    def fraction(self, x):
        """x: (n, N_FEATURES) -> f in (0, 1), shape (n,)."""
        h = np.maximum(x @ self.W1.T + self.b1, 0.0)
        h = np.maximum(h @ self.W2.T + self.b2, 0.0)
        z = (h @ self.W3.T + self.b3)[:, 0]
        return 1.0 / (1.0 + np.exp(-z))


def agent(policy: Policy, deadzone: float = DEADZONE):
    """Hedging rule for hedging.run_strategy: trade a learned share of the net delta."""
    def decide(ctx):
        P = ctx.P
        x = features(np, ctx.net, ctx.G, ctx.S, P.sigma_imp, P.Q, P.dt, P.cost_bps, P.impact_bps, P.kappa,
                     ctx.steps_left * P.dt * 252.0, ctx.tau * 252.0)
        f = policy.fraction(x)
        f = np.where(f < deadzone, 0.0, f)
        return -f * ctx.net
    return decide
