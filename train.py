"""Train the hedging agent (offline, needs torch). Writes policy.npz and training_log.json.

Method: the agent is a policy network pi(state) -> share of the net delta to trade. It is trained on simulated experience with
the per step reward of Kolm and Ritter (2019),

    r_t = dPi_t / u  -  (kappa / 2) * (dPi_t / u)^2,

where dPi_t is the change in the desk's total P&L over step t, transaction costs included, and u is the book's risk unit.
The sum of r_t is a mean variance objective, the same thing a Whalley-Wilmott band maximises. Because the simulator is
differentiable in the hedge, the policy is improved with pathwise gradients (deep hedging, Buehler et al. 2019), which has
far lower variance than value based methods such as DQN on this problem: the price noise does not depend on the action, so
Q learning has to average it away, while the pathwise gradient never sees it.

Every batch draws random books, vols, costs, jumps, hedge frequencies, horizons and risk aversion, so one network covers them.

    python train.py            (about 20 to 40 minutes on 2 CPU cores)
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import hedging as H
from policy import N_FEATURES, HIDDEN, features, Policy, agent

torch.set_num_threads(2)
SLOTS = 5


# ----------------------------------------------------------------------------------------- random training scenarios
def _pad(book: H.Book):
    k = len(book.K)
    K = np.full(SLOTS, 100.0)
    q = np.zeros(SLOTS)
    c = np.zeros(SLOTS, bool)
    K[:k], q[:k], c[:k] = book.K, book.qty, book.is_call
    return K, q, c


def random_books(rng, B):
    Ks, qs, cs = [], [], []
    for _ in range(B):
        contracts = int(rng.integers(10, 100))
        if rng.random() < 0.5:
            bk = H.random_book(100.0, contracts, 100, rng, SLOTS)
        else:
            bk = H.make_book(str(rng.choice(H.PRESETS[:4])), 100.0, contracts, 100)
        K, q, c = _pad(bk)
        Ks.append(K), qs.append(q), cs.append(c)
    return H.Book(np.array(Ks), np.array(qs), np.array(cs))


def random_batch(rng, B):
    spd = int(rng.choice([1, 1, 2]))
    days = int(rng.integers(5, 46))
    T = days * spd
    dt = 1.0 / (H.DAYS * spd)
    sig = rng.uniform(0.12, 0.60, B)
    ratio = rng.uniform(0.85, 1.25, B)
    jumps = rng.random(B) < 0.35
    mkt = H.Market(
        S0=100.0, sigma_imp=sig, sigma_real=sig * ratio, r=rng.uniform(0.0, 0.05, B),
        jump_lambda=np.where(jumps, rng.uniform(2.0, 25.0, B), 0.0),
        jump_mean=rng.uniform(-0.05, 0.01, B), jump_sd=rng.uniform(0.02, 0.06, B), vol_disp=rng.uniform(0.0, 0.3, B))
    hz = H.Horizon(days, days, spd)
    S, _, _ = H.simulate_prices(mkt, hz, B, antithetic=False, rng=rng)
    extra = np.where(rng.random(B) < 0.5, 0, rng.integers(0, 61, B))
    expiry = (days + extra) / H.DAYS
    tau = np.maximum(expiry[:, None] - np.arange(T + 1)[None, :] * dt, 0.0)
    book = random_books(rng, B)
    V, D, G = H.book_state(S, book, tau, mkt.r, sig)
    P = H.Paths(S=S, V=V, D=D, G=G, tau=tau, Q=book.gross, sigma_imp=sig, r=np.asarray(mkt.r),
                spread_bps=rng.uniform(0.0, 12.0, B), fee=rng.uniform(0.0, 0.01, B),
                impact_bps=np.where(rng.random(B) < 0.5, 0.0, rng.uniform(0.0, 25.0, B)),
                kappa=np.exp(rng.uniform(np.log(0.02), np.log(2.0), B)), dt=dt, T=T, spd=spd)
    return P


# ----------------------------------------------------------------------------------------- network and rollout
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1, self.f2, self.f3 = nn.Linear(N_FEATURES, HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.f3.weight)
        nn.init.constant_(self.f3.bias, 1.0)       # start close to "hedge most of the delta"

    def forward(self, x):
        return self.f3(torch.relu(self.f2(torch.relu(self.f1(x))))).squeeze(-1)


def rollout(net, P: H.Paths):
    """Differentiable simulation of the desk trading with the policy. Returns the mean per path return and diagnostics."""
    td = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    S, V, D, G = td(P.S), td(P.V), td(P.D), td(P.G)
    n, T = P.n, P.T
    Q, sig, r = td(P.Q), td(P.sigma_imp), td(P.r)
    sp, fee, imp, kap = td(P.spread_bps), td(P.fee), td(P.impact_bps), td(P.kappa)
    tau = td(P.tau)
    u = td(P.u)
    cost_bps = sp + fee / S[:, 0] * 1e4
    growth = torch.exp(r * P.dt)
    h = torch.zeros(n, dtype=torch.float64)
    B = -V[:, 0]
    pi_prev = torch.zeros(n, dtype=torch.float64)
    total = torch.zeros(n, dtype=torch.float64)
    cost_tot = torch.zeros(n, dtype=torch.float64)
    turn = torch.zeros(n, dtype=torch.float64)
    tcost = lambda d, s: d.abs() * (s * sp / 1e4 + fee) + s * (imp / 1e4) * d * d / Q
    for t in range(T):
        s, s1 = S[:, t], S[:, t + 1]
        net_d = D[:, t] + h
        tau_t = tau[t] if tau.dim() == 1 else tau[:, t]
        x = features(torch, net_d, G[:, t], s, sig, Q, P.dt, cost_bps, imp, kap,
                     (T - t) * P.dt * 252.0, tau_t * 252.0)
        f = torch.sigmoid(net(x.float())).double()
        d = -f * net_d
        c = tcost(d, s)
        h = h + d
        B = (B - d * s - c) * growth
        pi = V[:, t + 1] + B + h * s1
        dpi = pi - pi_prev
        if t == T - 1:
            c_end = tcost(-h, s1)
            dpi = dpi - c_end
            cost_tot = cost_tot + c_end
        pi_prev = pi
        rr = dpi / u
        total = total + rr - 0.5 * kap * rr * rr
        cost_tot = cost_tot + c
        turn = turn + d.abs() / Q
    return total.mean(), {"cost": (cost_tot / u).mean().item(), "turn": turn.mean().item()}


def export(net, path, meta):
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    np.savez(path, W1=sd["f1.weight"], b1=sd["f1.bias"], W2=sd["f2.weight"], b2=sd["f2.bias"],
             W3=sd["f3.weight"], b3=sd["f3.bias"], **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


# ----------------------------------------------------------------------------------------- validation on fixed scenarios
def val_scenarios():
    S0 = 100.0
    out = []
    spec = [
        ("Short straddle, 30d", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 0), H.Horizon(30, 30, 1), 0.1),
        ("Short straddle, high risk aversion", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 0), H.Horizon(30, 30, 1), 1.0),
        ("Random book, wide costs", "Customer flow (random book)", H.Market(sigma_imp=0.35, sigma_real=0.35), H.Costs(8, 0.0, 10), H.Horizon(20, 20, 1), 0.1),
        ("Strangle with jumps", "Short OTM strangle", H.Market(jump_lambda=10.0, jump_mean=-0.03, jump_sd=0.04), H.Costs(4, 0.0, 5), H.Horizon(30, 30, 1), 0.2),
        ("Long straddle, 2 steps per day", "Long ATM straddle", H.Market(sigma_imp=0.30, sigma_real=0.33), H.Costs(3, 0.0, 5), H.Horizon(20, 45, 2), 0.1),
        ("Risk reversal, quiet vol", "Risk reversal (short put, long call)", H.Market(sigma_imp=0.18, sigma_real=0.16), H.Costs(5, 0.0, 0), H.Horizon(40, 40, 1), 0.3),
    ]
    for name, bk, mkt, cs, hz, kap in spec:
        book = H.make_book(bk, S0, 50, 100, seed=7)
        P = H.build_paths(mkt, cs, hz, book, 2000, seed=123, kappa=kap)
        out.append((name, P))
    return out


def evaluate(policy, scen):
    rows = []
    for name, P in scen:
        kap = float(P.kappa[0])
        J = {}
        for lab, f in (("agent", agent(policy)), ("ww", H.whalley_wilmott(kap)), ("per", H.periodic(1))):
            s = H.summarize(H.run_strategy(lab, f, P), P)
            J[lab] = s["objective"] / float(P.u[0])
        rows.append((name, J))
    return rows


def policy_from(net):
    return Policy(*[a.detach().double().numpy() for a in
                    (net.f1.weight, net.f1.bias, net.f2.weight, net.f2.bias, net.f3.weight, net.f3.bias)])


def main(iters=3000, batch=1024, seed=0, lr0=2e-3, lr1=1e-4, init=None, out="policy", ema_decay=0.0):
    """ema_decay > 0 keeps a moving average of the weights and exports that, which removes most of the checkpoint noise."""
    rng = np.random.default_rng(seed)
    net = Net()
    if init:
        z = np.load(init)
        for layer, (w, b) in ((net.f1, ("W1", "b1")), (net.f2, ("W2", "b2")), (net.f3, ("W3", "b3"))):
            layer.weight.data = torch.as_tensor(z[w], dtype=torch.float32)
            layer.bias.data = torch.as_tensor(z[b], dtype=torch.float32)
    avg = Net()
    avg.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr1)
    scen = val_scenarios()
    log = {"iter": [], "train": [], "val": [], "val_ww": [], "val_per": [], "scenarios": [n for n, _ in scen], "detail": []}
    best, t0, ema, va = -1e9, time.time(), None, None
    for it in range(1, iters + 1):
        P = random_batch(rng, batch)
        obj, info = rollout(net, P)
        opt.zero_grad()
        (-obj).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if ema_decay > 0:
            with torch.no_grad():
                for pa, pn in zip(avg.parameters(), net.parameters()):
                    pa.mul_(ema_decay).add_(pn, alpha=1.0 - ema_decay)
        ema = obj.item() if ema is None else 0.98 * ema + 0.02 * obj.item()
        if it % 100 == 0 or it == 1:
            held = avg if ema_decay > 0 else net
            rows = evaluate(policy_from(held), scen)
            va = float(np.mean([j["agent"] for _, j in rows]))
            vw = float(np.mean([j["ww"] for _, j in rows]))
            vp = float(np.mean([j["per"] for _, j in rows]))
            log["iter"].append(it), log["train"].append(ema), log["val"].append(va)
            log["val_ww"].append(vw), log["val_per"].append(vp)
            log["detail"].append({n: j for n, j in rows})
            print(f"it {it:5d} | train {ema:8.3f} | val agent {va:8.3f}  WW {vw:8.3f}  periodic {vp:8.3f} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
            if va > best:
                best = va
                export(held, f"{out}.npz", {"val": va, "iters": it})
            json.dump(log, open(f"{out}_log.json", "w"))
    export(avg if ema_decay > 0 else net, f"{out}_last.npz", {"val": va, "iters": iters})
    print("done, best val", best)


if __name__ == "__main__":
    import os
    main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 3000,
         batch=int(os.environ.get("BATCH", 1024)), lr0=float(os.environ.get("LR0", 2e-3)),
         lr1=float(os.environ.get("LR1", 1e-4)), init=os.environ.get("INIT"), out=os.environ.get("OUT", "policy"),
         ema_decay=float(os.environ.get("EMA", 0.0)))
