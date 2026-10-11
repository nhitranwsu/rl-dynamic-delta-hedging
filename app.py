"""Delta hedging desk: a market maker's options book hedged by a learned policy and by classic desk rules."""
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import hedging as H
import ui
from black_scholes import bs_np
from policy import Policy, agent, features

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

st.set_page_config(page_title="Delta hedging desk", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.css(), unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------- state and defaults
DEFAULTS = dict(
    preset="Short ATM straddle", book_seed=3, S0=100.0, sig_i=25, prem=0, rate=3.0,
    jump_l=0, jump_m=-2.0, jump_s=4.0, disp=0.0,
    days=30, extra=0, spd=1, spread=3.0, fee=0.005, impact=5.0, kappa=0.1,
    per_k=1, band=3.0, n_paths=4000, seed=1, view="Monte Carlo",
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def table_from_book(book: H.Book, S0: float) -> pd.DataFrame:
    rows = [{"Type": "Call" if c else "Put", "Strike %": round(100.0 * k / S0, 1), "Contracts": int(round(q / 100.0))}
            for k, q, c in zip(book.K, book.qty, book.is_call) if abs(q) > 0]
    return pd.DataFrame(rows)


def reset_book():
    """New preset or new seed: rebuild the position table and drop the old editor state."""
    bk = H.make_book(st.session_state.preset, 100.0, 50, 100, seed=int(st.session_state.book_seed))
    st.session_state.book_df = table_from_book(bk, 100.0)
    st.session_state.editor_gen = st.session_state.get("editor_gen", 0) + 1


if "book_df" not in st.session_state:
    reset_book()


@st.cache_resource(show_spinner=False)
def load_policy():
    path = os.path.join(HERE, "policy.npz")
    return Policy.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_log():
    path = os.path.join(HERE, "training_log.json")
    return json.load(open(path)) if os.path.exists(path) else None


policy = load_policy()

# ----------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="tm pt">Book</div>', unsafe_allow_html=True)
    st.selectbox("Preset", H.PRESETS, key="preset", on_change=reset_book, label_visibility="collapsed")
    if st.session_state.preset.startswith("Customer"):
        st.number_input("Book seed", min_value=0, step=1, key="book_seed", on_change=reset_book)
    st.caption("Contracts of 100 shares. Negative means the desk is short. Up to 5 positions.")
    book_df = st.data_editor(
        st.session_state.book_df, num_rows="dynamic", hide_index=True, width="stretch",
        key=f"book_ed_{st.session_state.get('editor_gen', 0)}",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Call", "Put"], required=True, width=64),
            "Strike %": st.column_config.NumberColumn("K %", help="Strike as a percent of spot", min_value=80.0,
                                                      max_value=120.0, step=1.0, format="%.0f", required=True, width=56),
            "Contracts": st.column_config.NumberColumn("Qty", help="Contracts, negative means the desk is short",
                                                       min_value=-100, max_value=100, step=1, format="%d",
                                                       required=True, width=56),
        })

    st.markdown('<div class="tm pt sp">Market</div>', unsafe_allow_html=True)
    st.number_input("Spot S", min_value=5.0, max_value=2000.0, step=1.0, format="%.2f", key="S0")
    c1, c2 = st.columns(2)
    c1.number_input("Marked vol %", min_value=12, max_value=60, step=1, key="sig_i")
    c2.number_input("Realised vs marked %", min_value=-15, max_value=25, step=1, key="prem",
                    help="Realised vol minus marked vol, as a percent of marked vol. Zero means the desk marks fairly.")
    st.slider("Rate %", 0.0, 5.0, step=0.25, key="rate")
    st.slider("Jumps per year", 0, 25, key="jump_l")
    if st.session_state.jump_l > 0:
        c1, c2 = st.columns(2)
        c1.number_input("Jump mean %", min_value=-5.0, max_value=1.0, step=0.5, key="jump_m")
        c2.number_input("Jump std %", min_value=2.0, max_value=6.0, step=0.5, key="jump_s")
    st.slider("Vol dispersion across paths", 0.0, 0.3, step=0.05, key="disp",
              help="Realised vol is lognormal across paths with this dispersion. Zero means every path has the same vol.")

    st.markdown('<div class="tm pt sp">Hedging</div>', unsafe_allow_html=True)
    st.slider("Hedge for (trading days)", 5, 45, key="days")
    st.slider("Expiry beyond the hedge (days)", 0, 60, key="extra",
              help="If above zero the book is still alive when the hedge stops and is marked at that time.")
    st.radio("Hedges per day", [1, 2], key="spd", horizontal=True)

    st.markdown('<div class="tm pt sp">Costs</div>', unsafe_allow_html=True)
    st.slider("Half spread (bps)", 0.0, 12.0, step=0.5, key="spread")
    c1, c2 = st.columns(2)
    c1.number_input("Fee $/share", min_value=0.0, max_value=0.01, step=0.001, format="%.3f", key="fee")
    c2.number_input("Impact bps", min_value=0.0, max_value=25.0, step=1.0, format="%.0f", key="impact",
                    help="Extra cost, in bps of notional, for a single trade as large as the whole book. Quadratic in size.")

    st.markdown('<div class="tm pt sp">Objective and rules</div>', unsafe_allow_html=True)
    st.select_slider("Risk aversion κ", KAPPAS, key="kappa",
                     help="Higher means the desk pays more cost to keep hedging error down. The agent and the "
                          "Whalley-Wilmott band both use it. Defined as κ = κ̃ / u, u = 20% of the book's one day 1σ move.")
    st.slider("Periodic: hedge every k steps", 1, 10, key="per_k")
    st.slider("Threshold: band, % of book", 0.5, 20.0, step=0.5, key="band")

    st.markdown('<div class="tm pt sp">Simulation</div>', unsafe_allow_html=True)
    st.select_slider("Paths", [500, 1000, 2000, 4000, 8000], key="n_paths")
    st.number_input("Seed", min_value=0, step=1, key="seed")

# ----------------------------------------------------------------------------------------------- build the inputs
ss = st.session_state
S0 = float(ss.S0)
rows = book_df.dropna()
rows = rows[rows["Contracts"] != 0]
if len(rows) == 0:
    st.markdown(ui.brand_html(), unsafe_allow_html=True)
    st.warning("The book is empty. Add a position in the sidebar.")
    st.stop()
if len(rows) > 5:
    st.sidebar.warning("The agent was trained on books of up to 5 positions. Using the first 5.")
    rows = rows.iloc[:5]
BOOK = (tuple(rows["Type"] == "Call"), tuple(S0 * rows["Strike %"] / 100.0), tuple(100.0 * rows["Contracts"]))

CFG = dict(
    book=BOOK, S0=S0, sig_i=ss.sig_i / 100.0, sig_r=ss.sig_i / 100.0 * (1.0 + ss.prem / 100.0), r=ss.rate / 100.0,
    jump_l=float(ss.jump_l), jump_m=ss.jump_m / 100.0, jump_s=ss.jump_s / 100.0, disp=float(ss.disp),
    days=int(ss.days), extra=int(ss.extra), spd=int(ss.spd), spread=float(ss.spread), fee=float(ss.fee),
    impact=float(ss.impact), kappa=float(ss.kappa), per_k=int(ss.per_k), band=float(ss.band) / 100.0,
    n=int(ss.n_paths), seed=int(ss.seed))
BASE = tuple(sorted((k, v) for k, v in CFG.items() if k not in ("per_k", "band", "kappa")))
PER_GRID = (1, 2, 3, 4, 5)
THR_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)


@st.cache_resource(show_spinner=False, max_entries=6)
def core(base):
    """Price paths and every strategy that does not depend on the periodic or threshold sliders.

    The risk aversion only changes the Whalley-Wilmott band and the agent's input, so both are run for every value of the
    slider here and a change of kappa is just a lookup.
    """
    c = dict(base)
    is_call, K, qty = c["book"]
    book = H.Book(np.array(K), np.array(qty), np.array(is_call))
    mkt = H.Market(S0=c["S0"], sigma_imp=c["sig_i"], sigma_real=c["sig_r"], r=c["r"], jump_lambda=c["jump_l"],
                   jump_mean=c["jump_m"], jump_sd=c["jump_s"], vol_disp=c["disp"])
    costs = H.Costs(c["spread"], c["fee"], c["impact"])
    hz = H.Horizon(c["days"], c["days"] + c["extra"], c["spd"])
    P = H.build_paths(mkt, costs, hz, book, c["n"], seed=c["seed"], kappa=1.0)
    run = lambda rule, Pk=P: H.run_strategy("x", rule, Pk)
    return dict(
        P=P, book=book, unh=run(H.unhedged),
        per={k: run(H.periodic(k)) for k in PER_GRID if k <= P.T},
        thr={b_: run(H.threshold(b_)) for b_ in THR_GRID},
        ww={k: run(H.whalley_wilmott(k)) for k in KAPPAS},
        agent={k: run(agent(policy), replace(P, kappa=np.full(P.n, k))) for k in KAPPAS} if policy is not None else {},
    )


with st.spinner("Simulating"):
    CORE = core(BASE)
P = CORE["P"]
KAP = CFG["kappa"]
U = float(H.risk_unit(P.Q[0], P.S[0, 0], CFG["sig_i"]))
summ = lambda r: H.summarize(r, P, kappa=KAP)

RES = {ui.UNH: CORE["unh"], ui.PER: H.run_strategy("p", H.periodic(CFG["per_k"]), P),
       ui.THR: H.run_strategy("t", H.threshold(CFG["band"]), P), ui.WW: CORE["ww"][KAP]}
if CORE["agent"]:
    RES[ui.AGENT] = CORE["agent"][KAP]
SUMS = {n: summ(r) for n, r in RES.items()}

# the best rule after tuning on these same paths, which favours the rules, so beating it is a conservative test
cand = [(f"Periodic, every {k}", r) for k, r in CORE["per"].items()]
cand += [(f"Threshold ±{b_ * 100:.1f}%", r) for b_, r in CORE["thr"].items()]
cand += [("Whalley-Wilmott", RES[ui.WW]), ("Periodic (your setting)", RES[ui.PER]), ("Threshold (your setting)", RES[ui.THR])]
_scored = [(lab, r, summ(r)) for lab, r in cand]
_best = max(_scored, key=lambda t: t[2]["objective"])
TUNED = dict(label=_best[0], res=_best[1], summary=_best[2])
if ui.AGENT in RES:
    TUNED["diff"], TUNED["se"] = H.objective_diff(RES[ui.AGENT].pnl, _best[1].pnl, KAP / U)
SUMS[ui.TUNED] = TUNED["summary"]

# parameter sweeps for the cost against risk frontier
CURVES = {
    ui.PER: [(lambda s_: (s_["cost"], s_["std"]))(summ(r)) for r in CORE["per"].values()],
    ui.THR: [(lambda s_: (s_["cost"], s_["std"]))(summ(r)) for r in CORE["thr"].values()],
    ui.WW: [(lambda s_: (s_["cost"], s_["std"]))(H.summarize(r, P, kappa=k)) for k, r in CORE["ww"].items()],
}
if CORE["agent"]:
    CURVES[ui.AGENT] = [(lambda s_: (s_["cost"], s_["std"]))(H.summarize(r, P, kappa=k)) for k, r in CORE["agent"].items()]

LABEL = {ui.AGENT: "Agent (deep hedging)", ui.WW: "Whalley-Wilmott band",
         ui.PER: f"Periodic, every {CFG['per_k']} step{'s' if CFG['per_k'] > 1 else ''}",
         ui.THR: f"Threshold, ±{CFG['band'] * 100:.1f}% of book", ui.UNH: "Unhedged",
         ui.TUNED: f"Best tuned rule: {TUNED['label']}"}
NAMES = [n for n in ui.ORDER if n in RES]
RULES = [n for n in NAMES if n != ui.UNH]
BEST = max(RULES, key=lambda n: SUMS[n]["objective"])

# ----------------------------------------------------------------------------------------------- header
c_brand, c_view = st.columns([4.4, 3.0], vertical_alignment="center")
c_brand.markdown(ui.brand_html(), unsafe_allow_html=True)
c_view.radio("View", ["Monte Carlo", "Path replay", "Agent", "Model"], key="view", horizontal=True,
             label_visibility="collapsed")

BOOK_NAME = ss.preset if book_df.reset_index(drop=True).astype(str).equals(
    ss.book_df.reset_index(drop=True).astype(str)) else "Custom"
st.markdown(ui.meta_html(dict(
    book=BOOK_NAME,
    Q=float(P.Q[0]), delta=float(P.D[0, 0]), gamma=float(P.G[0, 0]), sig_i=CFG["sig_i"] * 100,
    sig_r=CFG["sig_r"] * 100, days=CFG["days"], spd=CFG["spd"], spread=CFG["spread"], impact=CFG["impact"],
    n=P.n)), unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- views
def monte_carlo():
    a = SUMS.get(ui.AGENT)
    if a is not None:
        d, se = TUNED["diff"], TUNED["se"]
        verdict = "within noise" if abs(d) < 2 * se else ("agent ahead" if d > 0 else "agent behind")
        t1 = ("Agent objective vs best tuned rule", ui.usd(d, dec=0),
              f'± ${se:,.0f} (paired, 1 s.e.), {verdict}. Rule: {TUNED["label"]}')
        per = SUMS[ui.PER]
        t2 = ("Hedging error (std of P&L)", f'${a["std"]:,.0f}', f'periodic ${per["std"]:,.0f}, unhedged ${SUMS[ui.UNH]["std"]:,.0f}')
        sav = 100.0 * (1.0 - a["cost"] / per["cost"]) if per["cost"] > 0 else 0.0
        t3 = ("Transaction cost", f'${a["cost"]:,.0f}', f'{ui._n(-sav, ".0f")}% against periodic (${per["cost"]:,.0f})')
        t4 = ("Turnover", f'{a["turnover"]:.2f}×', f'{a["trades"]:.1f} trades per path, book size {P.Q[0]:,.0f} sh')
    else:
        b = SUMS[BEST]
        t1 = ("Best objective", ui.usd(b["objective"], dec=0), LABEL[BEST])
        t2 = ("Hedging error (std of P&L)", f'${b["std"]:,.0f}', "")
        t3 = ("Transaction cost", f'${b["cost"]:,.0f}', "")
        t4 = ("Turnover", f'{b["turnover"]:.2f}×', "")
    st.markdown(ui.kpi_html([t1, t2, t3, t4]), unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(ui.strat_legend([n for n in NAMES if n != ui.UNH]), unsafe_allow_html=True)
        st.plotly_chart(ui.dist_figure({n: RES[n].pnl for n in NAMES}), theme=None, width="stretch",
                        config={"displayModeBar": False})
    with right:
        pts = {n: (SUMS[n]["cost"], SUMS[n]["std"]) for n in RULES if n != ui.TUNED}
        st.markdown(ui.legend_html([("line", ui.COLOR[n], n) for n in RULES if n in CURVES]
                                   + [("dot", ui.C["muted"], "large marker = your settings")]),
                    unsafe_allow_html=True)
        st.plotly_chart(ui.frontier_figure(pts, CURVES, BEST), theme=None, width="stretch",
                        config={"displayModeBar": False})

    st.markdown('<div class="tm pt sp">Results over all paths, same price paths for every strategy</div>',
                unsafe_allow_html=True)
    st.markdown(ui.metrics_table_html(SUMS, BEST, LABEL), unsafe_allow_html=True)
    st.markdown(
        '<div class="tm note">Objective is mean P&amp;L minus ½ κ times its variance, the quantity the agent and the '
        'Whalley-Wilmott band are built to maximise (highlighted row is the best of the first four). Mean P&amp;L shows ± one standard '
        'error over paths. The tuned rule is the best of periodic every 1 to 5 steps, thresholds from 0.5% to 8%, and the '
        'Whalley-Wilmott band, picked on these same paths, which favours the rules. Cost includes the half spread, fee and impact on every trade and the final unwind. '
        'Turnover is shares traded divided by book size.</div>', unsafe_allow_html=True)


def path_replay():
    agent_pnl = RES[ui.AGENT].pnl if ui.AGENT in RES else RES[BEST].pnl
    order = np.argsort(agent_pnl)
    c1, c2, c3 = st.columns([1.3, 1, 3])
    mode = c1.selectbox("Path", ["Median result", "Worst result", "Best result", "Pick a number"],
                        key="path_mode", help="Chosen by the agent's final P&L.")
    num = c2.number_input("Path #", min_value=1, max_value=P.n, value=1, step=1, key="path_num",
                          disabled=mode != "Pick a number")
    i = {"Median result": int(order[len(order) // 2]), "Worst result": int(order[0]), "Best result": int(order[-1]),
         "Pick a number": int(num) - 1}[mode]
    show = [n for n in NAMES]
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        st.markdown(ui.strat_legend(show), unsafe_allow_html=True)
        st.plotly_chart(ui.path_figure(P, RES, i, show), theme=None, width="stretch", config={"displayModeBar": False})
    with right:
        sel = st.radio("P&L bridge for", show, key="bridge_sel", format_func=lambda n: LABEL[n].split(",")[0])
        r = RES[sel]
        parts = {k: float(v[i]) for k, v in r.parts.items()}
        st.plotly_chart(ui.bridge_figure(parts, float(r.pnl[i])), theme=None, width="stretch",
                        config={"displayModeBar": False})
        st.markdown(ui.kv_html([
            ("Final P&L", ui._colored(r.pnl[i], ui.usd(r.pnl[i], dec=0))),
            ("Transaction cost", f"${-parts['Costs']:,.0f}"),
            ("Trades", f"{int(r.n_trades[i])}"),
            ("Turnover", f"{r.turnover[i]:.2f}× book"),
            ("Final spot", f"{P.S[i, -1]:,.2f} ({100 * (P.S[i, -1] / P.S[i, 0] - 1):+.1f}%)".replace("-", "−")),
        ]), unsafe_allow_html=True)
    who = ui.AGENT if ui.AGENT in RES else BEST
    st.markdown(f'<div class="tm pt sp">Trade blotter, {LABEL[who].split(",")[0]}</div>', unsafe_allow_html=True)
    tr = RES[who].trades[i]
    before = P.D[i, :-1] + np.concatenate([[0.0], RES[who].h[i, :-2]])     # book delta plus the stock held before the trade
    rows_ = []
    for t in np.where(np.abs(tr) > 1e-9)[0]:
        c = H.trade_cost(tr[t], P.S[i, t], P.Q[i], P.spread_bps[i], P.fee[i], P.impact_bps[i])
        rows_.append(dict(day=t / P.spd, shares=float(tr[t]), px=float(P.S[i, t]), cost=float(c),
                          before=float(before[t]), after=float(before[t] + tr[t])))
    st.markdown('<div style="max-height:340px;overflow:auto;border-radius:6px">' + ui.blotter_html(rows_) + "</div>",
                unsafe_allow_html=True)


def agent_view():
    if policy is None:
        st.warning("policy.npz is not in the app folder. Run train.py to create it.")
        return
    tau = np.linspace(P.tau[0], 0.5 / H.DAYS, 70)
    S = np.full((1, len(tau)), S0)
    _, D, G = H.book_state(S, CORE["book"], tau, CFG["r"], CFG["sig_i"])
    net_pct = np.linspace(-12, 12, 97)
    Q = float(P.Q[0])
    net = net_pct[None, :] / 100.0 * Q * np.ones((len(tau), 1))
    Gm = G[0][:, None] * np.ones((1, len(net_pct)))
    cost_bps = CFG["spread"] + CFG["fee"] / S0 * 1e4
    x = features(np, net, Gm, S0, CFG["sig_i"], Q, 1.0 / (H.DAYS * CFG["spd"]), cost_bps, CFG["impact"], CFG["kappa"],
                 tau[:, None] * H.DAYS, tau[:, None] * H.DAYS)
    f = policy.fraction(x.reshape(-1, x.shape[-1])).reshape(net.shape)
    f = np.where(f < 0.03, 0.0, f)
    u = float(H.risk_unit(Q, S0, CFG["sig_i"]))
    ww = H.ww_band(S0, G[0], tau, CFG["r"], cost_bps, CFG["kappa"], u) / Q * 100.0
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        st.markdown(ui.legend_html([("line", ui.C["long"], "share of net delta the agent trades"),
                                    ("dash", ui.C["time"], "Whalley-Wilmott no trade band")]), unsafe_allow_html=True)
        st.plotly_chart(ui.policy_figure(tau * H.DAYS, net_pct, f.T, ww), theme=None, width="stretch",
                        config={"displayModeBar": False})
        st.markdown(
            '<div class="tm note">Read it like a trader: along a column, how much of the unhedged delta does the agent '
            'take off at that point in the life of the book. Dark cells are the agent waiting, so the dark band around '
            'zero is its no trade region. It is never told about a band, it finds one, and it trades harder as expiry '
            'nears and gamma grows. The dashed Whalley-Wilmott band is derived for continuous trading, so with daily '
            'hedges it is too wide: in the held out tests the best Whalley-Wilmott setting was usually the one with '
            'double the risk aversion, a narrower band, which is closer to where the agent sits.'
            '</div>', unsafe_allow_html=True)
    with right:
        meta = policy.meta
        n_par = policy.W1.size + policy.b1.size + policy.W2.size + policy.b2.size + policy.W3.size + policy.b3.size
        st.markdown('<div class="tm pt">Agent</div>', unsafe_allow_html=True)
        st.markdown(ui.kv_html([
            ("Method", "Deep hedging, trained by pathwise policy gradient"),
            ("Reward per step", "ΔP&amp;L/u − ½ κ (ΔP&amp;L/u)²"),
            ("State", "9 scale free features"),
            ("Action", "share of net delta to trade, 0 to 1"),
            ("Network", f"2 × {policy.W1.shape[0]} ReLU, {n_par:,} weights"),
        ], wrap=True), unsafe_allow_html=True)
        log = load_log()
        if log:
            st.markdown('<div class="tm pt sp">Training</div>', unsafe_allow_html=True)
            st.markdown(ui.kv_html([
                ("Random scenarios seen", f'{log["final"]["scenarios_seen"] / 1e6:.1f} million'),
                ("Final objective, held out", f'{log["final"]["val"]:.3f}'),
                ("Whalley-Wilmott", f'{log["final"]["val_ww"]:.3f}'),
                ("Periodic", f'{log["final"]["val_per"]:.3f}'),
            ], wrap=True), unsafe_allow_html=True)
    log = load_log()
    if log:
        st.markdown('<div class="tm pt sp">Learning curve, first training run</div>', unsafe_allow_html=True)
        st.markdown(ui.legend_html([("line", ui.C["long"], "Agent"), ("dash", ui.C["aqua"], "Whalley-Wilmott"),
                                    ("dash", ui.C["time"], "Periodic")]), unsafe_allow_html=True)
        st.plotly_chart(ui.training_figure(log), theme=None, width="stretch", config={"displayModeBar": False})


@st.cache_resource(show_spinner=False)
def validation_checks():
    # no cost hedging error shrinks like 1/sqrt(N)
    mkt = H.Market()
    book = H.make_book("Short ATM straddle", 100.0)
    steps, stds, means, ses = [], [], [], []
    for spd in (1, 2, 4, 8, 16):
        hz = H.Horizon(30, 30, spd)
        Pz = H.build_paths(mkt, H.Costs(0.0, 0.0, 0.0), hz, book, 4000, seed=11)
        r = H.run_strategy("p", H.periodic(1), Pz)
        steps.append(Pz.T)
        stds.append(float(r.pnl.std()))
        means.append(float(r.pnl.mean()))
        ses.append(float(r.pnl.std() / np.sqrt(len(r.pnl))))
    slope = float(np.polyfit(np.log(steps), np.log(stds), 1)[0])
    # put call parity of the book valuation: long call + short put = S - K e^{-r tau}
    bk = H.Book(np.array([100.0, 100.0]), np.array([1.0, -1.0]), np.array([True, False]))
    V, _, _ = H.book_state(np.array([[80.0]]), bk, np.array([0.1]), 0.03, 0.25)
    par = abs(float(V[0, 0]) - (80.0 - 100.0 * np.exp(-0.03 * 0.1)))
    # the book valuation agrees with the Black-Scholes toolkit (the options pricer's engine)
    S_ = np.array([[90.0, 100.0, 110.0]])
    tau_ = np.array([0.2, 0.1, 0.05])
    bs_err = 0.0
    for is_call in (True, False):
        bk1 = H.Book(np.array([100.0]), np.array([1.0]), np.array([is_call]))
        V1, D1, G1 = H.book_state(S_, bk1, tau_, 0.03, 0.25)
        for j in range(3):
            ref = bs_np(S_[0, j], 100.0, tau_[j], 0.03, 0.25, is_call)
            bs_err = max(bs_err, abs(V1[0, j] - ref["price"]), abs(D1[0, j] - ref["delta"]), abs(G1[0, j] - ref["gamma"]))
    return dict(steps=steps, stds=stds, means=means, ses=ses, slope=slope, parity=par, bs_err=bs_err)


def model_view():
    chk = validation_checks()
    br = max(float(np.abs(sum(r.parts.values()) - r.pnl).max()) for r in RES.values())
    rows = [
        ("P&amp;L bridge: book + stock hedge + carry + costs equals final P&amp;L, every path, every strategy",
         f"max error ${br:.1e}", br < 1e-6),
        ("No cost hedging error falls like 1/√N (daily hedging up to 16 per day, straddle)",
         f'log-log slope {chk["slope"]:.2f}, expected −0.50', abs(chk["slope"] + 0.5) < 0.05),
        ("Fair marks, no cost, 16 hedges per day: mean P&amp;L is zero within noise",
         f'mean {ui.usd(chk["means"][-1], dec=0)}, standard error ${chk["ses"][-1]:,.0f}',
         abs(chk["means"][-1]) < 3 * chk["ses"][-1]),
        ("Book valuation obeys put call parity", f'error {chk["parity"]:.1e}', chk["parity"] < 1e-9),
        ("Book price, delta and gamma match the Black-Scholes toolkit (the options pricer's engine)",
         f'max error {chk["bs_err"]:.1e}', chk["bs_err"] < 1e-9),
        ("Every strategy runs on identical price paths", f"{P.n:,} independent paths, paired comparison", True),
    ]
    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        st.markdown('<div class="tm pt">Validation</div>', unsafe_allow_html=True)
        st.markdown(ui.checks_html(rows), unsafe_allow_html=True)
        st.markdown('<div class="tm pt sp">How the desk is modelled</div>', unsafe_allow_html=True)
        st.markdown(ui.kv_html([
            ("Total P&amp;L", "Π = V + B + h·S, book mark plus cash plus stock hedge"),
            ("Trade cost", "|d|·(S·half spread + fee) + S·impact·d²/Q"),
            ("Step", "trade at the start, then the price moves, cash earns r"),
            ("End of horizon", "remaining stock hedge is unwound at the same cost"),
            ("Hedge ratio", "Black-Scholes delta at the marked vol, summed over the book"),
            ("Risk unit u", "20% of the book's one day 1σ move, so κ is a pure number"),
        ]), unsafe_allow_html=True)
    with right:
        st.plotly_chart(ui.scaling_figure(chk["steps"], chk["stds"]), theme=None, width="stretch",
                        config={"displayModeBar": False})
        st.markdown('<div class="tm pt sp">References</div>', unsafe_allow_html=True)
        st.markdown(ui.refs_html(), unsafe_allow_html=True)
    st.markdown('<div class="tm pt sp">Assumptions and limits</div>', unsafe_allow_html=True)
    st.markdown(ui.assumptions_html(), unsafe_allow_html=True)


{"Monte Carlo": monte_carlo, "Path replay": path_replay, "Agent": agent_view, "Model": model_view}[ss.view]()
