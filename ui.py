"""Presentation layer for the delta hedging desk: theme, HTML panels and Plotly figures.

Everything here returns strings or figures, so app.py stays about inputs and layout only.
Same design system as the other two desk tools. Each strategy keeps one color everywhere: the learned agent is blue,
Whalley-Wilmott green, periodic amber, threshold purple, unhedged red and dotted. Green and red text only mean gain and loss.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C = dict(
    bg="#0d0d0d", panel="#1a1a19", panel2="#222221", line="#2c2c2a", axis="#383835",
    ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    long="#3987e5", short="#e66767", time="#c98500", vol="#9085e9",
    gain="#0ca30c", loss="#e66767", aqua="#199e70",
)
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

_CSS_BODY = """
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: 1.2rem 1.6rem 2.5rem; max-width: 100%; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stMarkdown p { font-size: 13px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); }

.tm, .tm * { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; box-sizing: border-box; }
.tm { color: var(--ink); font-size: 14px; font-variant-numeric: tabular-nums; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }
.mu { color: var(--muted); }

/* top bar */
.brand { display: flex; align-items: baseline; gap: 14px; padding: 12px 16px; background: var(--panel);
         border: 1px solid var(--line); border-radius: 6px; }
.brand b { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
.brand span { color: var(--ink2); font-size: 13px; }

/* status strip */
.meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 24px; padding: 10px 2px 12px; font-size: 13px;
        border-bottom: 1px solid var(--line); margin-bottom: 12px; }
.meta .kv em { font-style: normal; color: var(--muted); margin-right: 7px; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 3px 11px 3px 9px; border-radius: 999px;
        border: 1px solid var(--axis); font-weight: 600; font-size: 12px; color: var(--ink); }
.pill i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.pill.long i { background: var(--long); }
.pill.short i { background: var(--short); }

/* chart legend, rendered as HTML so it never wraps over the panel labels */
.leg { display: flex; flex-wrap: wrap; gap: 4px 18px; align-items: center; background: var(--panel); border: 1px solid var(--line);
       border-radius: 6px; padding: 8px 14px; margin: 0; color: var(--ink2); font-size: 12px; }
.leg span { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
.leg i { display: inline-block; width: 16px; height: 0; border-top: 2px solid; }
.leg i.dash { border-top-style: dashed; }
.leg i.dot { border-top-style: dotted; }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--line);
        border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-bottom: 12px; }
.kpi { background: var(--panel); padding: 11px 15px 12px; min-width: 0; }
.kpi .l { color: var(--muted); font-size: 12px; }
.kpi .v { font-size: 27px; font-weight: 600; line-height: 1.2; margin-top: 3px; white-space: nowrap; }
.kpi .s { color: var(--ink2); font-size: 12px; line-height: 1.35; margin-top: 3px; }

/* section titles */
.pt { color: var(--ink2); font-size: 13px; font-weight: 500; margin: 0 0 6px; }
.pt.sp { margin-top: 14px; }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; margin-bottom: 8px; line-height: 1.45; }

/* key value tables */
.kvt { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.kvt table { width: 100%; border-collapse: collapse; font-size: 13px; }
.tm table th, .tm table td { border-left: 0; border-right: 0; border-top: 0; background: transparent; }
.kvt td { padding: 4px 10px; min-height: 28px; border-bottom: 1px solid var(--line); line-height: 1.3; }
.kvt tr:last-child td { border-bottom: 0; }
.kvt td:first-child { color: var(--ink2); white-space: normal; }
.kvt td:last-child { text-align: right; font-weight: 500; white-space: nowrap; }
.kvt tr.tot td { border-top: 1px solid var(--axis); font-weight: 600; }
.kvt tr.tot td:first-child { color: var(--ink); }

/* generic data tables */
.blot { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.blot table { width: 100%; border-collapse: collapse; font-size: 13px; }
.blot th { color: var(--muted); font-weight: 500; font-size: 12px; text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
.blot td { text-align: right; padding: 0 10px; height: 27px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.blot tr:last-child td { border-bottom: 0; }
.blot th:first-child, .blot td:first-child { text-align: left; }
.blot.txt th, .blot.txt td { text-align: left; vertical-align: top; height: auto; padding: 10px 14px; line-height: 1.5; font-size: 13px; white-space: normal; }
.blot.txt td:first-child { color: var(--ink); font-weight: 500; white-space: nowrap; }
.blot.txt td { color: var(--ink2); }

/* option chain */
.chain th.grp { text-align: center; color: var(--ink2); font-weight: 600; letter-spacing: .04em; border-bottom: 1px solid var(--axis); }
.chain td.k { text-align: center; font-weight: 600; background: var(--panel2); color: var(--ink); }
.chain td.mid { color: var(--ink2); background: rgba(240,239,236,.03); }
.chain td.itm { background: rgba(57,135,229,.10); }
.chain tr.atm td { border-top: 1px solid var(--ink2); border-bottom: 1px solid var(--ink2); }
.chain th.k { text-align: center; }
.sep { border-left: 1px solid var(--axis); }

/* solver result */
.res { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; display: flex;
       justify-content: space-between; align-items: baseline; gap: 16px; margin-top: 8px; }
.res .l { color: var(--muted); font-size: 12px; }
.res .v { font-size: 26px; font-weight: 600; line-height: 1.2; }
.res .s { color: var(--ink2); font-size: 12px; text-align: right; }
.res.err { border-color: var(--short); }

.chip { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 9px; vertical-align: middle; }
.blot.hl tr.best td { background: rgba(57,135,229,.08); }
.blot.hl td:first-child { color: var(--ink); font-weight: 500; }
.blot.txt.wrap td:first-child { white-space: normal; width: 46%; }
.kvt.wr td:last-child { white-space: normal; text-align: right; }
.longc { color: var(--long); font-weight: 600; }
.shortc { color: var(--short); font-weight: 600; }

/* author and copyright */
.author { padding: 10px 13px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel2); font-size: 12px;
          color: var(--ink2); line-height: 1.45; margin-bottom: 14px; }
.author b { display: block; color: var(--ink); font-weight: 600; font-size: 14px; letter-spacing: -0.01em; }
.foot { margin-top: 30px; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; line-height: 1.5; }
"""


def css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in C.items()) + "}"
    font = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
    return "<style>" + font + root + _CSS_BODY + "</style>"


# ------------------------------------------------------------------------------------------------- helpers
def _sg(x):
    return "gain" if x >= 0 else "loss"


def _rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def usd(x, sign=True, dec=2):
    s = f"{abs(x):,.{dec}f}"
    if sign:
        return f"{'+' if x >= 0 else '−'}${s}"
    return f"{'−' if x < 0 else ''}${s}"


def _n(x, spec):
    """Signed number with a true minus sign, e.g. _n(-1.5, ".2f") gives '−1.50'."""
    return format(x, "+" + spec).replace("-", "−")


def sgn(x, dec=4):
    return f"{'+' if x >= 0 else '−'}{abs(x):.{dec}f}"


def _dec(price):
    return 2 if abs(price) >= 1.0 else 4


def _px(x, dec=None):
    d = _dec(x) if dec is None else dec
    return f"{x:,.{d}f}"


def _colored(x, text):
    return f'<span class="{_sg(x)}">{text}</span>'


def expiry_label(c):
    if c.dpy == 252.0:
        return f"{c.T * 252.0 * 6.5:.2f} trading hrs"
    return f"{c.T * 365.0:.1f} days"




def css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in C.items()) + "}"
    font = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
    return "<style>" + font + root + _CSS_BODY + "</style>"


def _sg(x):
    return "gain" if x >= 0 else "loss"


def _rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def usd(x, sign=True, dec=2):
    s = f"{abs(x):,.{dec}f}"
    if sign:
        return f"{'+' if x >= 0 else '−'}${s}"
    return f"{'−' if x < 0 else ''}${s}"


def _n(x, spec):
    """Signed number with a true minus sign, e.g. _n(-1.5, ".2f") gives '−1.50'."""
    return format(x, "+" + spec).replace("-", "−")


def _colored(x, text):
    return f'<span class="{_sg(x)}">{text}</span>'


# ------------------------------------------------------------------------------------------------- strategies
AGENT = "Agent"
WW = "Whalley-Wilmott"
PER = "Periodic"
THR = "Threshold"
UNH = "Unhedged"
TUNED = "Best tuned"
ORDER = [AGENT, WW, PER, THR, TUNED, UNH]
COLOR = {AGENT: C["long"], WW: C["aqua"], PER: C["time"], THR: C["vol"], UNH: C["short"], TUNED: C["ink2"]}
DASH = {AGENT: "solid", WW: "solid", PER: "solid", THR: "solid", UNH: "dot", TUNED: "dash"}
WIDTH = {AGENT: 2.75, WW: 1.75, PER: 1.75, THR: 1.75, UNH: 1.5, TUNED: 1.5}


def _chip(name):
    return f'<i class="chip" style="background:{COLOR[name]}"></i>'


# ------------------------------------------------------------------------------------------------- HTML panels


def author_html():
    return ('<div class="tm author"><b>Nhi Q Tran</b>'
            'Designed and built by Nhi Q Tran. Source code is private.</div>')


def foot_html():
    return ('<div class="tm foot">&copy; 2026 Nhi Q Tran. All rights reserved. '
            'The source code is private and may not be copied or redistributed. '
            'Research and education tool, not investment advice.</div>')


def brand_html():
    return ('<div class="tm brand"><b>Delta hedging desk</b>'
            '<span>Hedging an options book: learned policy against desk rules</span></div>')


def meta_html(m):
    gam = "short gamma" if m["gamma"] < 0 else "long gamma"
    cls = "short" if m["gamma"] < 0 else "long"
    return (f'<div class="tm meta"><span class="pill {cls}"><i></i>{gam.upper()}</span>'
            f'<span class="kv"><em>Book</em>{m["book"]}</span>'
            f'<span class="kv"><em>Size</em>{m["Q"]:,.0f} sh gross</span>'
            f'<span class="kv"><em>Start delta</em>{_n(m["delta"], ",.0f")} sh</span>'
            f'<span class="kv"><em>Start gamma</em>{_n(m["gamma"], ",.1f")} sh per $</span>'
            f'<span class="kv"><em>Vol</em>{m["sig_i"]:.0f}% marked, {m["sig_r"]:.0f}% realised</span>'
            f'<span class="kv"><em>Horizon</em>{m["days"]}d, {m["spd"]}× per day</span>'
            f'<span class="kv"><em>Costs</em>{m["spread"]:.1f} bps half spread'
            f'{", impact " + format(m["impact"], ".0f") + " bps" if m["impact"] else ""}</span>'
            f'<span class="kv"><em>Paths</em>{m["n"]:,}</span></div>')


def _tile(label, value, sub=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def kpi_html(tiles):
    return '<div class="tm kpis">' + "".join(_tile(*t) for t in tiles) + "</div>"


def legend_html(items):
    """items: (kind, color, label) with kind in line, dash, dot."""
    out = ""
    for kind, color, label in items:
        cls = "" if kind == "line" else kind
        out += f'<span><i class="{cls}" style="border-color:{color}"></i>{label}</span>'
    return '<div class="tm leg">' + out + "</div>"


def strat_legend(names, extra=()):
    items = [("dot" if DASH[n] == "dot" else "line", COLOR[n], n) for n in names]
    return legend_html(items + list(extra))


def metrics_table_html(rows, best, labels):
    """rows: {name: summary dict}. best: name with the highest objective."""
    head = ("<tr><th>Strategy</th><th>Mean P&amp;L</th><th>Std (hedging error)</th><th>Worst 5% avg</th>"
            "<th>Cost</th><th>Turnover</th><th>Trades</th><th>Avg |net δ|</th><th>Objective</th></tr>")
    body = ""
    for name in ORDER:
        if name not in rows:
            continue
        s = rows[name]
        cls = ' class="best"' if name == best else ""
        body += (f"<tr{cls}><td>{_chip(name)}{labels.get(name, name)}</td>"
                 f'<td>{_colored(s["mean"], usd(s["mean"], dec=0))}'
                 f' <span class="mu">±{s["se_mean"]:,.0f}</span></td>'
                 f'<td>${s["std"]:,.0f}</td>'
                 f'<td>{_colored(s["es5"], usd(s["es5"], dec=0))}</td>'
                 f'<td>${s["cost"]:,.0f}</td><td>{s["turnover"]:.2f}×</td><td>{s["trades"]:.1f}</td>'
                 f'<td>{s["abs_net"] * 100:.1f}%</td>'
                 f'<td>{_colored(s["objective"], usd(s["objective"], dec=0))}</td></tr>')
    return f'<div class="tm blot hl"><table>{head}{body}</table></div>'


def blotter_html(rows):
    head = ("<tr><th>Day</th><th>Side</th><th>Shares</th><th>Price</th><th>Notional</th><th>Cost</th>"
            "<th>Net δ before</th><th>Net δ after</th></tr>")
    body = ""
    for r in rows:
        side = f'<span class="{"long" if r["shares"] > 0 else "short"}c">{"BUY" if r["shares"] > 0 else "SELL"}</span>'
        body += (f'<tr><td>{r["day"]:.1f}</td><td>{side}</td><td>{abs(r["shares"]):,.0f}</td>'
                 f'<td>{r["px"]:,.2f}</td><td>${abs(r["shares"]) * r["px"]:,.0f}</td><td>${r["cost"]:,.2f}</td>'
                 f'<td>{_n(r["before"], ",.0f")}</td><td>{_n(r["after"], ",.0f")}</td></tr>')
    if not rows:
        body = '<tr><td colspan="8" class="mu" style="text-align:center">no trades on this path</td></tr>'
    return f'<div class="tm blot"><table>{head}{body}</table></div>'


def kv_html(rows, total_last=False, wrap=False):
    body = ""
    for i, (k, v) in enumerate(rows):
        cls = ' class="tot"' if (total_last and i == len(rows) - 1) else ""
        body += f"<tr{cls}><td>{k}</td><td>{v}</td></tr>"
    return f'<div class="tm kvt{" wr" if wrap else ""}"><table>{body}</table></div>'


def checks_html(rows):
    """rows: (what, result text, ok)."""
    body = ""
    for what, res, ok in rows:
        mark = '<span class="gain">PASS</span>' if ok else '<span class="loss">CHECK</span>'
        body += f"<tr><td>{what}</td><td>{res}</td><td>{mark}</td></tr>"
    return ('<div class="tm blot txt wrap"><table><tr><th>Check</th><th>Result</th><th></th></tr>'
            f"{body}</table></div>")


def assumptions_html():
    rows = [
        ("Price process", "GBM with optional Merton jumps and a per path vol dispersion, constant parameters.",
         "Stochastic vol, skew dynamics, vol that moves with spot, gap risk at events."),
        ("Marks and delta", "Black-Scholes at one flat vol. The desk marks and hedges at its implied vol, "
                            "the world moves at the realised vol.", "A live surface and model delta that includes skew."),
        ("Risk being hedged", "Delta only. The book keeps its gamma, vega and theta, so P&L still moves with realised vol.",
         "Gamma and vega hedged with listed options, and the hedge cost of those."),
        ("Transaction costs", "Half spread and per share fee are linear, impact is quadratic in trade size.",
         "Queue position, passive against aggressive fills, venue fees, borrow on the stock hedge."),
        ("Objective", "Mean minus half kappa times variance of P&L, with kappa set by the risk aversion slider.",
         "A desk limit framework: delta, gamma and vega limits, stop outs, capital."),
        ("Agent", "Trained offline on random books, vols, costs, jumps and horizons inside the ranges of the sliders. "
                  "Outside those ranges the app does not let you run it.", "Out of sample behaviour in a real market."),
        ("Single underlying", "One stock, no correlation, no hedge of the stock with an index.",
         "Cross asset books and basis risk."),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return ('<div class="tm blot txt"><table><tr><th>Component</th><th>This app</th>'
            f'<th>What a desk will push on</th></tr>{body}</table></div>')


def refs_html():
    rows = [
        ("Whalley and Wilmott (1997)", "Optimal hedging with transaction costs: the no trade band and its width formula."),
        ("Leland (1985)", "Option pricing and replication with transaction costs."),
        ("Kolm and Ritter (2019)", "Dynamic replication and hedging as reinforcement learning, with the mean variance reward used here."),
        ("Buehler, Gonon, Teichmann and Wood (2019)", "Deep hedging: learning a hedging policy by gradient descent on a risk measure."),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
    return f'<div class="tm blot txt wrap"><table>{body}</table></div>'


# ------------------------------------------------------------------------------------------------- figures
def _style(fig, height, margin):
    fig.update_layout(
        height=height, paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
        font=dict(family=FONT, size=12, color=C["ink2"]), margin=margin, showlegend=False,
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["axis"], font=dict(family=FONT, size=12, color=C["ink"])),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C["axis"], tickfont=dict(size=12, color=C["muted"]))
    fig.update_yaxes(gridcolor=C["line"], gridwidth=1, zeroline=False, linecolor=C["axis"],
                     tickfont=dict(size=12, color=C["muted"]))


def _label(fig, text, idx=1):
    ax = "" if idx == 1 else str(idx)
    fig.add_annotation(xref=f"x{ax} domain", yref=f"y{ax} domain", x=0, y=1.0, text=text, showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=13, color=C["ink2"]))


def _smooth_density(x, lo, hi, bins=60, sd=2.2):
    cnt, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    k = np.exp(-0.5 * (np.arange(-6, 7) / sd) ** 2)
    k /= k.sum()
    return 0.5 * (edges[:-1] + edges[1:]), np.convolve(cnt, k, mode="same")


def dist_figure(pnls):
    """pnls: {name: array of final P&L}. Unhedged is left out, its spread would flatten everything else."""
    names = [n for n in ORDER if n in pnls and n != UNH]
    allv = np.concatenate([pnls[n] for n in names])
    lo, hi = np.percentile(allv, [0.3, 99.7])
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    fig = go.Figure()
    for n in names:
        x, y = _smooth_density(pnls[n], lo, hi)
        kw = dict(fill="tozeroy", fillcolor=_rgba(COLOR[n], 0.10)) if n == AGENT else {}
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=n, line=dict(color=COLOR[n], width=WIDTH[n]),
                                 hovertemplate=f"{n}<br>P&L %{{x:$,.0f}}<extra></extra>", **kw))
    _style(fig, 400, dict(l=24, r=20, t=30, b=44))
    fig.update_xaxes(tickformat="$,.0f", zeroline=True, zerolinecolor=C["axis"], range=[lo, hi])
    fig.update_yaxes(showticklabels=False, rangemode="tozero")
    _label(fig, "Distribution of final P&L (unhedged left out)")
    return fig


def frontier_figure(points, curves, selected):
    """points: {name: (cost, std)} for the chosen settings. curves: {name: [(cost, std), ...]} parameter sweeps."""
    fig = go.Figure()
    for n in ORDER:
        if n in curves and len(curves[n]) > 1:
            xs, ys = zip(*curves[n])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=COLOR[n], width=1.5),
                                     marker=dict(size=5, color=COLOR[n]), opacity=0.8, hoverinfo="skip"))
    for n in ORDER:
        if n in points and n != UNH:
            c, s = points[n]
            fig.add_trace(go.Scatter(x=[c], y=[s], mode="markers", name=n,
                                     marker=dict(size=13, color=COLOR[n], line=dict(width=2, color=C["panel"])),
                                     hovertemplate=f"{n}<br>cost %{{x:$,.0f}}<br>std %{{y:$,.0f}}<extra></extra>"))
    _style(fig, 400, dict(l=64, r=20, t=30, b=48))
    fig.update_xaxes(title_text="Mean hedging cost", title_font=dict(size=12, color=C["muted"]), tickformat="$,.0f",
                     showgrid=True, gridcolor=C["line"], rangemode="tozero")
    fig.update_yaxes(title_text="Std of P&L", title_font=dict(size=12, color=C["muted"]), tickformat="$,.0f")
    _label(fig, "Cost against risk. Lower left is better")
    return fig


def path_figure(P, res, i, names):
    t = np.arange(P.T + 1) / P.spd
    Q = float(P.Q[i])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.26, 0.26, 0.30, 0.18])
    fig.add_trace(go.Scatter(x=t, y=P.S[i], mode="lines", line=dict(color=C["ink2"], width=1.75),
                             hovertemplate="day %{x:.1f}<br>spot %{y:,.2f}<extra></extra>"), row=1, col=1)
    if AGENT in res:
        tr = res[AGENT].trades[i]
        idx = np.where(np.abs(tr) > 1e-9)[0]
        for sign, sym, col, nm in ((1, "triangle-up", C["long"], "buy"), (-1, "triangle-down", C["short"], "sell")):
            sel = idx[np.sign(tr[idx]) == sign]
            if len(sel):
                fig.add_trace(go.Scatter(
                    x=t[sel], y=P.S[i, sel], mode="markers", name=nm,
                    marker=dict(symbol=sym, size=np.clip(6 + 14 * np.abs(tr[sel]) / Q / 0.1, 6, 18), color=col,
                                line=dict(width=1, color=C["panel"])),
                    hovertemplate=f"agent {nm} %{{customdata:,.0f}} sh<extra></extra>", customdata=np.abs(tr[sel])),
                    row=1, col=1)
    for n in names:
        r = res[n]
        kw = dict(color=COLOR[n], width=WIDTH[n], dash=DASH[n])
        net = r.net[i, :P.T]
        fig.add_trace(go.Scatter(x=t[:P.T], y=100 * net / Q, mode="lines", line=kw, line_shape="hv",
                                 hovertemplate=f"{n}<br>net δ %{{y:.1f}}% of book<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=r.cum[i], mode="lines", line=kw,
                                 hovertemplate=f"{n}<br>P&L %{{y:$,.0f}}<extra></extra>"), row=3, col=1)
        if n != UNH:
            fig.add_trace(go.Scatter(x=t, y=r.cost_cum[i], mode="lines", line=dict(color=COLOR[n], width=1.5),
                                     line_shape="hv", hovertemplate=f"{n}<br>cost %{{y:$,.0f}}<extra></extra>"),
                          row=4, col=1)
    _style(fig, 780, dict(l=64, r=20, t=26, b=44))
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor=C["axis"], row=2, col=1)
    fig.update_yaxes(tickformat="$,.0f", zeroline=True, zerolinecolor=C["axis"], row=3, col=1)
    fig.update_yaxes(tickformat="$,.0f", rangemode="tozero", row=4, col=1)
    fig.update_xaxes(title_text="Trading day", title_font=dict(size=12, color=C["muted"]), row=4, col=1)
    for k, txt in enumerate(["Spot, with the agent's trades", "Net delta after trading, % of book",
                             "Cumulative P&L", "Cumulative transaction cost"], 1):
        _label(fig, txt, k)
    return fig


def bridge_figure(parts, total):
    names = list(parts.keys()) + ["Net P&L"]
    vals = list(parts.values()) + [total]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=vals, measure=["relative"] * (len(names) - 1) + ["total"],
        text=[usd(v, dec=0) for v in vals], textposition="outside", textfont=dict(size=12, color=C["ink2"]),
        increasing=dict(marker=dict(color=C["long"])), decreasing=dict(marker=dict(color=C["short"])),
        totals=dict(marker=dict(color=C["ink2"])), connector=dict(line=dict(color=C["axis"], width=1)),
        hovertemplate="%{y}: %{x:+,.0f}<extra></extra>", cliponaxis=False))
    cum, pts = 0.0, [0.0]
    for v in vals[:-1]:
        cum += v
        pts.append(cum)
    pts.append(vals[-1])
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    _style(fig, 310, dict(l=118, r=64, t=30, b=40))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickformat="$,.0f", gridcolor=C["line"], showgrid=True, zeroline=True, zerolinecolor=C["axis"],
                     range=[lo - 1.05 * span, hi + 0.5 * span], nticks=3, tickangle=0)
    _label(fig, "P&L bridge for this path")
    return fig


def policy_figure(days, net_pct, f, ww_pct):
    """Heatmap of the share of the net delta the agent trades, with the Whalley-Wilmott band on top."""
    fig = go.Figure(go.Heatmap(
        x=days, y=net_pct, z=f, zmin=0, zmax=1, colorscale=[[0, C["panel"]], [0.5, "#27548c"], [1, C["long"]]],
        colorbar=dict(title=dict(text="share traded", font=dict(size=12, color=C["muted"])), thickness=10, len=0.9,
                      tickfont=dict(size=11, color=C["muted"]), outlinewidth=0, tickformat=".0%"),
        hovertemplate="%{x:.1f} days to expiry<br>net δ %{y:.1f}% of book<br>trade %{z:.0%} of it<extra></extra>"))
    for sgn_ in (1, -1):
        fig.add_trace(go.Scatter(x=days, y=sgn_ * ww_pct, mode="lines", line=dict(color=C["time"], width=2, dash="dash"),
                                 hoverinfo="skip"))
    _style(fig, 420, dict(l=64, r=20, t=30, b=48))
    fig.update_xaxes(title_text="Days to expiry (the hedge runs to expiry)", title_font=dict(size=12, color=C["muted"]),
                     autorange="reversed")
    fig.update_yaxes(title_text="Net delta, % of book", ticksuffix="%", title_font=dict(size=12, color=C["muted"]),
                     gridcolor="rgba(0,0,0,0)", range=[float(net_pct.min()), float(net_pct.max())])
    _label(fig, "What the agent does at each state")
    return fig


def training_figure(log):
    it = log["iter"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=it, y=log["val_per"], mode="lines", line=dict(color=C["time"], width=1.75, dash="dash"),
                             hovertemplate="Periodic %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=it, y=log["val_ww"], mode="lines", line=dict(color=C["aqua"], width=1.75, dash="dash"),
                             hovertemplate="Whalley-Wilmott %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=it, y=log["val"], mode="lines", line=dict(color=C["long"], width=2.75),
                             hovertemplate="Agent %{y:.3f}<extra></extra>"))
    _style(fig, 330, dict(l=64, r=20, t=30, b=48))
    fig.update_xaxes(title_text="Training iteration (1,024 random scenarios each)",
                     title_font=dict(size=12, color=C["muted"]))
    fig.update_yaxes(title_text="Objective, risk units", title_font=dict(size=12, color=C["muted"]), tickformat=".2f")
    _label(fig, "Six held out scenarios during training, higher is better. A second pass then averaged the weights")
    return fig


def scaling_figure(steps, stds):
    steps = np.asarray(steps, float)
    stds = np.asarray(stds, float)
    k = float(np.median(stds * np.sqrt(steps)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=stds, mode="lines+markers", line=dict(color=C["vol"], width=2.25),
                             marker=dict(size=7, color=C["vol"], line=dict(width=2, color=C["panel"])),
                             hovertemplate="%{x:.0f} hedges<br>std %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=steps, y=k / np.sqrt(steps), mode="lines", line=dict(color=C["muted"], width=1.25, dash="dash"),
                             hoverinfo="skip"))
    _style(fig, 330, dict(l=64, r=20, t=30, b=48))
    fig.update_xaxes(type="log", title_text="Hedges over the life of the book", title_font=dict(size=12, color=C["muted"]),
                     tickvals=list(steps), ticktext=[f"{int(v)}" for v in steps], minor=dict(ticks="", showgrid=False))
    tv = [float(v) for v in np.geomspace(stds.min(), stds.max(), 4)]
    fig.update_yaxes(type="log", tickvals=tv, ticktext=[f"${v:,.0f}" for v in tv], minor=dict(ticks="", showgrid=False))
    _label(fig, "No cost hedging error against hedge count (dashed: 1/√N)")
    return fig
