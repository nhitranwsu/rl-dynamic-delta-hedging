"""Presentation layer for the RL Delta Hedging Simulator."""
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
[data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); }

.tm, .tm * { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; box-sizing: border-box; }
.tm { color: var(--ink); font-size: 14px; font-variant-numeric: tabular-nums; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }

.brand { display: flex; align-items: baseline; gap: 14px; padding: 12px 16px; background: var(--panel);
         border: 1px solid var(--line); border-radius: 6px; margin-bottom: 12px; }
.brand b { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
.brand span { color: var(--ink2); font-size: 13px; }

.pt { color: var(--ink2); font-size: 13px; font-weight: 500; margin: 0 0 6px; }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; margin-bottom: 8px; line-height: 1.45; }

/* stats table */
.blot { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.blot table { width: 100%; border-collapse: collapse; font-size: 13px; }
.blot th { color: var(--muted); font-weight: 500; font-size: 12px; text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
.blot td { text-align: right; padding: 0 9px; height: 34px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.blot tr:last-child td { border-bottom: 0; }
.blot th:first-child, .blot td:first-child { text-align: left; color: var(--ink); }
"""

def css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in C.items()) + "}"
    font = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
    return "<style>" + font + root + _CSS_BODY + "</style>"

def brand_html():
    return ('<div class="tm brand"><b>RL Hedging Engine</b>'
            '<span>Deep Q-Network benchmark against deterministic strategies</span></div>')

def benchmark_table_html(results):
    head = "<tr><th>Strategy</th><th>Final P&amp;L</th><th>Trades</th><th>Fees Paid</th><th>Avg Delta Exposure</th></tr>"
    body = ""
    for strat, data in results.items():
        pnl = data["final_pnl"]
        trades = data["trades"]
        fees = data["costs"]
        avg_delta = sum(abs(x) for x in data["delta_path"]) / len(data["delta_path"])
        
        pnl_str = f'<span class="{"gain" if pnl >= 0 else "loss"}">{pnl:+.2f}</span>'
        body += f"<tr><td>{strat}</td><td>{pnl_str}</td><td>{trades}</td><td>${fees:.2f}</td><td>{avg_delta:.1f} shs</td></tr>"
    return f'<div class="tm blot"><table>{head}{body}</table></div>'

def plot_benchmark(results):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Cumulative P&L (Mark to Market)", "Net Delta Exposure"))
    
    for strat, data in results.items():
        color = C["aqua"]
        if "No Hedge" in strat: color = C["short"]
        elif "Periodic" in strat: color = C["time"]
        elif "Threshold" in strat: color = C["vol"]

        # P&L path
        fig.add_trace(go.Scatter(y=data["pnl"], mode="lines", name=strat,
                                 line=dict(color=color, width=2.5 if strat.startswith("RL") else 1.5)),
                      row=1, col=1)
        # Delta path
        fig.add_trace(go.Scatter(y=data["delta_path"], mode="lines", name=strat, showlegend=False,
                                 line=dict(color=color, width=1.5)),
                      row=2, col=1)

    fig.update_layout(
        height=650, paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
        font=dict(family=FONT, size=12, color=C["ink2"]), margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["axis"], font=dict(family=FONT, size=12, color=C["ink"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C["axis"], tickfont=dict(color=C["muted"]))
    fig.update_xaxes(title_text="Trading Days Passed", row=2, col=1)
    fig.update_yaxes(gridcolor=C["line"], zerolinecolor=C["axis"], linecolor=C["axis"], tickfont=dict(color=C["muted"]))
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)
    
    return fig