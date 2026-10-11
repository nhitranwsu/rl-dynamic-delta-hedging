import streamlit as st
import ui
from benchmark import BenchmarkRunner

st.set_page_config(page_title="RL Delta Hedging", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.css(), unsafe_allow_html=True)

st.markdown(ui.brand_html(), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="tm pt sp">Simulation Parameters</div>', unsafe_allow_html=True)
    seed = st.number_input("Random Seed (Scenario)", value=42, step=1, 
                           help="Change this to generate a completely new price path and options book.")
    threshold = st.slider("Strict Threshold Rule Size", 10.0, 100.0, 25.0, 5.0,
                          help="The hard-coded delta limit before the deterministic model decides to hedge.")
    st.button("Run Benchmark", type="primary", width="stretch")

@st.cache_resource
def load_runner():
    try:
        return BenchmarkRunner("dqn_hedger.pth")
    except FileNotFoundError:
        return None

runner = load_runner()

if runner is None:
    st.error("Model weights not found. Please ensure `dqn_hedger.pth` is in the same directory.")
else:
    # Run the head-to-head comparison
    results = runner.run_comparison(env_seed=seed, threshold=threshold)
    
    left, right = st.columns([2.2, 1], gap="medium")
    
    with left:
        st.markdown('<div class="tm pt">Strategy Trajectories (Paired Scenario)</div>', unsafe_allow_html=True)
        fig = ui.plot_benchmark(results)
        st.plotly_chart(fig, theme=None, use_container_width=True, config={"displayModeBar": False})
        
    with right:
        st.markdown('<div class="tm pt">Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown(ui.benchmark_table_html(results), unsafe_allow_html=True)
        
        st.markdown('<div class="tm pt sp">Key Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="tm note"><b>RL Agent (DQN):</b> Learns a dynamic threshold. Because it factors in Time to Expiry (Gamma risk), it avoids overtrading on small delta moves, saving massive transaction costs while capping tail risk.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tm note"><b>Periodic (Daily):</b> The naive industry standard. It achieves near-zero delta exposure but suffers extreme "hedge turnover," bleeding cash from spread/fee costs.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tm note"><b>Threshold:</b> Ignores time and gamma. It strictly hedges when a static delta bound is breached, which is rigid and often sub-optimal.</div>', unsafe_allow_html=True)