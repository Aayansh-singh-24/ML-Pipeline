import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
    --accent-cyan: #0099cc;
    --accent-purple: #7c3aed;
    --accent-pink: #db2777;
    --accent-green: #059669;
    --accent-orange: #d97706;
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background-color: #f8fafc !important;
}

.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1600px !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    color: #0099cc;
    letter-spacing: 0.05em;
    margin: 0;
}
.hero-sub {
    color: #475569;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

/* Pipeline Steps Bar */
.pipeline-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 1.5rem 0;
    overflow-x: auto;
    flex-wrap: nowrap;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.step-item {
    display: flex;
    align-items: center;
    flex-shrink: 0;
}
.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    position: relative;
}
.step-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 700;
    border: 2px solid;
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
}
.step-circle.pending {
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #94a3b8;
}
.step-circle.active {
    background: #e0f7ff;
    border-color: #0099cc;
    color: #0099cc;
    box-shadow: 0 0 0 4px rgba(0,153,204,0.15);
}
.step-circle.done {
    background: #d1fae5;
    border-color: #059669;
    color: #059669;
}
.step-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    text-align: center;
    max-width: 80px;
    line-height: 1.2;
    font-family: 'Space Mono', monospace;
}
.step-label.pending { color: #94a3b8; }
.step-label.active { color: #0099cc; font-weight: 700; }
.step-label.done { color: #059669; }

.step-connector {
    height: 2px;
    width: 40px;
    margin-bottom: 22px;
    flex-shrink: 0;
}
.step-connector.done { background: #059669; }
.step-connector.active { background: linear-gradient(90deg, #059669, #0099cc); }
.step-connector.pending { background: #e2e8f0; }

/* Section Cards */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0099cc, #7c3aed);
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0099cc;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #0099cc; }
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0099cc;
    display: block;
}
.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
    font-family: 'Space Mono', monospace;
}

/* ─── BUTTONS ─── */
.stButton > button {
    background: #ffffff !important;
    border: 1.5px solid #0099cc !important;
    color: #0099cc !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: #e0f7ff !important;
    box-shadow: 0 0 0 3px rgba(0,153,204,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: #0099cc !important;
    color: #ffffff !important;
    border-color: #0099cc !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #007aab !important;
    border-color: #007aab !important;
}

/* ─── SELECTBOX ─── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}
.stSelectbox > div > div > div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1e293b !important;
}
.stSelectbox span,
.stSelectbox div[class*="ValueContainer"] span,
.stSelectbox div[class*="singleValue"] {
    color: #1e293b !important;
    font-size: 0.9rem !important;
}
.stSelectbox svg {
    fill: #1e293b !important;
    color: #1e293b !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {
    color: #1e293b !important;
    background: #ffffff !important;
    font-size: 0.9rem !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {
    background: #e0f7ff !important;
    color: #0c4a6e !important;
}
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="menu"] li[aria-selected="true"] {
    background: #e0f7ff !important;
    color: #0c4a6e !important;
    font-weight: 700 !important;
}

/* ─── MULTISELECT ─── */
.stMultiSelect > div > div {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    min-height: 42px !important;
}
.stMultiSelect div[class*="placeholder"],
.stMultiSelect div[class*="Placeholder"] {
    color: #64748b !important;
    font-size: 0.9rem !important;
}
.stMultiSelect span[data-baseweb="tag"],
.stMultiSelect div[data-baseweb="tag"] {
    background: #e0f7ff !important;
    color: #0c4a6e !important;
    border: 1px solid #7dd3f0 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
}
.stMultiSelect span[data-baseweb="tag"] svg,
.stMultiSelect div[data-baseweb="tag"] svg {
    fill: #0c4a6e !important;
}
.stMultiSelect input {
    color: #1e293b !important;
    background: transparent !important;
}
.stMultiSelect svg {
    fill: #1e293b !important;
}

/* ─── NUMBER INPUT ─── */
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #0099cc !important;
    box-shadow: 0 0 0 3px rgba(0,153,204,0.12) !important;
}
.stNumberInput button {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #1e293b !important;
}
.stNumberInput button:hover {
    background: #e0f7ff !important;
    color: #0099cc !important;
}

/* ─── TEXT INPUT ─── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0099cc !important;
    box-shadow: 0 0 0 3px rgba(0,153,204,0.12) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
}

/* ─── LABELS ─── */
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label,
.stTextInput label,
.stSlider label,
.stRadio label,
.stCheckbox label,
.stFileUploader label {
    color: #334155 !important;
    font-size: 0.82rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}

/* ─── CHECKBOX ─── */
.stCheckbox > label {
    color: #1e293b !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stCheckbox > label > span {
    color: #1e293b !important;
}
.stCheckbox input[type="checkbox"] + div {
    border-color: #94a3b8 !important;
    background: #ffffff !important;
}
.stCheckbox input[type="checkbox"]:checked + div {
    background: #0099cc !important;
    border-color: #0099cc !important;
}

/* ─── SLIDER ─── */
.stSlider > div > div > div > div {
    background: #0099cc !important;
}
.stSlider > div > div > div {
    background: #cbd5e1 !important;
}
.stSlider > div > div > div > div > div {
    color: #1e293b !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* ─── RADIO BUTTONS ─── */
.stRadio > div {
    display: flex !important;
    gap: 0.75rem !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
}
.stRadio > div > label {
    background: #f8fafc !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: none !important;
    font-size: 0.9rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: normal !important;
    color: #1e293b !important;
    font-weight: 400 !important;
}
.stRadio > div > label:hover {
    background: #e0f7ff !important;
    border-color: #0099cc !important;
    color: #0c4a6e !important;
}
.stRadio > div > label > div > p {
    color: #1e293b !important;
    font-size: 0.9rem !important;
}

/* ─── FILE UPLOADER ─── */
.stFileUploader > div {
    background: #f8fafc !important;
    border: 1.5px dashed #94a3b8 !important;
    border-radius: 12px !important;
}
.stFileUploader > div > div {
    color: #334155 !important;
}
.stFileUploader small, .stFileUploader p {
    color: #475569 !important;
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0099cc !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    color: #1e293b !important;
}

/* ─── EXPANDERS ─── */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderHeader:hover {
    border-color: #0099cc !important;
    color: #0099cc !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
}
.streamlit-expanderHeader svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* ─── DATAFRAME ─── */
.stDataFrame { border-radius: 8px !important; overflow: hidden !important; }
.stDataFrame th {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    font-weight: 700 !important;
}
.stDataFrame td {
    color: #1e293b !important;
}

/* ─── ALERTS ─── */
.stSuccess > div {
    background: #d1fae5 !important;
    border: 1px solid #059669 !important;
    border-radius: 8px !important;
    color: #064e3b !important;
}
.stWarning > div {
    background: #fef3c7 !important;
    border: 1px solid #d97706 !important;
    border-radius: 8px !important;
    color: #78350f !important;
}
.stError > div {
    background: #fee2e2 !important;
    border: 1px solid #dc2626 !important;
    border-radius: 8px !important;
    color: #7f1d1d !important;
}
.stInfo > div {
    background: #e0f7ff !important;
    border: 1px solid #0099cc !important;
    border-radius: 8px !important;
    color: #0c4a6e !important;
}
.stSuccess p, .stSuccess div[data-testid="stMarkdownContainer"] { color: #064e3b !important; }
.stWarning p, .stWarning div[data-testid="stMarkdownContainer"] { color: #78350f !important; }
.stError p, .stError div[data-testid="stMarkdownContainer"] { color: #7f1d1d !important; }
.stInfo p, .stInfo div[data-testid="stMarkdownContainer"] { color: #0c4a6e !important; }

/* ─── SPINNER ─── */
.stSpinner > div { border-top-color: #0099cc !important; }

/* ─── PROGRESS BAR ─── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #0099cc, #7c3aed) !important;
}
.stProgress > div > div { background: #e2e8f0 !important; }

hr { border-color: #e2e8f0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0099cc; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 8px;
}
.badge-cyan { background: #e0f7ff; color: #0c4a6e; border: 1px solid #7dd3f0; }
.badge-green { background: #d1fae5; color: #064e3b; border: 1px solid #6ee7b7; }
.badge-purple { background: #ede9fe; color: #4c1d95; border: 1px solid #c4b5fd; }
.badge-orange { background: #fef3c7; color: #78350f; border: 1px solid #fcd34d; }

.stat-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.stat-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    color: #475569;
    font-family: 'Space Mono', monospace;
}
.stat-pill span { color: #0099cc; font-weight: 700; }

h1, h2, h3 { color: #1e293b !important; }
p, li { color: #475569 !important; }

.step-header-large {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 900;
    color: #1e293b;
    margin: 0;
}
.step-number-badge {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    color: #0099cc;
    opacity: 0.4;
}

.highlight-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0099cc;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.8rem 0;
}
.highlight-box-warn {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #d97706;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.8rem 0;
}

div[data-testid="stMarkdownContainer"] p { color: #334155 !important; }
div[data-testid="stMarkdownContainer"] li { color: #334155 !important; }
div[data-testid="InputInstructions"],
.stCaption, small { color: #64748b !important; font-size: 0.78rem !important; }
div[data-testid="stMetric"] label { color: #475569 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1e293b !important; }
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: #059669 !important; }
</style>
""", unsafe_allow_html=True)


PLOT_COLORS = ['#0099cc', '#7c3aed', '#db2777', '#059669', '#d97706',
               '#2563eb', '#dc2626', '#0d9488', '#7c3aed', '#ea580c']

def plot_layout(title="", height=None):
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f8fafc',
        font=dict(color='#475569', family='DM Sans', size=12),
        title=dict(text=title, font=dict(color='#1e293b', size=14), x=0),
        xaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', zerolinecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', zerolinecolor='#e2e8f0'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#475569')),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if height:
        layout['height'] = height
    return layout


def init_state():
    defaults = {
        'step': 0,
        'problem_type': None,
        'df': None,
        'df_cleaned': None,
        'target': None,
        'selected_features': None,
        'outlier_indices': [],
        'feature_selected_cols': None,
        'X_train': None, 'X_test': None,
        'y_train': None, 'y_test': None,
        'model': None,
        'model_name': None,
        'model_params': {},
        'cv_results': None,
        'metrics': {},
        'best_params': None,
        'k_folds': 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

STEPS = [
    ("⚗️", "Problem\nType"),
    ("📂", "Data\nInput"),
    ("🔭", "EDA"),
    ("🔧", "Data\nEngineering"),
    ("🎯", "Feature\nSelection"),
    ("✂️", "Data\nSplit"),
    ("🤖", "Model\nSelection"),
    ("📊", "Training\n& KFold"),
    ("📈", "Metrics"),
    ("🎛️", "HP\nTuning"),
]


def render_pipeline_bar():
    current = st.session_state.step
    html = '<div class="pipeline-bar">'
    for i, (icon, label) in enumerate(STEPS):
        if i < current:
            status = "done"
            icon_disp = "✓"
        elif i == current:
            status = "active"
            icon_disp = icon
        else:
            status = "pending"
            icon_disp = icon

        html += f'<div class="step-item">'
        html += f'<div class="step-node">'
        html += f'<div class="step-circle {status}">{icon_disp}</div>'
        html += f'<div class="step-label {status}">{label}</div>'
        html += f'</div>'

        if i < len(STEPS) - 1:
            conn_status = "done" if i < current else ("active" if i == current else "pending")
            html += f'<div class="step-connector {conn_status}"></div>'

        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_card(title, icon=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)


def nav_buttons(back=True, next_label="Continue →", next_key=None):
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if back and st.session_state.step > 0:
            if st.button("← Back", key=f"back_{st.session_state.step}"):
                st.session_state.step -= 1
                st.rerun()
    with c3:
        return st.button(next_label, key=next_key or f"next_{st.session_state.step}", type="primary")


def step_problem_type():
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 2rem;">
        <div class="step-number-badge">01</div>
        <div class="step-header-large">Choose Problem Type</div>
        <p style="color: #64748b; margin-top:0.5rem; font-family:'Space Mono',monospace; font-size:0.78rem; letter-spacing:0.1em;">
            SELECT YOUR MACHINE LEARNING TASK
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div style="background:#ffffff; border:2px solid #e2e8f0; border-radius:16px; padding:2rem; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🎯</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.05rem; color:#1e293b; font-weight:700;">Classification</div>
            <p style="color:#64748b; font-size:0.82rem; margin-top:0.5rem; line-height:1.6;">
                Predict discrete categories or labels.<br>Binary, multi-class, or multi-label tasks.
            </p>
            <div style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span class="badge badge-cyan">Logistic</span>
                <span class="badge badge-purple">SVM</span>
                <span class="badge badge-green">Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Select Classification", key="cls_btn", use_container_width=True):
            st.session_state.problem_type = "Classification"
            st.session_state.step = 1
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#ffffff; border:2px solid #e2e8f0; border-radius:16px; padding:2rem; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📈</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.05rem; color:#1e293b; font-weight:700;">Regression</div>
            <p style="color:#64748b; font-size:0.82rem; margin-top:0.5rem; line-height:1.6;">
                Predict continuous numerical values.<br>Linear, polynomial, or non-linear relationships.
            </p>
            <div style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span class="badge badge-orange">Linear</span>
                <span class="badge badge-purple">SVR</span>
                <span class="badge badge-cyan">Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Select Regression", key="reg_btn", use_container_width=True):
            st.session_state.problem_type = "Regression"
            st.session_state.step = 1
            st.rerun()


def step_data_input():
    pt_badge = 'badge-cyan' if st.session_state.problem_type == 'Classification' else 'badge-orange'
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">02</div>
        <div>
            <div class="step-header-large">Data Input</div>
            <span class="badge {pt_badge}">{st.session_state.problem_type}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁  Upload CSV", "🧪  Sample Dataset"])

    with tab1:
        uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown('<p style="font-size:0.85rem; margin-bottom:1rem; color:#334155;">Load a built-in dataset to explore the pipeline:</p>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🌸 Iris (Classification)", use_container_width=True):
                from sklearn.datasets import load_iris
                d = load_iris(as_frame=True)
                st.session_state.df = pd.concat([d.data, d.target.rename("target")], axis=1)
                st.session_state.problem_type = "Classification"
                st.rerun()
        with col_b:
            if st.button("🏠 California Housing (Regression)", use_container_width=True):
                from sklearn.datasets import fetch_california_housing
                d = fetch_california_housing(as_frame=True)
                st.session_state.df = pd.concat([d.data, d.target.rename("target")], axis=1)
                st.session_state.problem_type = "Regression"
                st.rerun()
        with col_c:
            if st.button("🍷 Wine (Classification)", use_container_width=True):
                from sklearn.datasets import load_wine
                d = load_wine(as_frame=True)
                st.session_state.df = pd.concat([d.data, d.target.rename("target")], axis=1)
                st.session_state.problem_type = "Classification"
                st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown('<hr style="margin:1rem 0;">', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        missing = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">Rows: <span>{df.shape[0]:,}</span></div>
            <div class="stat-pill">Columns: <span>{df.shape[1]}</span></div>
            <div class="stat-pill">Numeric: <span>{len(numeric_cols)}</span></div>
            <div class="stat-pill">Categorical: <span>{len(cat_cols)}</span></div>
            <div class="stat-pill">Missing: <span>{missing}</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Preview Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        section_card("Select Target Feature", "🎯")
        target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1)
        st.session_state.target = target

        feature_cols = [c for c in df.columns if c != target]
        section_card("Select Input Features", "📌")
        sel_features = st.multiselect("Features to include", feature_cols, default=feature_cols)
        st.session_state.selected_features = sel_features

        if sel_features:
            section_card("Data Shape — PCA Projection", "🔮")
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            sub = df[sel_features + [target]].dropna()
            num_feats = [c for c in sel_features if c in numeric_cols]

            if len(num_feats) >= 2:
                Xs = StandardScaler().fit_transform(sub[num_feats])
                n_comp = min(3, len(num_feats))
                pca = PCA(n_components=n_comp)
                pcs = pca.fit_transform(Xs)
                var = pca.explained_variance_ratio_

                pca_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_comp)])
                pca_df["target"] = sub[target].astype(str).values

                c1, c2 = st.columns([2, 1])
                with c1:
                    if n_comp >= 3:
                        fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3",
                                            color="target", opacity=0.8,
                                            title="PCA — 3D Projection",
                                            color_discrete_sequence=PLOT_COLORS)
                    else:
                        fig = px.scatter(pca_df, x="PC1", y="PC2", color="target",
                                         title="PCA — 2D Projection", opacity=0.8,
                                         color_discrete_sequence=PLOT_COLORS)
                    fig.update_layout(**plot_layout("PCA Projection"))
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig2 = go.Figure(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(var))],
                        y=var * 100,
                        marker=dict(color=PLOT_COLORS[:len(var)]),
                        text=[f"{v:.1f}%" for v in var * 100],
                        textposition='outside',
                        textfont=dict(color='#1e293b', size=11)
                    ))
                    fig2.update_layout(**plot_layout("Explained Variance"))
                    fig2.update_yaxes(title="Variance %")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Select at least 2 numeric features for PCA visualization.")

        if nav_buttons(back=True, next_label="Proceed to EDA →", next_key="data_next"):
            if sel_features and target:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Please select target and at least one feature.")


def step_eda():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">03</div>
        <div class="step-header-large">Exploratory Data Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target
    feats = st.session_state.selected_features
    cols = feats + [target]
    sub = df[cols].copy()
    num_cols = sub.select_dtypes(include=np.number).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distribution", "🔗 Correlation", "📦 Boxplots", "🎯 Target Analysis", "📋 Summary Stats"
    ])

    with tab1:
        section_card("Feature Distributions", "📊")
        if num_cols:
            selected_col = st.selectbox("Select feature", num_cols)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sub, x=selected_col, nbins=40,
                                   color_discrete_sequence=['#0099cc'],
                                   title=f"Distribution: {selected_col}")
                fig.update_layout(**plot_layout(f"Distribution: {selected_col}"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Violin(y=sub[selected_col].dropna(), box_visible=True,
                                         line_color='#7c3aed', fillcolor='rgba(124,58,237,0.15)',
                                         name=selected_col, meanline_visible=True))
                fig2.update_layout(**plot_layout(f"Violin: {selected_col}"))
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_card("Correlation Matrix", "🔗")
        if len(num_cols) > 1:
            corr = sub[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Pearson Correlation Heatmap")
            fig.update_layout(**plot_layout("Pearson Correlation Heatmap"))
            st.plotly_chart(fig, use_container_width=True)

            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    v = corr.iloc[i, j]
                    if abs(v) > 0.7:
                        high_corr.append((corr.columns[i], corr.columns[j], round(v, 3)))
            if high_corr:
                st.warning(f"⚠️ {len(high_corr)} highly correlated pair(s) found (|r| > 0.7):")
                hc_df = pd.DataFrame(high_corr, columns=["Feature A", "Feature B", "Correlation"])
                st.dataframe(hc_df, use_container_width=True, hide_index=True)

    with tab3:
        section_card("Outlier Visualization — Boxplots", "📦")
        if num_cols:
            fig = go.Figure()
            for i, col in enumerate(num_cols):
                fig.add_trace(go.Box(
                    y=sub[col].dropna(), name=col,
                    marker_color=PLOT_COLORS[i % len(PLOT_COLORS)],
                    boxpoints='outliers', jitter=0.3,
                ))
            fig.update_layout(**plot_layout("Feature Boxplots"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section_card("Target Variable Analysis", "🎯")
        if target in num_cols:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sub, x=target, nbins=40,
                                   color_discrete_sequence=['#db2777'],
                                   title=f"Target Distribution: {target}")
                fig.update_layout(**plot_layout(f"Target Distribution: {target}"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                from scipy import stats as sp_stats
                sorted_vals = np.sort(sub[target].dropna())
                theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_vals)))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=theoretical, y=sorted_vals, mode='markers',
                                          marker=dict(color='#059669', size=4, opacity=0.7),
                                          name='Data'))
                fig2.add_trace(go.Scatter(x=[theoretical[0], theoretical[-1]],
                                          y=[theoretical[0], theoretical[-1]],
                                          mode='lines', line=dict(color='#db2777', dash='dash'),
                                          name='Normal'))
                fig2.update_layout(**plot_layout("Q-Q Plot"))
                fig2.update_xaxes(title="Theoretical Quantiles")
                fig2.update_yaxes(title="Sample Quantiles")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            vc = sub[target].value_counts()
            fig = px.bar(x=vc.index.astype(str), y=vc.values,
                         labels={'x': target, 'y': 'Count'},
                         title="Class Distribution",
                         color=vc.values,
                         color_continuous_scale=['#7c3aed', '#0099cc'])
            fig.update_layout(**plot_layout("Class Distribution"))
            st.plotly_chart(fig, use_container_width=True)

            if vc.max() / vc.min() > 3:
                st.warning("⚠️ Significant class imbalance detected.")

    with tab5:
        section_card("Descriptive Statistics", "📋")
        st.dataframe(sub[num_cols].describe().round(4), use_container_width=True)
        miss = sub.isnull().sum()
        miss = miss[miss > 0]
        if len(miss) > 0:
            st.warning(f"Missing values found in {len(miss)} column(s):")
            fig = px.bar(x=miss.index, y=miss.values,
                         labels={'x': 'Column', 'y': 'Missing Count'},
                         color=miss.values,
                         color_continuous_scale=['#d97706', '#db2777'])
            fig.update_layout(**plot_layout("Missing Values"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✓ No missing values found!")

    if nav_buttons(next_label="Data Engineering →", next_key="eda_next"):
        st.session_state.step = 3
        st.rerun()


def step_data_engineering():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">04</div>
        <div class="step-header-large">Data Engineering & Cleaning</div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df.copy()
    feats = st.session_state.selected_features
    target = st.session_state.target
    num_cols = df[feats].select_dtypes(include=np.number).columns.tolist()
    all_cols = feats + [target]
    sub = df[all_cols].copy()

    section_card("Missing Value Imputation", "🔧")
    miss_cols = [c for c in num_cols if sub[c].isnull().sum() > 0]
    if miss_cols:
        method = st.selectbox("Imputation method", ["Mean", "Median", "Mode", "Drop Rows"])
        cols_to_impute = st.multiselect("Columns to impute", miss_cols, default=miss_cols)
        if st.button("Apply Imputation", key="impute_btn"):
            for c in cols_to_impute:
                if method == "Mean":
                    sub[c].fillna(sub[c].mean(), inplace=True)
                elif method == "Median":
                    sub[c].fillna(sub[c].median(), inplace=True)
                elif method == "Mode":
                    sub[c].fillna(sub[c].mode()[0], inplace=True)
                elif method == "Drop Rows":
                    sub.dropna(subset=cols_to_impute, inplace=True)
            st.success("✓ Imputation applied!")
    else:
        st.success("✓ No missing values in selected features.")

    st.markdown('<hr style="margin:1rem 0;">', unsafe_allow_html=True)

    section_card("Outlier Detection", "🔍")
    method_out = st.selectbox("Detection method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])

    outlier_indices = []
    if st.button("Detect Outliers", key="detect_out"):
        X_out = sub[num_cols].dropna()

        if method_out == "IQR":
            mask = pd.Series([False] * len(X_out), index=X_out.index)
            for c in num_cols:
                Q1, Q3 = X_out[c].quantile(0.25), X_out[c].quantile(0.75)
                IQR = Q3 - Q1
                mask |= (X_out[c] < Q1 - 1.5*IQR) | (X_out[c] > Q3 + 1.5*IQR)
            outlier_indices = X_out[mask].index.tolist()
        elif method_out == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X_out)
            preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(Xs)
            outlier_indices = X_out[preds == -1].index.tolist()
        elif method_out == "DBSCAN":
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X_out)
            preds = DBSCAN(eps=0.5, min_samples=5).fit_predict(Xs)
            outlier_indices = X_out[preds == -1].index.tolist()
        elif method_out == "OPTICS":
            from sklearn.cluster import OPTICS
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X_out)
            preds = OPTICS(min_samples=5).fit_predict(Xs)
            outlier_indices = X_out[preds == -1].index.tolist()

        st.session_state.outlier_indices = outlier_indices

    if st.session_state.outlier_indices:
        n_out = len(st.session_state.outlier_indices)
        pct = 100 * n_out / len(sub)
        st.markdown(f"""
        <div class="highlight-box-warn">
            <div style="font-weight:700; color:#92400e; font-size:0.9rem;">⚠️ Outliers Detected</div>
            <div style="color:#78350f; margin-top:0.3rem; font-size:0.85rem;">{n_out} rows ({pct:.1f}% of data) identified as outliers</div>
        </div>
        """, unsafe_allow_html=True)

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X_vis = sub[num_cols].fillna(sub[num_cols].median())
        Xs = StandardScaler().fit_transform(X_vis)
        pcs = PCA(n_components=2).fit_transform(Xs)
        vis_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=sub.index)
        vis_df["type"] = "Normal"
        vis_df.loc[st.session_state.outlier_indices, "type"] = "Outlier"

        fig = px.scatter(vis_df, x="PC1", y="PC2", color="type",
                         color_discrete_map={"Normal": "#0099cc", "Outlier": "#db2777"},
                         title="Outlier Visualization (PCA Space)",
                         opacity=0.7)
        fig.update_layout(**plot_layout("Outlier Visualization (PCA Space)"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Remove Outliers", key="remove_out", type="primary"):
                sub = sub.drop(index=st.session_state.outlier_indices, errors='ignore')
                st.session_state.df_cleaned = sub
                st.session_state.outlier_indices = []
                st.success(f"✓ Removed {n_out} outlier rows. New shape: {sub.shape}")
                st.rerun()
        with col2:
            if st.button("Keep Outliers", key="keep_out"):
                st.session_state.df_cleaned = sub
                st.session_state.outlier_indices = []
                st.info("Outliers retained in dataset.")
                st.rerun()

        with st.expander(f"View Outlier Rows ({n_out})"):
            st.dataframe(sub.loc[st.session_state.outlier_indices], use_container_width=True)
    else:
        if st.session_state.df_cleaned is None:
            st.session_state.df_cleaned = sub

    if nav_buttons(next_label="Feature Selection →", next_key="eng_next"):
        if st.session_state.df_cleaned is None:
            st.session_state.df_cleaned = sub
        st.session_state.step = 4
        st.rerun()


def step_feature_selection():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">05</div>
        <div class="step-header-large">Feature Selection</div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    feats = st.session_state.selected_features
    target = st.session_state.target
    num_feats = [c for c in feats if c in df.select_dtypes(include=np.number).columns]
    sub = df[num_feats + [target]].dropna()

    from sklearn.preprocessing import LabelEncoder
    y = sub[target].copy()
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.values
    X = sub[num_feats]

    method = st.selectbox("Selection Method", [
        "Variance Threshold", "Correlation with Target", "Information Gain (Mutual Info)"
    ])

    results = {}

    if method == "Variance Threshold":
        from sklearn.feature_selection import VarianceThreshold
        threshold = st.slider("Variance Threshold", 0.0, 2.0, 0.1, 0.05)
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(X)
        variances = pd.Series(sel.variances_, index=num_feats).sort_values(ascending=False)
        results = variances.to_dict()
        selected = variances[variances >= threshold].index.tolist()

    elif method == "Correlation with Target":
        threshold = st.slider("Min |Correlation|", 0.0, 1.0, 0.1, 0.05)
        corrs = pd.Series({c: abs(X[c].corr(pd.Series(y))) for c in num_feats}).sort_values(ascending=False)
        results = corrs.to_dict()
        selected = corrs[corrs >= threshold].index.tolist()

    elif method == "Information Gain (Mutual Info)":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        threshold = st.slider("Min Information Gain", 0.0, 1.0, 0.05, 0.01)
        fn = mutual_info_classif if st.session_state.problem_type == "Classification" else mutual_info_regression
        scores = fn(X, y, random_state=42)
        mi = pd.Series(scores, index=num_feats).sort_values(ascending=False)
        results = mi.to_dict()
        selected = mi[mi >= threshold].index.tolist()

    if results:
        col1, col2 = st.columns([2, 1])
        with col1:
            res_series = pd.Series(results).sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                y=res_series.index, x=res_series.values, orientation='h',
                marker=dict(color=['#059669' if n in selected else '#cbd5e1' for n in res_series.index]),
                text=[f"{v:.4f}" for v in res_series.values],
                textposition='outside',
                textfont=dict(color='#1e293b', size=10)
            ))
            fig.update_layout(**plot_layout(f"Feature Scores — {method}"),
                              height=max(300, len(results) * 35))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="section-card" style="margin-top:0; text-align:center;">
                <div class="section-title" style="justify-content:center;">📌 Selected Features</div>
                <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#059669; font-weight:900;">{len(selected)}</div>
                <div style="color:#64748b; font-size:0.75rem; font-family:'Space Mono',monospace;">OF {len(num_feats)} FEATURES</div>
                <hr style="margin:0.8rem 0; border-color:#e2e8f0;">
            """, unsafe_allow_html=True)
            for f in selected:
                st.markdown(f'<div style="color:#1e293b; font-size:0.82rem; padding:3px 0; text-align:left;">✓ {f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        final_sel = st.multiselect("Confirm/modify selected features", num_feats, default=selected)
        st.session_state.feature_selected_cols = final_sel if final_sel else selected

    if nav_buttons(next_label="Data Split →", next_key="feat_next"):
        if not st.session_state.feature_selected_cols:
            st.session_state.feature_selected_cols = num_feats
        st.session_state.step = 5
        st.rerun()


def step_data_split():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">06</div>
        <div class="step-header-large">Train / Test Split</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    target = st.session_state.target
    feats = st.session_state.feature_selected_cols or st.session_state.selected_features
    feats = [f for f in feats if f in df.columns]

    sub = df[feats + [target]].dropna()

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random Seed", 0, 999, 42)
    with col3:
        stratify_opt = st.checkbox("Stratify Split",
                                    value=(st.session_state.problem_type == "Classification"))

    if st.button("Apply Split", key="split_btn", type="primary"):
        X = sub[feats]
        y = sub[target]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y)

        strat = y if stratify_opt and st.session_state.problem_type == "Classification" else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                        random_state=int(random_state),
                                                        stratify=strat)
        except:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                        random_state=int(random_state))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        st.session_state.X_train = X_tr_s
        st.session_state.X_test = X_te_s
        st.session_state.y_train = y_tr
        st.session_state.y_test = y_te

        train_n, test_n = len(X_tr), len(X_te)
        total = train_n + test_n

        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.markdown(f'<div class="metric-card"><span class="metric-value">{train_n:,}</span><span class="metric-label">Train Samples</span></div>', unsafe_allow_html=True)
        col_v2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#7c3aed">{test_n:,}</span><span class="metric-label">Test Samples</span></div>', unsafe_allow_html=True)
        col_v3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#059669">{len(feats)}</span><span class="metric-label">Features</span></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Pie(
            values=[train_n, test_n],
            labels=[f'Train ({100*(1-test_size):.0f}%)', f'Test ({100*test_size:.0f}%)'],
            marker_colors=['#0099cc', '#7c3aed'],
            hole=0.6,
            textinfo='label+percent',
            textfont=dict(color='#1e293b', size=12)
        ))
        fig.update_layout(**plot_layout("Data Split Distribution"),
                          showlegend=False, height=300,
                          annotations=[dict(text=f'{total:,}<br>Total', x=0.5, y=0.5,
                                            font=dict(size=14, color='#1e293b'), showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
        st.success("✓ Data split complete. Features standardized with StandardScaler.")

    if nav_buttons(next_label="Model Selection →", next_key="split_next"):
        if st.session_state.X_train is None:
            st.error("Please apply the split first.")
        else:
            st.session_state.step = 6
            st.rerun()


def step_model_selection():
    pt_badge = 'badge-cyan' if st.session_state.problem_type == 'Classification' else 'badge-orange'
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">07</div>
        <div class="step-header-large">Model Selection</div>
        <span class="badge {pt_badge}">{st.session_state.problem_type}</span>
    </div>
    """, unsafe_allow_html=True)

    is_cls = st.session_state.problem_type == "Classification"

    if is_cls:
        models_info = {
            "Logistic Regression": {"icon": "📐", "desc": "Linear decision boundary, fast and interpretable", "color": "#0099cc"},
            "SVM (Support Vector Machine)": {"icon": "⚔️", "desc": "Kernel-based, powerful for non-linear boundaries", "color": "#7c3aed"},
            "Random Forest": {"icon": "🌲", "desc": "Ensemble of decision trees, robust to overfitting", "color": "#059669"},
            "K-Nearest Neighbors": {"icon": "🔵", "desc": "Instance-based, no training phase", "color": "#d97706"},
        }
    else:
        models_info = {
            "Linear Regression": {"icon": "📏", "desc": "Simple linear relationship modeling", "color": "#0099cc"},
            "SVM Regression (SVR)": {"icon": "⚔️", "desc": "Kernel-based regression with epsilon tube", "color": "#7c3aed"},
            "Random Forest Regressor": {"icon": "🌲", "desc": "Ensemble method, handles non-linearity well", "color": "#059669"},
            "K-Nearest Neighbors Regressor": {"icon": "🔵", "desc": "Predict based on nearest neighbors", "color": "#d97706"},
        }

    selected_model = st.session_state.get('model_name', list(models_info.keys())[0])

    cols = st.columns(len(models_info))
    for i, (name, info) in enumerate(models_info.items()):
        with cols[i]:
            is_sel = (name == selected_model)
            border = info['color'] if is_sel else '#e2e8f0'
            bg = '#f0f9ff' if is_sel else '#ffffff'
            bw = '2px' if is_sel else '1px'
            st.markdown(f"""
            <div style="background:{bg}; border:{bw} solid {border}; border-radius:12px; padding:1.2rem; text-align:center; min-height:140px;">
                <div style="font-size:1.8rem;">{info['icon']}</div>
                <div style="font-size:0.78rem; color:#1e293b; font-weight:700; margin-top:0.5rem; font-family:'Space Mono',monospace;">{name}</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:0.4rem; line-height:1.4;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
            if st.button(f"Select", key=f"sel_{i}", use_container_width=True):
                st.session_state.model_name = name
                st.rerun()

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    section_card("Model Configuration", "⚙️")

    model_name = st.session_state.model_name or list(models_info.keys())[0]
    params = {}

    if "SVM" in model_name:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        C = st.slider("C (Regularization)", 0.01, 100.0, 1.0, 0.1)
        params = {"kernel": kernel, "C": C}
        if kernel == "poly":
            params["degree"] = st.slider("Degree", 2, 6, 3)

    elif "Random Forest" in model_name:
        n_est = st.slider("Number of Trees", 10, 500, 100, 10)
        max_depth = st.select_slider("Max Depth", options=[None, 3, 5, 7, 10, 15, 20], value=None)
        params = {"n_estimators": n_est, "max_depth": max_depth}

    elif "Linear" in model_name or "Logistic" in model_name:
        if is_cls:
            C = st.slider("C (Inverse Regularization)", 0.01, 100.0, 1.0, 0.1)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            params = {"C": C, "solver": solver, "max_iter": 1000}
        else:
            fit_intercept = st.checkbox("Fit Intercept", True)
            params = {"fit_intercept": fit_intercept}

    elif "K-Nearest" in model_name:
        k = st.slider("K (Neighbors)", 1, 20, 5)
        metric = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"])
        params = {"n_neighbors": k, "metric": metric}

    st.session_state.model_params = params
    st.session_state.model_name = model_name

    if nav_buttons(next_label="Train & KFold →", next_key="model_next"):
        st.session_state.step = 7
        st.rerun()


def step_training():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">08</div>
        <div class="step-header-large">Model Training & K-Fold CV</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    k = st.number_input("K (Number of Folds)", min_value=2, max_value=20, value=5)
    st.session_state.k_folds = k

    is_cls = st.session_state.problem_type == "Classification"
    model_name = st.session_state.model_name
    params = st.session_state.model_params

    def build_model():
        if is_cls:
            if "Logistic" in model_name:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**params, random_state=42)
            elif "SVM" in model_name:
                from sklearn.svm import SVC
                return SVC(**params, probability=True, random_state=42)
            elif "Random Forest" in model_name:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**params, random_state=42)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(**params)
        else:
            if "Linear" in model_name:
                from sklearn.linear_model import LinearRegression
                return LinearRegression(**params)
            elif "SVR" in model_name or "SVM" in model_name:
                from sklearn.svm import SVR
                return SVR(**params)
            elif "Random Forest" in model_name:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params, random_state=42)
            else:
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(**params)

    if st.button("🚀 Start Training", key="train_btn", type="primary"):
        X_tr = st.session_state.X_train
        y_tr = st.session_state.y_train
        X_te = st.session_state.X_test
        y_te = st.session_state.y_test

        with st.spinner("Training model..."):
            model = build_model()
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) if is_cls else KFold(n_splits=k, shuffle=True, random_state=42)
            scoring = 'accuracy' if is_cls else 'r2'
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring)
            model.fit(X_tr, y_tr)

            st.session_state.model = model
            st.session_state.cv_results = cv_scores

        section_card(f"{k}-Fold Cross-Validation Results", "🔄")
        fold_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(k)],
            "Score": cv_scores
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fold_df["Fold"], y=fold_df["Score"],
            marker=dict(
                color=cv_scores,
                colorscale=[[0, '#db2777'], [0.5, '#7c3aed'], [1, '#0099cc']],
                showscale=False
            ),
            text=[f"{s:.4f}" for s in cv_scores],
            textposition='outside',
            textfont=dict(color='#1e293b')
        ))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#059669",
                      annotation_text=f"Mean: {cv_scores.mean():.4f}",
                      annotation_font_color="#059669")
        fig.update_layout(**plot_layout(f"CV {scoring.upper()} per Fold"))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{cv_scores.mean():.4f}</span><span class="metric-label">Mean CV Score</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#7c3aed">{cv_scores.std():.4f}</span><span class="metric-label">Std Deviation</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#059669">{cv_scores.max():.4f}</span><span class="metric-label">Best Fold</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#db2777">{cv_scores.min():.4f}</span><span class="metric-label">Worst Fold</span></div>', unsafe_allow_html=True)

        st.success(f"✓ Model trained! Mean CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if nav_buttons(next_label="View Metrics →", next_key="train_next"):
        if st.session_state.model is None:
            st.error("Please train the model first.")
        else:
            st.session_state.step = 8
            st.rerun()


# ─────────────────────────────────────────────────────────────────
#  STEP 9 — METRICS  (fully rewritten with precision/recall +
#            model-specific visualizations)
# ─────────────────────────────────────────────────────────────────
def step_metrics():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">09</div>
        <div class="step-header-large">Performance Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn import metrics as skm
    is_cls = st.session_state.problem_type == "Classification"
    model      = st.session_state.model
    model_name = st.session_state.model_name or ""
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test

    if model is None:
        st.error("No trained model found. Please go back and train a model.")
        return

    y_pred    = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)

    # ── helper: determine if binary or multiclass ──────────────────
    classes = np.unique(y_te)
    is_binary = len(classes) == 2

    # ══════════════════════════════════════════════════════════════
    #  CLASSIFICATION
    # ══════════════════════════════════════════════════════════════
    if is_cls:
        train_acc = skm.accuracy_score(y_tr, y_pred_tr)
        test_acc  = skm.accuracy_score(y_te, y_pred)
        train_f1  = skm.f1_score(y_tr, y_pred_tr, average='weighted')
        test_f1   = skm.f1_score(y_te, y_pred,    average='weighted')
        precision = skm.precision_score(y_te, y_pred, average='weighted', zero_division=0)
        recall    = skm.recall_score(y_te, y_pred,    average='weighted', zero_division=0)

        # ── top metric row ─────────────────────────────────────────
        diff = train_acc - test_acc
        if diff > 0.1:
            fit_label, fit_color, fit_msg = "OVERFITTING", "#dc2626", f"Train–Test gap: {diff:.3f}"
        elif test_acc < 0.6:
            fit_label, fit_color, fit_msg = "UNDERFITTING", "#d97706", "Low test accuracy"
        else:
            fit_label, fit_color, fit_msg = "GOOD FIT", "#059669", "Model generalises well"

        cols = st.columns(6)
        metrics_top = [
            ("Test Accuracy",  f"{test_acc:.4f}",  "#0099cc"),
            ("Precision",      f"{precision:.4f}", "#7c3aed"),
            ("Recall",         f"{recall:.4f}",    "#db2777"),
            ("F1 Score",       f"{test_f1:.4f}",   "#d97706"),
            ("Train Accuracy", f"{train_acc:.4f}", "#059669"),
            (fit_msg,          fit_label,           fit_color),
        ]
        for col, (lbl, val, clr) in zip(cols, metrics_top):
            col.markdown(
                f'<div class="metric-card"><span class="metric-value" style="color:{clr}; font-size:1.1rem;">{val}</span>'
                f'<span class="metric-label">{lbl}</span></div>',
                unsafe_allow_html=True
            )

        # ── tabs ───────────────────────────────────────────────────
        tab_labels = ["🗺️ Confusion Matrix", "📊 Class Report",
                      "🔄 Train vs Test", "🎯 Model-Specific"]
        # add ROC tab only for binary problems that support predict_proba
        has_proba = hasattr(model, "predict_proba")
        if is_binary and has_proba:
            tab_labels.append("📉 ROC / PR Curve")

        tabs = st.tabs(tab_labels)

        # ── Confusion Matrix ───────────────────────────────────────
        with tabs[0]:
            cm     = skm.confusion_matrix(y_te, y_pred)
            labels = [str(c) for c in sorted(classes)]
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            labels=dict(x="Predicted", y="Actual"),
                            color_continuous_scale=[[0,'#f8fafc'],[0.5,'#7c3aed'],[1,'#0099cc']],
                            x=labels, y=labels)
            fig.update_layout(**plot_layout("Confusion Matrix"))
            st.plotly_chart(fig, use_container_width=True)

            # per-class breakdown
            st.markdown("#### Per-Class Metrics")
            per_class = skm.classification_report(y_te, y_pred, output_dict=True, zero_division=0)
            pc_df = pd.DataFrame(per_class).T
            numeric_pc = pc_df.select_dtypes(include=np.number)
            pc_df[numeric_pc.columns] = numeric_pc.round(4)
            st.dataframe(pc_df, use_container_width=True)

        # ── Classification Report ──────────────────────────────────
        with tabs[1]:
            report = skm.classification_report(y_te, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)

            # per-class precision / recall / f1 bar chart
            class_rows = [k for k in report if k not in ['accuracy','macro avg','weighted avg']]
            prec_vals = [report[k]['precision'] for k in class_rows]
            rec_vals  = [report[k]['recall']    for k in class_rows]
            f1_vals   = [report[k]['f1-score']  for k in class_rows]

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(name='Precision', x=class_rows, y=prec_vals,
                                    marker_color='#0099cc', opacity=0.85))
            fig_pr.add_trace(go.Bar(name='Recall',    x=class_rows, y=rec_vals,
                                    marker_color='#db2777', opacity=0.85))
            fig_pr.add_trace(go.Bar(name='F1 Score',  x=class_rows, y=f1_vals,
                                    marker_color='#7c3aed', opacity=0.85))
            fig_pr.update_layout(**plot_layout("Precision / Recall / F1 per Class"),
                                 barmode='group')
            fig_pr.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Train vs Test ──────────────────────────────────────────
        with tabs[2]:
            fig_tvt = go.Figure()
            cats = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_vals = [
                train_acc,
                skm.precision_score(y_tr, y_pred_tr, average='weighted', zero_division=0),
                skm.recall_score(y_tr, y_pred_tr, average='weighted', zero_division=0),
                train_f1,
            ]
            test_vals  = [test_acc, precision, recall, test_f1]
            fig_tvt.add_trace(go.Bar(name='Train', x=cats, y=train_vals,
                                     marker_color='#0099cc', opacity=0.85))
            fig_tvt.add_trace(go.Bar(name='Test',  x=cats, y=test_vals,
                                     marker_color='#7c3aed', opacity=0.85))
            fig_tvt.update_layout(**plot_layout("Train vs Test — All Metrics"), barmode='group')
            fig_tvt.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_tvt, use_container_width=True)

            # ── radar chart ────────────────────────────────────────
            radar_cats = ['Accuracy', 'Precision', 'Recall', 'F1']
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=train_vals + [train_vals[0]], theta=radar_cats + [radar_cats[0]],
                fill='toself', name='Train',
                line_color='#0099cc', fillcolor='rgba(0,153,204,0.15)'
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=test_vals + [test_vals[0]], theta=radar_cats + [radar_cats[0]],
                fill='toself', name='Test',
                line_color='#7c3aed', fillcolor='rgba(124,58,237,0.15)'
            ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                           gridcolor='#e2e8f0',
                                           tickfont=dict(color='#475569'))),
                **plot_layout("Performance Radar"),
                showlegend=True, height=400
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Model-Specific Visualizations ─────────────────────────
        with tabs[3]:
            st.markdown(f"#### Model: `{model_name}`")

            # ── Logistic Regression: coefficients ─────────────────
            if "Logistic" in model_name:
                section_card("Decision Coefficients", "📐")
                feats = st.session_state.feature_selected_cols or st.session_state.selected_features
                coef = model.coef_
                if coef.shape[0] == 1:                  # binary
                    coef_df = pd.DataFrame({"Feature": feats, "Coefficient": coef[0]})
                    coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
                    fig_coef = go.Figure(go.Bar(
                        y=coef_df["Feature"], x=coef_df["Coefficient"], orientation='h',
                        marker=dict(color=['#db2777' if v < 0 else '#059669' for v in coef_df["Coefficient"]]),
                        text=[f"{v:.4f}" for v in coef_df["Coefficient"]],
                        textposition='outside', textfont=dict(color='#1e293b', size=10)
                    ))
                    fig_coef.update_layout(**plot_layout("Feature Coefficients"),
                                           height=max(300, len(feats)*30))
                    st.plotly_chart(fig_coef, use_container_width=True)
                else:                                   # multiclass — heatmap
                    coef_df = pd.DataFrame(coef, columns=feats,
                                           index=[f"Class {c}" for c in classes])
                    fig_coef = px.imshow(coef_df, text_auto=".2f", aspect="auto",
                                         color_continuous_scale="RdBu_r",
                                         title="Coefficient Heatmap (class × feature)")
                    fig_coef.update_layout(**plot_layout())
                    st.plotly_chart(fig_coef, use_container_width=True)

            # ── SVM: support vectors + decision scores ─────────────
            elif "SVM" in model_name:
                section_card("Decision Function Scores", "⚔️")
                from sklearn.decomposition import PCA
                pca2 = PCA(n_components=2)
                X_2d = pca2.fit_transform(X_te)
                df_sv = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
                df_sv["True Label"] = y_te.astype(str)
                df_sv["Predicted"]  = y_pred.astype(str)
                df_sv["Correct"]    = (y_te == y_pred)
                fig_sv = px.scatter(df_sv, x="PC1", y="PC2",
                                    color="True Label",
                                    symbol="Correct",
                                    symbol_map={True: "circle", False: "x"},
                                    title="SVM — PCA Test-Set Projection (✗ = misclassified)",
                                    color_discrete_sequence=PLOT_COLORS,
                                    opacity=0.8)
                fig_sv.update_layout(**plot_layout())
                st.plotly_chart(fig_sv, use_container_width=True)
                st.info(f"Support vectors: **{model.n_support_.sum()}** total across {len(classes)} class(es).")

            # ── Random Forest: feature importances ─────────────────
            elif "Random Forest" in model_name:
                section_card("Feature Importances (Gini / MDI)", "🌲")
                feats = st.session_state.feature_selected_cols or st.session_state.selected_features
                imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True)
                fig_imp = go.Figure(go.Bar(
                    y=imp.index, x=imp.values, orientation='h',
                    marker=dict(
                        color=imp.values,
                        colorscale=[[0,'#e2e8f0'],[0.6,'#7c3aed'],[1,'#059669']],
                        showscale=False
                    ),
                    text=[f"{v:.4f}" for v in imp.values],
                    textposition='outside',
                    textfont=dict(color='#1e293b', size=10)
                ))
                fig_imp.update_layout(**plot_layout("Feature Importances"),
                                      height=max(300, len(feats)*32))
                st.plotly_chart(fig_imp, use_container_width=True)

                # single tree visualisation (depth-limited)
                with st.expander("🌳 View a Single Decision Tree (depth ≤ 3)"):
                    try:
                        from sklearn.tree import export_text
                        tree_ = model.estimators_[0]
                        tree_text = export_text(tree_,
                                                feature_names=list(feats),
                                                max_depth=3)
                        st.code(tree_text, language="text")
                    except Exception as e:
                        st.warning(f"Could not render tree: {e}")

            # ── KNN: neighbour distance distribution ───────────────
            elif "K-Nearest" in model_name:
                section_card("Neighbour Distance Distribution", "🔵")
                dists, _ = model.kneighbors(X_te)
                mean_dists = dists.mean(axis=1)
                fig_knn = go.Figure()
                fig_knn.add_trace(go.Histogram(x=mean_dists, nbinsx=30,
                                               marker_color='#0099cc', opacity=0.8,
                                               name='Mean K-Distance'))
                fig_knn.update_layout(**plot_layout(f"Mean Distance to {model.n_neighbors} Neighbours"))
                fig_knn.update_xaxes(title="Mean Distance")
                fig_knn.update_yaxes(title="Count")
                st.plotly_chart(fig_knn, use_container_width=True)

                # prediction confidence: fraction of k neighbours agreeing
                _, idx = model.kneighbors(X_te)
                confidence = []
                for i, nbrs in enumerate(idx):
                    nbr_labels = y_tr[nbrs] if hasattr(y_tr, '__getitem__') else np.array(y_tr)[nbrs]
                    conf = (nbr_labels == y_pred[i]).mean()
                    confidence.append(conf)
                fig_conf = px.histogram(x=confidence, nbins=20,
                                        color_discrete_sequence=['#7c3aed'],
                                        title="Prediction Confidence (neighbour agreement)")
                fig_conf.update_layout(**plot_layout())
                fig_conf.update_xaxes(title="Confidence", range=[0, 1])
                st.plotly_chart(fig_conf, use_container_width=True)

        # ── ROC / PR Curve (binary only) ───────────────────────────
        if is_binary and has_proba and len(tabs) > 4:
            with tabs[4]:
                y_prob = model.predict_proba(X_te)[:, 1]

                fpr, tpr, _ = skm.roc_curve(y_te, y_prob)
                auc_val = skm.roc_auc_score(y_te, y_prob)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    line=dict(color='#0099cc', width=2.5),
                    name=f'ROC (AUC = {auc_val:.4f})'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], mode='lines',
                    line=dict(color='#94a3b8', dash='dash'),
                    name='Random Classifier'
                ))
                fig_roc.update_layout(**plot_layout("ROC Curve"),
                                      xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

                # Precision-Recall curve
                prec_c, rec_c, _ = skm.precision_recall_curve(y_te, y_prob)
                ap = skm.average_precision_score(y_te, y_prob)
                fig_apc = go.Figure()
                fig_apc.add_trace(go.Scatter(
                    x=rec_c, y=prec_c, mode='lines',
                    line=dict(color='#db2777', width=2.5),
                    name=f'PR Curve (AP = {ap:.4f})',
                    fill='tozeroy', fillcolor='rgba(219,39,119,0.08)'
                ))
                fig_apc.update_layout(**plot_layout("Precision-Recall Curve"),
                                      xaxis_title="Recall",
                                      yaxis_title="Precision")
                st.plotly_chart(fig_apc, use_container_width=True)

                col_roc1, col_roc2 = st.columns(2)
                col_roc1.markdown(
                    f'<div class="metric-card"><span class="metric-value" style="color:#0099cc">{auc_val:.4f}</span>'
                    f'<span class="metric-label">ROC-AUC</span></div>', unsafe_allow_html=True
                )
                col_roc2.markdown(
                    f'<div class="metric-card"><span class="metric-value" style="color:#db2777">{ap:.4f}</span>'
                    f'<span class="metric-label">Avg Precision (PR-AUC)</span></div>', unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════
    #  REGRESSION
    # ══════════════════════════════════════════════════════════════
    else:
        train_r2 = skm.r2_score(y_tr, y_pred_tr)
        test_r2  = skm.r2_score(y_te, y_pred)
        rmse     = np.sqrt(skm.mean_squared_error(y_te, y_pred))
        mae      = skm.mean_absolute_error(y_te, y_pred)
        mape     = np.mean(np.abs((y_te - y_pred) / (np.abs(y_te) + 1e-8))) * 100
        residuals = y_te - y_pred

        diff = train_r2 - test_r2
        if diff > 0.15:
            fit_label, fit_color = "OVERFITTING", "#dc2626"
        elif test_r2 < 0.5:
            fit_label, fit_color = "UNDERFITTING", "#d97706"
        else:
            fit_label, fit_color = "GOOD FIT", "#059669"

        cols = st.columns(6)
        reg_metrics = [
            ("Test R²",      f"{test_r2:.4f}",   "#0099cc"),
            ("Train R²",     f"{train_r2:.4f}",  "#7c3aed"),
            ("RMSE",         f"{rmse:.4f}",       "#db2777"),
            ("MAE",          f"{mae:.4f}",         "#d97706"),
            ("MAPE (%)",     f"{mape:.2f}%",       "#059669"),
            (f"Δ {diff:.3f}", fit_label,          fit_color),
        ]
        for col, (lbl, val, clr) in zip(cols, reg_metrics):
            col.markdown(
                f'<div class="metric-card"><span class="metric-value" style="color:{clr}; font-size:1.05rem;">{val}</span>'
                f'<span class="metric-label">{lbl}</span></div>',
                unsafe_allow_html=True
            )

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Actual vs Predicted", "📉 Residuals", "🔄 Train vs Test", "🎛️ Model-Specific"
        ])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode='markers',
                                     marker=dict(color='#0099cc', size=6, opacity=0.65),
                                     name='Test Predictions'))
            mn = min(float(y_te.min()), float(y_pred.min()))
            mx = max(float(y_te.max()), float(y_pred.max()))
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                     line=dict(color='#db2777', dash='dash', width=2),
                                     name='Perfect Fit'))
            fig.update_layout(**plot_layout("Actual vs Predicted"),
                              xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig_res = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Residuals vs Predicted", "Residuals Distribution"])
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                         marker=dict(color='#7c3aed', size=5, opacity=0.7)), row=1, col=1)
            fig_res.add_hline(y=0, line_dash="dash", line_color="#db2777", row=1, col=1)
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                            marker_color='#059669', opacity=0.8), row=1, col=2)
            fig_res.update_layout(**plot_layout(), showlegend=False)
            fig_res.update_xaxes(gridcolor='#e2e8f0')
            fig_res.update_yaxes(gridcolor='#e2e8f0')
            st.plotly_chart(fig_res, use_container_width=True)

        with tab3:
            # sample-by-sample comparison (first 60 test points)
            n_show = min(60, len(y_te))
            idx_show = np.arange(n_show)
            fig_tvt = go.Figure()
            fig_tvt.add_trace(go.Scatter(x=idx_show, y=y_te[:n_show],
                                         mode='lines+markers',
                                         name='Actual',
                                         line=dict(color='#0099cc', width=2),
                                         marker=dict(size=5)))
            fig_tvt.add_trace(go.Scatter(x=idx_show, y=y_pred[:n_show],
                                         mode='lines+markers',
                                         name='Predicted',
                                         line=dict(color='#db2777', width=2, dash='dot'),
                                         marker=dict(size=5, symbol='diamond')))
            fig_tvt.update_layout(**plot_layout(f"Actual vs Predicted — first {n_show} test samples"),
                                   xaxis_title="Sample Index",
                                   yaxis_title="Value")
            st.plotly_chart(fig_tvt, use_container_width=True)

            # error bars version
            errors = np.abs(residuals[:n_show])
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(x=idx_show, y=errors,
                                     marker=dict(
                                         color=errors,
                                         colorscale=[[0,'#d1fae5'],[0.5,'#fef3c7'],[1,'#fee2e2']],
                                         showscale=True,
                                         colorbar=dict(title='|Error|')
                                     ),
                                     name='|Error|'))
            fig_err.update_layout(**plot_layout(f"|Absolute Error| per Test Sample"),
                                   xaxis_title="Sample Index",
                                   yaxis_title="|Error|")
            st.plotly_chart(fig_err, use_container_width=True)

        with tab4:
            st.markdown(f"#### Model: `{model_name}`")

            # ── Linear Regression: coefficients ───────────────────
            if "Linear" in model_name:
                section_card("Feature Coefficients", "📏")
                feats = st.session_state.feature_selected_cols or st.session_state.selected_features
                coef_df = pd.DataFrame({
                    "Feature": feats,
                    "Coefficient": model.coef_
                }).sort_values("Coefficient", key=abs, ascending=True)
                fig_coef = go.Figure(go.Bar(
                    y=coef_df["Feature"], x=coef_df["Coefficient"], orientation='h',
                    marker=dict(color=['#db2777' if v < 0 else '#059669' for v in coef_df["Coefficient"]]),
                    text=[f"{v:.4f}" for v in coef_df["Coefficient"]],
                    textposition='outside', textfont=dict(color='#1e293b', size=10)
                ))
                fig_coef.update_layout(**plot_layout("Feature Coefficients (Linear Regression)"),
                                       height=max(300, len(feats)*32))
                st.plotly_chart(fig_coef, use_container_width=True)
                st.info(f"Intercept: **{model.intercept_:.4f}**")

            # ── SVR: support vectors ───────────────────────────────
            elif "SVR" in model_name or ("SVM" in model_name and not is_cls):
                section_card("SVR Support Vector Distribution", "⚔️")
                from sklearn.decomposition import PCA
                pca2 = PCA(n_components=2)
                X_2d = pca2.fit_transform(X_te)
                df_svr = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
                df_svr["Actual"]    = y_te
                df_svr["Predicted"] = y_pred
                df_svr["|Error|"]   = np.abs(residuals)
                fig_svr = px.scatter(df_svr, x="PC1", y="PC2",
                                     color="|Error|",
                                     color_continuous_scale=[[0,'#d1fae5'],[0.5,'#fef3c7'],[1,'#fee2e2']],
                                     title="SVR — Test Set (colour = |error|)",
                                     size=df_svr["|Error|"].clip(upper=df_svr["|Error|"].quantile(0.95))+0.5,
                                     opacity=0.8)
                fig_svr.update_layout(**plot_layout())
                st.plotly_chart(fig_svr, use_container_width=True)
                st.info(f"Support vectors: **{model.n_support_[0]}**")

            # ── Random Forest: importances ─────────────────────────
            elif "Random Forest" in model_name:
                section_card("Feature Importances (MDI)", "🌲")
                feats = st.session_state.feature_selected_cols or st.session_state.selected_features
                imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True)
                fig_imp = go.Figure(go.Bar(
                    y=imp.index, x=imp.values, orientation='h',
                    marker=dict(
                        color=imp.values,
                        colorscale=[[0,'#e2e8f0'],[0.6,'#7c3aed'],[1,'#059669']],
                        showscale=False
                    ),
                    text=[f"{v:.4f}" for v in imp.values],
                    textposition='outside',
                    textfont=dict(color='#1e293b', size=10)
                ))
                fig_imp.update_layout(**plot_layout("Feature Importances"),
                                      height=max(300, len(feats)*32))
                st.plotly_chart(fig_imp, use_container_width=True)

                with st.expander("🌳 View a Single Decision Tree (depth ≤ 3)"):
                    try:
                        from sklearn.tree import export_text
                        tree_ = model.estimators_[0]
                        tree_text = export_text(tree_, feature_names=list(feats), max_depth=3)
                        st.code(tree_text, language="text")
                    except Exception as e:
                        st.warning(f"Could not render tree: {e}")

            # ── KNN Regressor: distance distribution ──────────────
            elif "K-Nearest" in model_name:
                section_card("Neighbour Distance Distribution", "🔵")
                dists, _ = model.kneighbors(X_te)
                mean_dists = dists.mean(axis=1)
                fig_knn = go.Figure()
                fig_knn.add_trace(go.Scatter(x=mean_dists, y=np.abs(residuals),
                                             mode='markers',
                                             marker=dict(color='#0099cc', size=6, opacity=0.7),
                                             name='|Error| vs Mean Dist'))
                fig_knn.update_layout(**plot_layout(f"Mean Neighbour Distance vs |Error|"),
                                      xaxis_title="Mean Neighbour Distance",
                                      yaxis_title="|Residual|")
                st.plotly_chart(fig_knn, use_container_width=True)

    if nav_buttons(next_label="Hyperparameter Tuning →", next_key="met_next"):
        st.session_state.step = 9
        st.rerun()


def step_hyperparameter_tuning():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">10</div>
        <div class="step-header-large">Hyperparameter Tuning</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn import metrics as skm
    import time

    is_cls = st.session_state.problem_type == "Classification"
    model_name = st.session_state.model_name
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test

    def get_param_grid():
        if "SVM" in model_name:
            return {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto"]}
        elif "Random Forest" in model_name:
            return {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
        elif "Linear" in model_name or "Logistic" in model_name:
            if is_cls:
                return {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]}
            else:
                return {"fit_intercept": [True, False], "positive": [False, True]}
        else:
            return {"n_neighbors": [3, 5, 7, 10, 15], "weights": ["uniform", "distance"], "metric": ["minkowski", "euclidean", "manhattan"]}

    def build_base_model():
        if is_cls:
            if "Logistic" in model_name:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000, random_state=42)
            elif "SVM" in model_name:
                from sklearn.svm import SVC
                return SVC(random_state=42)
            elif "Random Forest" in model_name:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=42)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
        else:
            if "Linear" in model_name:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif "SVR" in model_name or "SVM" in model_name:
                from sklearn.svm import SVR
                return SVR()
            elif "Random Forest" in model_name:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(random_state=42)
            else:
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor()

    param_grid = get_param_grid()

    section_card("Search Configuration", "⚙️")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_method = st.selectbox("Search Strategy", ["Grid Search", "Randomized Search"])
    with col2:
        cv_k = st.slider("CV Folds for Tuning", 2, 10, 3)
    with col3:
        if search_method == "Randomized Search":
            n_iter = st.slider("# Iterations", 5, 50, 20)
        else:
            n_iter = None

    scoring = 'accuracy' if is_cls else 'r2'

    section_card("Parameter Grid", "🔧")
    for param, values in param_grid.items():
        st.markdown(f'<div style="color:#475569; font-size:0.82rem; margin:3px 0; font-family:Space Mono,monospace;"><span style="color:#0099cc; font-weight:700;">{param}</span>: {values}</div>', unsafe_allow_html=True)

    if st.button("🔬 Start Tuning", key="tune_btn", type="primary"):
        with st.spinner("Running hyperparameter search..."):
            start = time.time()
            base = build_base_model()
            if search_method == "Grid Search":
                searcher = GridSearchCV(base, param_grid, cv=cv_k, scoring=scoring,
                                        n_jobs=-1, verbose=0, return_train_score=True)
            else:
                searcher = RandomizedSearchCV(base, param_grid, cv=cv_k, scoring=scoring,
                                              n_iter=n_iter, random_state=42, n_jobs=-1,
                                              verbose=0, return_train_score=True)
            searcher.fit(X_tr, y_tr)
            elapsed = time.time() - start

        st.session_state.best_params = searcher.best_params_
        best_model = searcher.best_estimator_
        y_pred_best = best_model.predict(X_te)

        orig_model  = st.session_state.model
        y_pred_orig = orig_model.predict(X_te)

        if is_cls:
            orig_score = skm.accuracy_score(y_te, y_pred_orig)
            best_score = skm.accuracy_score(y_te, y_pred_best)
            metric_name = "Accuracy"
        else:
            orig_score = skm.r2_score(y_te, y_pred_orig)
            best_score = skm.r2_score(y_te, y_pred_best)
            metric_name = "R²"

        improvement = best_score - orig_score

        section_card("Tuning Results", "🏆")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{orig_score:.4f}</span><span class="metric-label">Original {metric_name}</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#059669">{best_score:.4f}</span><span class="metric-label">Tuned {metric_name}</span></div>', unsafe_allow_html=True)
        imp_color = "#059669" if improvement >= 0 else "#dc2626"
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{imp_color}">{"+" if improvement>=0 else ""}{improvement:.4f}</span><span class="metric-label">Improvement</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#d97706">{elapsed:.1f}s</span><span class="metric-label">Search Time</span></div>', unsafe_allow_html=True)

        section_card("Best Parameters Found", "✨")
        bp_cols = st.columns(len(st.session_state.best_params))
        for i, (k, v) in enumerate(st.session_state.best_params.items()):
            bp_cols[i].markdown(f'<div class="metric-card"><span class="metric-value" style="font-size:1.1rem; color:#db2777">{v}</span><span class="metric-label">{k}</span></div>', unsafe_allow_html=True)

        cv_df = pd.DataFrame(searcher.cv_results_)
        cv_df = cv_df.sort_values('rank_test_score').head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(cv_df))),
            y=cv_df['mean_test_score'],
            error_y=dict(type='data', array=cv_df['std_test_score'], visible=True,
                         color='rgba(0,0,0,0.15)'),
            marker=dict(
                color=cv_df['mean_test_score'],
                colorscale=[[0,'#e2e8f0'],[0.5,'#7c3aed'],[1,'#0099cc']],
                showscale=False
            ),
            name='Test Score'
        ))
        if 'mean_train_score' in cv_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(cv_df))),
                y=cv_df['mean_train_score'],
                mode='lines+markers',
                line=dict(color='#059669', dash='dot'),
                name='Train Score'
            ))
        fig.update_layout(**plot_layout(f"Top {len(cv_df)} Parameter Combinations"))
        fig.update_xaxes(title="Parameter Set Rank")
        fig.update_yaxes(title=f"CV {metric_name}")
        st.plotly_chart(fig, use_container_width=True)

        if improvement > 0:
            if st.button("✅ Update model with best parameters", key="update_model"):
                st.session_state.model = best_model
                st.success("✓ Model updated with best parameters!")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #e0f7ff, #ede9fe);
                    border:1px solid #a5b4fc; border-radius:12px; padding:1.5rem; margin-top:1rem; text-align:center;">
            <div style="font-family:'Orbitron',monospace; color:#1e293b; font-size:1rem; font-weight:700; letter-spacing:0.1em;">
                🎉 Pipeline Complete!
            </div>
            <p style="color:#475569; font-size:0.85rem; margin-top:0.5rem;">
                Your ML pipeline from data input to hyperparameter tuning is complete.
            </p>
            <div style="margin-top:0.8rem;">
                <span class="badge badge-green">Model: {model_name}</span>
                <span class="badge badge-cyan">Best CV: {searcher.best_score_:.4f}</span>
                <span class="badge badge-purple">{st.session_state.problem_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if nav_buttons(back=True, next_label="🔄 Restart Pipeline", next_key="restart_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_state()
        st.rerun()


def main():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚗️ ML Pipeline Studio</div>
        <div class="hero-sub">End-to-End Machine Learning Orchestration</div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_bar()
    st.markdown('<hr style="margin:0 0 1.5rem 0;">', unsafe_allow_html=True)

    step = st.session_state.step

    if step == 0:
        step_problem_type()
    elif step == 1:
        step_data_input()
    elif step == 2:
        step_eda()
    elif step == 3:
        step_data_engineering()
    elif step == 4:
        step_feature_selection()
    elif step == 5:
        step_data_split()
    elif step == 6:
        step_model_selection()
    elif step == 7:
        step_training()
    elif step == 8:
        step_metrics()
    elif step == 9:
        step_hyperparameter_tuning()


if __name__ == "__main__":
    main()