import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1629;
    --bg-card: #141b2d;
    --bg-card-hover: #1a2240;
    --accent-cyan: #00d4ff;
    --accent-purple: #8b5cf6;
    --accent-pink: #f472b6;
    --accent-green: #10b981;
    --accent-orange: #f59e0b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: #1e2d4a;
    --border-bright: #2d4070;
    --step-active: #00d4ff;
    --step-done: #10b981;
    --step-pending: #334155;
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: var(--bg-primary) !important;
    background-image: 
        radial-gradient(ellipse at 20% 0%, rgba(0,212,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(139,92,246,0.06) 0%, transparent 50%) !important;
}

.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1600px !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
    position: relative;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    margin: 0;
    text-shadow: none;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.15em;
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
    font-size: 1.1rem;
    font-weight: 700;
    border: 2px solid;
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
}
.step-circle.pending {
    background: var(--bg-card);
    border-color: var(--step-pending);
    color: var(--text-muted);
}
.step-circle.active {
    background: rgba(0,212,255,0.15);
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    box-shadow: 0 0 20px rgba(0,212,255,0.4);
}
.step-circle.done {
    background: rgba(16,185,129,0.15);
    border-color: var(--accent-green);
    color: var(--accent-green);
    box-shadow: 0 0 12px rgba(16,185,129,0.3);
}
.step-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    text-align: center;
    max-width: 80px;
    line-height: 1.2;
    font-family: 'Space Mono', monospace;
}
.step-label.pending { color: var(--text-muted); }
.step-label.active { color: var(--accent-cyan); font-weight: 700; }
.step-label.done { color: var(--accent-green); }

.step-connector {
    height: 2px;
    width: 50px;
    margin-bottom: 22px;
    flex-shrink: 0;
}
.step-connector.done { background: var(--accent-green); }
.step-connector.active { background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan)); }
.step-connector.pending { background: var(--step-pending); }

/* Section Cards */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-cyan);
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
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent-cyan); }
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-cyan);
    display: block;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
    font-family: 'Space Mono', monospace;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1)) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.2) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 900 !important;
}

/* Inputs */
.stSelectbox > div > div, .stMultiSelect > div > div,
.stNumberInput > div > div > input, .stTextInput > div > div > input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stMultiSelect label, .stNumberInput label,
.stTextInput label, .stSlider label, .stRadio label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Radio buttons */
.stRadio > div {
    display: flex !important;
    gap: 1rem !important;
    flex-direction: row !important;
}
.stRadio > div > label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: none !important;
    font-size: 0.9rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: normal !important;
}

/* File uploader */
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* Dataframe */
.stDataFrame { border-radius: 8px !important; overflow: hidden !important; }

/* Alerts */
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid var(--accent-green) !important; border-radius: 8px !important; color: var(--accent-green) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid var(--accent-orange) !important; border-radius: 8px !important; }
.stError { background: rgba(244,114,182,0.1) !important; border: 1px solid var(--accent-pink) !important; border-radius: 8px !important; }
.stInfo { background: rgba(0,212,255,0.07) !important; border: 1px solid rgba(0,212,255,0.3) !important; border-radius: 8px !important; color: var(--text-primary) !important; }

/* Progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* Tag Badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 8px;
}
.badge-cyan { background: rgba(0,212,255,0.15); color: var(--accent-cyan); border: 1px solid rgba(0,212,255,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }
.badge-purple { background: rgba(139,92,246,0.15); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3); }
.badge-orange { background: rgba(245,158,11,0.15); color: var(--accent-orange); border: 1px solid rgba(245,158,11,0.3); }

/* Stat row */
.stat-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.stat-pill {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: 'Space Mono', monospace;
}
.stat-pill span { color: var(--accent-cyan); font-weight: 700; }

h1, h2, h3 { color: var(--text-primary) !important; }
p, li { color: var(--text-secondary) !important; }

.step-header-large {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--text-primary);
    margin: 0;
}
.step-number-badge {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    font-weight: 900;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────
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
    ("🎛️", "Hyperparameter\nTuning"),
]


# ─── Pipeline Progress Bar ─────────────────────────────────────────────────────
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
    st.markdown('<hr style="margin:0 0 1.5rem 0;">', unsafe_allow_html=True)


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


# ─── STEP 0: Problem Type ──────────────────────────────────────────────────────
def step_problem_type():
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 2rem;">
        <div class="step-number-badge">01</div>
        <div class="step-header-large">Problem Type</div>
        <p style="color: #64748b; margin-top:0.5rem; font-family:'Space Mono',monospace; font-size:0.8rem; letter-spacing:0.1em;">
            SELECT THE MACHINE LEARNING TASK
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div style="background:#141b2d; border:2px solid #1e2d4a; border-radius:16px; padding:2rem; text-align:center; cursor:pointer; transition:all 0.3s;" 
             onmouseover="this.style.borderColor='#00d4ff'" onmouseout="this.style.borderColor='#1e2d4a'">
            <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.1rem; color:#e2e8f0; font-weight:700;">Classification</div>
            <p style="color:#64748b; font-size:0.82rem; margin-top:0.5rem; line-height:1.5;">
                Predict discrete categories or labels.<br>Binary, multi-class, or multi-label tasks.
            </p>
            <div style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span class="badge badge-cyan">Logistic</span>
                <span class="badge badge-purple">SVM</span>
                <span class="badge badge-green">Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Classification", key="cls_btn", use_container_width=True):
            st.session_state.problem_type = "Classification"
            st.session_state.step = 1
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#141b2d; border:2px solid #1e2d4a; border-radius:16px; padding:2rem; text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">📈</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.1rem; color:#e2e8f0; font-weight:700;">Regression</div>
            <p style="color:#64748b; font-size:0.82rem; margin-top:0.5rem; line-height:1.5;">
                Predict continuous numerical values.<br>Linear, polynomial, or non-linear relationships.
            </p>
            <div style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span class="badge badge-orange">Linear</span>
                <span class="badge badge-purple">SVR</span>
                <span class="badge badge-cyan">Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Regression", key="reg_btn", use_container_width=True):
            st.session_state.problem_type = "Regression"
            st.session_state.step = 1
            st.rerun()


# ─── STEP 1: Data Input ────────────────────────────────────────────────────────
def step_data_input():
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">02</div>
        <div>
            <div class="step-header-large">Data Input</div>
            <span class="badge badge-{'cyan' if st.session_state.problem_type=='Classification' else 'orange'}">{st.session_state.problem_type}</span>
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
        st.markdown('<div style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;">Load a built-in dataset to explore the pipeline:</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🌸 Iris (Classification)", use_container_width=True):
                from sklearn.datasets import load_iris
                d = load_iris(as_frame=True)
                st.session_state.df = pd.concat([d.data, d.target.rename("target")], axis=1)
                st.session_state.problem_type = "Classification"
                st.rerun()
        with col_b:
            if st.button("🏠 Boston Housing (Regression)", use_container_width=True):
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

        # Stats row
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

        # Target selection
        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        section_card("Select Target Feature", "🎯")
        target = st.selectbox("Target column", df.columns.tolist(),
                               index=len(df.columns)-1)
        st.session_state.target = target

        feature_cols = [c for c in df.columns if c != target]
        section_card("Select Input Features", "📌")
        sel_features = st.multiselect("Features to include", feature_cols, default=feature_cols)
        st.session_state.selected_features = sel_features

        if sel_features:
            # PCA Visualization
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
                                            color_discrete_sequence=px.colors.qualitative.Bold)
                    else:
                        fig = px.scatter(pca_df, x="PC1", y="PC2", color="target",
                                         title="PCA — 2D Projection", opacity=0.8,
                                         color_discrete_sequence=px.colors.qualitative.Bold)
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,14,26,0.8)',
                        font=dict(color='#94a3b8', family='DM Sans'),
                        title_font=dict(color='#e2e8f0', size=14),
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig2 = go.Figure(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(var))],
                        y=var * 100,
                        marker=dict(
                            color=['#00d4ff', '#8b5cf6', '#f472b6'][:len(var)],
                            opacity=0.85
                        ),
                        text=[f"{v:.1f}%" for v in var * 100],
                        textposition='outside',
                        textfont=dict(color='#e2e8f0', size=11)
                    ))
                    fig2.update_layout(
                        title="Explained Variance",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(10,14,26,0.8)',
                        font=dict(color='#94a3b8', family='DM Sans'),
                        title_font=dict(color='#e2e8f0', size=14),
                        xaxis=dict(gridcolor='#1e2d4a'),
                        yaxis=dict(gridcolor='#1e2d4a', title="Variance %"),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Select at least 2 numeric features for PCA visualization.")

        if nav_buttons(back=True, next_label="Proceed to EDA →", next_key="data_next"):
            if sel_features and target:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Please select target and at least one feature.")


# ─── STEP 2: EDA ──────────────────────────────────────────────────────────────
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
        n = len(num_cols)
        cols_per_row = min(3, n)
        if n > 0:
            selected_col = st.selectbox("Select feature", num_cols)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sub, x=selected_col, nbins=40,
                                   color_discrete_sequence=['#00d4ff'],
                                   title=f"Distribution: {selected_col}")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(10,14,26,0.8)',
                                  font=dict(color='#94a3b8'),
                                  xaxis=dict(gridcolor='#1e2d4a'),
                                  yaxis=dict(gridcolor='#1e2d4a'),
                                  margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Violin(y=sub[selected_col].dropna(), box_visible=True,
                                         line_color='#8b5cf6', fillcolor='rgba(139,92,246,0.2)',
                                         name=selected_col, meanline_visible=True))
                fig2.update_layout(title=f"Violin: {selected_col}",
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(10,14,26,0.8)',
                                   font=dict(color='#94a3b8'),
                                   yaxis=dict(gridcolor='#1e2d4a'),
                                   margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_card("Correlation Matrix", "🔗")
        if len(num_cols) > 1:
            corr = sub[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Pearson Correlation Heatmap")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#94a3b8'),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # High correlations
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
            colors = ['#00d4ff', '#8b5cf6', '#f472b6', '#10b981', '#f59e0b',
                      '#3b82f6', '#ec4899', '#14b8a6', '#a78bfa', '#fb923c']
            for i, col in enumerate(num_cols):
                fig.add_trace(go.Box(
                    y=sub[col].dropna(), name=col,
                    marker_color=colors[i % len(colors)],
                    boxpoints='outliers', jitter=0.3,
                    fillcolor=f"rgba({','.join(str(int(c,16)) for c in [colors[i%len(colors)][1:3], colors[i%len(colors)][3:5], colors[i%len(colors)][5:]])},0.15)"
                ))
            fig.update_layout(title="Feature Boxplots",
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(10,14,26,0.8)',
                              font=dict(color='#94a3b8'),
                              yaxis=dict(gridcolor='#1e2d4a'),
                              xaxis=dict(gridcolor='#1e2d4a'),
                              showlegend=False,
                              margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section_card("Target Variable Analysis", "🎯")
        if target in num_cols:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sub, x=target, nbins=40,
                                   color_discrete_sequence=['#f472b6'],
                                   title=f"Target Distribution: {target}")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(10,14,26,0.8)',
                                  font=dict(color='#94a3b8'),
                                  xaxis=dict(gridcolor='#1e2d4a'),
                                  yaxis=dict(gridcolor='#1e2d4a'),
                                  margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                # Q-Q plot
                from scipy import stats as sp_stats
                sorted_vals = np.sort(sub[target].dropna())
                theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_vals)))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=theoretical, y=sorted_vals, mode='markers',
                                          marker=dict(color='#10b981', size=4, opacity=0.7),
                                          name='Data'))
                fig2.add_trace(go.Scatter(x=[theoretical[0], theoretical[-1]],
                                          y=[theoretical[0], theoretical[-1]],
                                          mode='lines', line=dict(color='#f472b6', dash='dash'),
                                          name='Normal'))
                fig2.update_layout(title="Q-Q Plot",
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(10,14,26,0.8)',
                                   font=dict(color='#94a3b8'),
                                   xaxis=dict(gridcolor='#1e2d4a', title="Theoretical Quantiles"),
                                   yaxis=dict(gridcolor='#1e2d4a', title="Sample Quantiles"),
                                   margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            vc = sub[target].value_counts()
            fig = px.bar(x=vc.index.astype(str), y=vc.values,
                         labels={'x': target, 'y': 'Count'},
                         title="Class Distribution",
                         color=vc.values,
                         color_continuous_scale=['#8b5cf6', '#00d4ff'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font=dict(color='#94a3b8'),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Class imbalance warning
            if vc.max() / vc.min() > 3:
                st.warning("⚠️ Significant class imbalance detected.")

    with tab5:
        section_card("Descriptive Statistics", "📋")
        st.dataframe(sub[num_cols].describe().round(4), use_container_width=True)
        # Missing values
        miss = sub.isnull().sum()
        miss = miss[miss > 0]
        if len(miss) > 0:
            st.warning(f"Missing values found in {len(miss)} column(s):")
            fig = px.bar(x=miss.index, y=miss.values,
                         labels={'x': 'Column', 'y': 'Missing Count'},
                         color=miss.values, color_continuous_scale=['#f59e0b', '#f472b6'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font=dict(color='#94a3b8'),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✓ No missing values found!")

    if nav_buttons(next_label="Data Engineering →", next_key="eda_next"):
        st.session_state.step = 3
        st.rerun()


# ─── STEP 3: Data Engineering ─────────────────────────────────────────────────
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

    # Imputation
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

    # Outlier Detection
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
        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.4); border-radius:10px; padding:1rem; margin:1rem 0;">
            <div style="font-family:'Orbitron',monospace; color:#f59e0b; font-size:0.9rem; font-weight:700;">⚠️ OUTLIERS DETECTED</div>
            <div style="color:#94a3b8; margin-top:0.4rem; font-size:0.85rem;">{n_out} rows ({pct:.1f}% of data) identified as outliers</div>
        </div>
        """, unsafe_allow_html=True)

        # Visualization of outliers using PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X_vis = sub[num_cols].fillna(sub[num_cols].median())
        Xs = StandardScaler().fit_transform(X_vis)
        pcs = PCA(n_components=2).fit_transform(Xs)
        vis_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=sub.index)
        vis_df["type"] = "Normal"
        vis_df.loc[st.session_state.outlier_indices, "type"] = "Outlier"

        fig = px.scatter(vis_df, x="PC1", y="PC2", color="type",
                         color_discrete_map={"Normal": "#00d4ff", "Outlier": "#f472b6"},
                         title="Outlier Visualization (PCA Space)",
                         opacity=0.7)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(10,14,26,0.8)',
                           font=dict(color='#94a3b8'),
                           xaxis=dict(gridcolor='#1e2d4a'),
                           yaxis=dict(gridcolor='#1e2d4a'),
                           margin=dict(l=0,r=0,t=40,b=0))
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


# ─── STEP 4: Feature Selection ────────────────────────────────────────────────
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
            colors_list = ['#f472b6' if v < list(results.values())[-1] * 0.3 else '#00d4ff'
                           for v in res_series.values]
            fig = go.Figure(go.Bar(
                y=res_series.index, x=res_series.values, orientation='h',
                marker=dict(color=['#10b981' if n in selected else '#475569' for n in res_series.index]),
                text=[f"{v:.4f}" for v in res_series.values],
                textposition='outside', textfont=dict(color='#e2e8f0', size=10)
            ))
            fig.update_layout(title=f"Feature Scores — {method}",
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font=dict(color='#94a3b8'),
                               xaxis=dict(gridcolor='#1e2d4a'),
                               yaxis=dict(gridcolor='#1e2d4a'),
                               margin=dict(l=0,r=0,t=40,b=0),
                               height=max(300, len(results) * 35))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="section-card" style="margin-top:0;">
                <div class="section-title">📌 SELECTED</div>
                <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#10b981; font-weight:900;">{len(selected)}</div>
                <div style="color:#64748b; font-size:0.75rem; font-family:'Space Mono',monospace;">OF {len(num_feats)} FEATURES</div>
                <hr style="margin:0.8rem 0; border-color:#1e2d4a;">
            """, unsafe_allow_html=True)
            for f in selected:
                st.markdown(f'<div style="color:#e2e8f0; font-size:0.82rem; padding:3px 0;">✓ {f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Allow manual override
        final_sel = st.multiselect("Confirm/modify selected features", num_feats, default=selected)
        st.session_state.feature_selected_cols = final_sel if final_sel else selected

    if nav_buttons(next_label="Data Split →", next_key="feat_next"):
        if not st.session_state.feature_selected_cols:
            st.session_state.feature_selected_cols = num_feats
        st.session_state.step = 5
        st.rerun()


# ─── STEP 5: Data Split ───────────────────────────────────────────────────────
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
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05,
                               help="Fraction of data for testing")
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

        # Visualize split
        train_n, test_n = len(X_tr), len(X_te)
        total = train_n + test_n

        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.markdown(f'<div class="metric-card"><span class="metric-value">{train_n:,}</span><span class="metric-label">Train Samples</span></div>', unsafe_allow_html=True)
        col_v2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#8b5cf6">{test_n:,}</span><span class="metric-label">Test Samples</span></div>', unsafe_allow_html=True)
        col_v3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#10b981">{len(feats)}</span><span class="metric-label">Features</span></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Pie(
            values=[train_n, test_n],
            labels=[f'Train ({100*(1-test_size):.0f}%)', f'Test ({100*test_size:.0f}%)'],
            marker_colors=['#00d4ff', '#8b5cf6'],
            hole=0.6,
            textinfo='label+percent',
            textfont=dict(color='#e2e8f0', size=12)
        ))
        fig.update_layout(
            title="Data Split Distribution",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            showlegend=False,
            annotations=[dict(text=f'{total:,}<br>Total', x=0.5, y=0.5,
                              font=dict(size=14, color='#e2e8f0'), showarrow=False)],
            margin=dict(l=0,r=0,t=40,b=0), height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("✓ Data split complete. Features standardized with StandardScaler.")

    if nav_buttons(next_label="Model Selection →", next_key="split_next"):
        if st.session_state.X_train is None:
            st.error("Please apply the split first.")
        else:
            st.session_state.step = 6
            st.rerun()


# ─── STEP 6: Model Selection ──────────────────────────────────────────────────
def step_model_selection():
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">07</div>
        <div class="step-header-large">Model Selection</div>
        <span class="badge badge-{'cyan' if st.session_state.problem_type=='Classification' else 'orange'}">{st.session_state.problem_type}</span>
    </div>
    """, unsafe_allow_html=True)

    is_cls = st.session_state.problem_type == "Classification"

    if is_cls:
        models_info = {
            "Logistic Regression": {"icon": "📐", "desc": "Linear decision boundary, fast, interpretable", "color": "#00d4ff"},
            "SVM (Support Vector Machine)": {"icon": "⚔️", "desc": "Kernel-based, powerful for non-linear boundaries", "color": "#8b5cf6"},
            "Random Forest": {"icon": "🌲", "desc": "Ensemble of decision trees, robust to overfitting", "color": "#10b981"},
            "K-Nearest Neighbors": {"icon": "🔵", "desc": "Instance-based, no training phase", "color": "#f59e0b"},
        }
    else:
        models_info = {
            "Linear Regression": {"icon": "📏", "desc": "Simple linear relationship modeling", "color": "#00d4ff"},
            "SVM Regression (SVR)": {"icon": "⚔️", "desc": "Kernel-based regression with epsilon tube", "color": "#8b5cf6"},
            "Random Forest Regressor": {"icon": "🌲", "desc": "Ensemble method, handles non-linearity well", "color": "#10b981"},
            "K-Nearest Neighbors Regressor": {"icon": "🔵", "desc": "Predict based on nearest neighbors", "color": "#f59e0b"},
        }

    selected_model = st.session_state.get('model_name', list(models_info.keys())[0])

    cols = st.columns(len(models_info))
    for i, (name, info) in enumerate(models_info.items()):
        with cols[i]:
            is_sel = (name == selected_model)
            border = info['color'] if is_sel else '#1e2d4a'
            bg = f"rgba{tuple(int(info['color'][j:j+2],16) for j in (1,3,5))}".replace('rgba', 'rgba(').replace(')', ',0.1)') if is_sel else 'var(--bg-card)'
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {border}; border-radius:12px; padding:1.2rem; text-align:center; min-height:140px;">
                <div style="font-size:2rem;">{info['icon']}</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#e2e8f0; font-weight:700; margin-top:0.5rem; letter-spacing:0.05em;">{name}</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:0.4rem; line-height:1.4;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
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


# ─── STEP 7: Training & KFold ─────────────────────────────────────────────────
def step_training():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">08</div>
        <div class="step-header-large">Model Training & K-Fold CV</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn import metrics as skm

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
            y_pred = model.predict(X_te)

            st.session_state.model = model
            st.session_state.cv_results = cv_scores
            st.session_state.y_pred = y_pred

        # KFold results
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
                colorscale=[[0, '#f472b6'], [0.5, '#8b5cf6'], [1, '#00d4ff']],
                showscale=False
            ),
            text=[f"{s:.4f}" for s in cv_scores],
            textposition='outside',
            textfont=dict(color='#e2e8f0')
        ))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#10b981",
                      annotation_text=f"Mean: {cv_scores.mean():.4f}",
                      annotation_font_color="#10b981")
        fig.update_layout(
            title=f"CV {scoring.upper()} per Fold",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.8)',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e2d4a'),
            yaxis=dict(gridcolor='#1e2d4a'),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{cv_scores.mean():.4f}</span><span class="metric-label">Mean CV Score</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#8b5cf6">{cv_scores.std():.4f}</span><span class="metric-label">Std Deviation</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#10b981">{cv_scores.max():.4f}</span><span class="metric-label">Best Fold</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f472b6">{cv_scores.min():.4f}</span><span class="metric-label">Worst Fold</span></div>', unsafe_allow_html=True)

        st.success(f"✓ Model trained! Mean CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if nav_buttons(next_label="View Metrics →", next_key="train_next"):
        if st.session_state.model is None:
            st.error("Please train the model first.")
        else:
            st.session_state.step = 8
            st.rerun()


# ─── STEP 8: Metrics ─────────────────────────────────────────────────────────
def step_metrics():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <div class="step-number-badge">09</div>
        <div class="step-header-large">Performance Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn import metrics as skm
    is_cls = st.session_state.problem_type == "Classification"
    model = st.session_state.model
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test

    if model is None:
        st.error("No trained model found. Please go back and train a model.")
        return

    y_pred = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)

    if is_cls:
        train_acc = skm.accuracy_score(y_tr, y_pred_tr)
        test_acc = skm.accuracy_score(y_te, y_pred)
        train_f1 = skm.f1_score(y_tr, y_pred_tr, average='weighted')
        test_f1 = skm.f1_score(y_te, y_pred, average='weighted')

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{test_acc:.4f}</span><span class="metric-label">Test Accuracy</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#8b5cf6">{test_f1:.4f}</span><span class="metric-label">F1 Score</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#10b981">{train_acc:.4f}</span><span class="metric-label">Train Accuracy</span></div>', unsafe_allow_html=True)

        # Overfit/underfit
        diff = train_acc - test_acc
        if diff > 0.1:
            label, color, msg = "OVERFITTING", "#f472b6", f"Train-Test gap: {diff:.3f}"
        elif test_acc < 0.6:
            label, color, msg = "UNDERFITTING", "#f59e0b", "Low test accuracy"
        else:
            label, color, msg = "GOOD FIT", "#10b981", "Model generalizes well"
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}; font-size:1rem;">{label}</span><span class="metric-label">{msg}</span></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🗺️ Confusion Matrix", "📊 Classification Report", "🔄 Train vs Test"])

        with tab1:
            cm = skm.confusion_matrix(y_te, y_pred)
            labels = [str(i) for i in sorted(set(y_te))]
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            labels=dict(x="Predicted", y="Actual"),
                            color_continuous_scale=[[0,'#0a0e1a'],[0.5,'#8b5cf6'],[1,'#00d4ff']],
                            x=labels, y=labels,
                            title="Confusion Matrix")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#94a3b8'),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            report = skm.classification_report(y_te, y_pred, output_dict=True)
            rep_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)

        with tab3:
            fig = go.Figure()
            cats = ['Accuracy', 'F1 Score']
            train_vals = [train_acc, train_f1]
            test_vals = [test_acc, test_f1]
            fig.add_trace(go.Bar(name='Train', x=cats, y=train_vals,
                                  marker_color='#00d4ff', opacity=0.85))
            fig.add_trace(go.Bar(name='Test', x=cats, y=test_vals,
                                  marker_color='#8b5cf6', opacity=0.85))
            fig.update_layout(
                barmode='group', title="Train vs Test Performance",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,14,26,0.8)',
                font=dict(color='#94a3b8'),
                yaxis=dict(gridcolor='#1e2d4a', range=[0,1]),
                xaxis=dict(gridcolor='#1e2d4a'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        train_r2 = skm.r2_score(y_tr, y_pred_tr)
        test_r2 = skm.r2_score(y_te, y_pred)
        rmse = np.sqrt(skm.mean_squared_error(y_te, y_pred))
        mae = skm.mean_absolute_error(y_te, y_pred)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{test_r2:.4f}</span><span class="metric-label">Test R²</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#8b5cf6">{train_r2:.4f}</span><span class="metric-label">Train R²</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f472b6">{rmse:.4f}</span><span class="metric-label">RMSE</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f59e0b">{mae:.4f}</span><span class="metric-label">MAE</span></div>', unsafe_allow_html=True)

        diff = train_r2 - test_r2
        if diff > 0.15:
            label, color = "OVERFITTING", "#f472b6"
        elif test_r2 < 0.5:
            label, color = "UNDERFITTING", "#f59e0b"
        else:
            label, color = "GOOD FIT", "#10b981"
        c5.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}; font-size:1rem;">{label}</span><span class="metric-label">Train-Test Δ: {diff:.3f}</span></div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🎯 Predictions vs Actual", "📉 Residuals"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode='markers',
                                      marker=dict(color='#00d4ff', size=6, opacity=0.7),
                                      name='Predictions'))
            mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx],
                                      line=dict(color='#f472b6', dash='dash'), name='Perfect Fit'))
            fig.update_layout(title="Predicted vs Actual",
                               xaxis_title="Actual", yaxis_title="Predicted",
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font=dict(color='#94a3b8'),
                               xaxis=dict(gridcolor='#1e2d4a'),
                               yaxis=dict(gridcolor='#1e2d4a'),
                               legend=dict(bgcolor='rgba(0,0,0,0)'),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            residuals = y_te - y_pred
            fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Residuals vs Predicted", "Residuals Distribution"])
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                      marker=dict(color='#8b5cf6', size=5, opacity=0.7)), row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="#f472b6", row=1, col=1)
            fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                        marker_color='#10b981', opacity=0.8), row=1, col=2)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font=dict(color='#94a3b8'),
                               showlegend=False,
                               margin=dict(l=0,r=0,t=40,b=0))
            fig.update_xaxes(gridcolor='#1e2d4a')
            fig.update_yaxes(gridcolor='#1e2d4a')
            st.plotly_chart(fig, use_container_width=True)

    if nav_buttons(next_label="Hyperparameter Tuning →", next_key="met_next"):
        st.session_state.step = 9
        st.rerun()


# ─── STEP 9: Hyperparameter Tuning ───────────────────────────────────────────
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

    # Build param grid based on model
    def get_param_grid():
        if "SVM" in model_name:
            return {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        elif "Random Forest" in model_name:
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        elif "Linear" in model_name or "Logistic" in model_name:
            if is_cls:
                return {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]}
            else:
                return {"fit_intercept": [True, False], "positive": [False, True]}
        else:  # KNN
            return {
                "n_neighbors": [3, 5, 7, 10, 15],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "euclidean", "manhattan"]
            }

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

    scoring = 'accuracy' if is_cls else 'r2'

    section_card("Parameter Grid", "🔧")
    for param, values in param_grid.items():
        st.markdown(f'<div style="color:#94a3b8; font-size:0.82rem; margin:3px 0; font-family:Space Mono,monospace;"><span style="color:#00d4ff;">{param}</span>: {values}</div>', unsafe_allow_html=True)

    if st.button("🔬 Start Tuning", key="tune_btn", type="primary"):
        with st.spinner("Running hyperparameter search... this may take a moment."):
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

        # Original model performance
        orig_model = st.session_state.model
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

        # Results display
        section_card("Tuning Results", "🏆")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><span class="metric-value">{orig_score:.4f}</span><span class="metric-label">Original {metric_name}</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#10b981">{best_score:.4f}</span><span class="metric-label">Tuned {metric_name}</span></div>', unsafe_allow_html=True)
        imp_color = "#10b981" if improvement >= 0 else "#f472b6"
        c3.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{imp_color}">{"+" if improvement>=0 else ""}{improvement:.4f}</span><span class="metric-label">Improvement</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f59e0b">{elapsed:.1f}s</span><span class="metric-label">Search Time</span></div>', unsafe_allow_html=True)

        # Best params
        st.markdown('<div style="margin:1rem 0;">', unsafe_allow_html=True)
        section_card("Best Parameters Found", "✨")
        bp_cols = st.columns(len(st.session_state.best_params))
        for i, (k, v) in enumerate(st.session_state.best_params.items()):
            bp_cols[i].markdown(f'<div class="metric-card"><span class="metric-value" style="font-size:1.1rem; color:#f472b6">{v}</span><span class="metric-label">{k}</span></div>', unsafe_allow_html=True)

        # CV Results visualization
        cv_df = pd.DataFrame(searcher.cv_results_)
        cv_df = cv_df.sort_values('rank_test_score').head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(cv_df))),
            y=cv_df['mean_test_score'],
            error_y=dict(type='data', array=cv_df['std_test_score'], visible=True,
                         color='rgba(255,255,255,0.3)'),
            marker=dict(
                color=cv_df['mean_test_score'],
                colorscale=[[0,'#334155'],[0.5,'#8b5cf6'],[1,'#00d4ff']],
                showscale=False
            ),
            name='Test Score'
        ))
        if 'mean_train_score' in cv_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(cv_df))),
                y=cv_df['mean_train_score'],
                mode='lines+markers',
                line=dict(color='#10b981', dash='dot'),
                name='Train Score'
            ))
        fig.update_layout(
            title=f"Top {len(cv_df)} Parameter Combinations (sorted by rank)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.8)',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e2d4a', title="Parameter Set Rank"),
            yaxis=dict(gridcolor='#1e2d4a', title=f"CV {metric_name}"),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        if improvement > 0:
            if st.button("✅ Update model with best parameters", key="update_model"):
                st.session_state.model = best_model
                st.success("✓ Model updated with best parameters!")

        # Summary card
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(0,212,255,0.05), rgba(139,92,246,0.05)); 
                    border:1px solid rgba(0,212,255,0.2); border-radius:12px; padding:1.5rem; margin-top:1rem; text-align:center;">
            <div style="font-family:'Orbitron',monospace; color:#e2e8f0; font-size:1rem; font-weight:700; letter-spacing:0.1em;">
                🎉 PIPELINE COMPLETE
            </div>
            <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
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


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚗️ ML PIPELINE STUDIO</div>
        <div class="hero-sub">End-to-End Machine Learning Orchestration</div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_bar()

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