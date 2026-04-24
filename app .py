import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CC Info · ML Dataset Explorer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fonts & base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1c2c;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #c9cfe0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1a1c2c !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f3f7;
        border-radius: 8px;
        padding: 0.4rem 1.1rem;
        font-weight: 500;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1c2c !important;
        color: #ffffff !important;
    }

    /* Dataframe */
    .dataframe thead th { background-color: #f1f3f7 !important; font-weight: 600; }

    /* Divider */
    hr { border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "cc_info.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Mask card number for display
    df["card_display"] = df["credit_card"].astype(str).str[:4] + " **** **** " + df["credit_card"].astype(str).str[-4:]
    return df

df_raw = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 CC Info Explorer")
    st.markdown("_ML Dataset · Credit Card Records_")
    st.divider()

    st.markdown("### 🔍 Filters")
    states = sorted(df_raw["state"].unique())
    sel_states = st.multiselect("State(s)", states, placeholder="All states")

    limit_min, limit_max = int(df_raw["credit_card_limit"].min()), int(df_raw["credit_card_limit"].max())
    sel_range = st.slider(
        "Credit Limit Range ($)",
        min_value=limit_min, max_value=limit_max,
        value=(limit_min, limit_max), step=1000,
        format="$%d"
    )

    st.divider()
    st.caption(f"Dataset: **{len(df_raw):,} records** · 5 features")
    st.caption("Built with Streamlit 🎈")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_states:
    df = df[df["state"].isin(sel_states)]
df = df[(df["credit_card_limit"] >= sel_range[0]) & (df["credit_card_limit"] <= sel_range[1])]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Credit Card Dataset Explorer")
st.markdown(
    f"Showing **{len(df):,}** of **{len(df_raw):,}** records · "
    f"{'All states' if not sel_states else ', '.join(sel_states)}"
)
st.divider()


# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records", f"{len(df):,}")
k2.metric("Avg Credit Limit", f"${df['credit_card_limit'].mean():,.0f}")
k3.metric("Unique States", df["state"].nunique())
k4.metric("Unique Cities", df["city"].nunique())

st.markdown("")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Geographic", "📋 Data Table", "🤖 ML Insights"])

CHART_TEMPLATE = "plotly_white"
PRIMARY   = "#4f46e5"
SECONDARY = "#818cf8"


# ─── Tab 1 · Overview ─────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Credit Limit Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="credit_card_limit", nbins=30,
            color_discrete_sequence=[PRIMARY],
            template=CHART_TEMPLATE,
            labels={"credit_card_limit": "Credit Limit ($)", "count": "Frequency"},
        )
        fig_hist.update_traces(marker_line_width=0.5, marker_line_color="white")
        fig_hist.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_tickprefix="$", xaxis_tickformat=",",
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Credit Limit · Box Plot</p>', unsafe_allow_html=True)
        fig_box = px.box(
            df, y="credit_card_limit",
            color_discrete_sequence=[SECONDARY],
            template=CHART_TEMPLATE,
            labels={"credit_card_limit": "Credit Limit ($)"},
        )
        fig_box.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            yaxis_tickprefix="$", yaxis_tickformat=",",
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Top states bar
    st.markdown('<p class="section-header">Avg Credit Limit by State (Top 15)</p>', unsafe_allow_html=True)
    state_avg = (
        df.groupby("state")["credit_card_limit"]
        .mean()
        .reset_index()
        .sort_values("credit_card_limit", ascending=False)
        .head(15)
    )
    fig_bar = px.bar(
        state_avg, x="state", y="credit_card_limit",
        color="credit_card_limit", color_continuous_scale="Blues",
        template=CHART_TEMPLATE,
        labels={"credit_card_limit": "Avg Credit Limit ($)", "state": "State"},
    )
    fig_bar.update_layout(
        height=350, margin=dict(l=0, r=0, t=20, b=0),
        yaxis_tickprefix="$", yaxis_tickformat=",",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── Tab 2 · Geographic ───────────────────────────────────────────────────────
with tab2:
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown('<p class="section-header">Record Count by State</p>', unsafe_allow_html=True)
        state_count = df.groupby("state").size().reset_index(name="count")
        fig_choropleth = px.choropleth(
            state_count,
            locations="state", locationmode="USA-states",
            color="count", scope="usa",
            color_continuous_scale="Purples",
            template=CHART_TEMPLATE,
            labels={"count": "# Records"},
        )
        fig_choropleth.update_layout(
            height=370, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_choropleth, use_container_width=True)

    with col_g2:
        st.markdown('<p class="section-header">Avg Credit Limit by State</p>', unsafe_allow_html=True)
        state_avg_map = df.groupby("state")["credit_card_limit"].mean().reset_index()
        fig_choro2 = px.choropleth(
            state_avg_map,
            locations="state", locationmode="USA-states",
            color="credit_card_limit", scope="usa",
            color_continuous_scale="Blues",
            template=CHART_TEMPLATE,
            labels={"credit_card_limit": "Avg Limit ($)"},
        )
        fig_choro2.update_layout(
            height=370, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_choro2, use_container_width=True)

    # Top cities
    st.markdown('<p class="section-header">Top 10 Cities by Record Count</p>', unsafe_allow_html=True)
    city_count = df.groupby(["city", "state"]).size().reset_index(name="count").sort_values("count", ascending=True).tail(10)
    fig_city = px.bar(
        city_count, y="city", x="count", orientation="h",
        color="count", color_continuous_scale="Purples",
        template=CHART_TEMPLATE,
        labels={"count": "# Records", "city": "City"},
        text="count",
    )
    fig_city.update_traces(textposition="outside")
    fig_city.update_layout(
        height=340, margin=dict(l=0, r=10, t=20, b=0),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_city, use_container_width=True)


# ─── Tab 3 · Data Table ───────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Raw Dataset (filtered)</p>', unsafe_allow_html=True)

    search = st.text_input("🔎 Search city or state", placeholder="e.g. New York or NY")
    display_df = df.copy()
    if search:
        mask = (
            display_df["city"].str.contains(search, case=False, na=False) |
            display_df["state"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    show_cols = ["card_display", "city", "state", "zipcode", "credit_card_limit"]
    rename_map = {
        "card_display": "Card Number",
        "city": "City", "state": "State",
        "zipcode": "ZIP Code", "credit_card_limit": "Credit Limit ($)"
    }

    st.dataframe(
        display_df[show_cols].rename(columns=rename_map).reset_index(drop=True),
        use_container_width=True, height=480
    )
    st.caption(f"{len(display_df):,} rows shown")

    # Download
    csv_bytes = display_df.drop(columns=["card_display"]).to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered CSV", csv_bytes, "cc_info_filtered.csv", "text/csv")


# ─── Tab 4 · ML Insights ──────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Feature Summary for ML Training</p>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Numeric Feature Stats**")
        num_stats = df[["credit_card_limit", "zipcode"]].describe().T
        num_stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        st.dataframe(num_stats.style.format("{:.1f}"), use_container_width=True)

    with col_m2:
        st.markdown("**Categorical Feature Cardinality**")
        cat_info = pd.DataFrame({
            "Feature": ["state", "city", "zipcode"],
            "Unique Values": [df["state"].nunique(), df["city"].nunique(), df["zipcode"].nunique()],
            "Top Value": [
                df["state"].value_counts().idxmax(),
                df["city"].value_counts().idxmax(),
                str(df["zipcode"].value_counts().idxmax()),
            ],
            "Top Freq": [
                df["state"].value_counts().max(),
                df["city"].value_counts().max(),
                df["zipcode"].value_counts().max(),
            ],
        })
        st.dataframe(cat_info, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown('<p class="section-header">Credit Limit · Percentile Ladder</p>', unsafe_allow_html=True)

    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    pct_vals = df["credit_card_limit"].quantile([p / 100 for p in pcts]).values
    fig_pct = go.Figure(go.Scatter(
        x=pcts, y=pct_vals,
        mode="lines+markers",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=8, color=PRIMARY),
        fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
    ))
    fig_pct.update_layout(
        xaxis_title="Percentile", yaxis_title="Credit Limit ($)",
        template=CHART_TEMPLATE, height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_tickprefix="$", yaxis_tickformat=",",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("")
    st.info(
        "💡 **ML Notes** — `credit_card_limit` is the most informative numeric feature. "
        "`state` and `city` need target-encoding or one-hot for tree-based models. "
        "`zipcode` can serve as a geographic proxy. No null values detected.",
        icon=None
    )
