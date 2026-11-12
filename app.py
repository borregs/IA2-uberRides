import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
import recordlinkage
import missingno as msng
from textblob import TextBlob

# Drop-in replacements for pyjanitor (works with any pandas version)
def clean_names(df: pd.DataFrame, remove_special: bool = True) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.astype(str).str.strip().str.lower()
    if remove_special:
        cols = cols.str.replace(r"[^a-z0-9_]", "_", regex=True)
    df.columns = cols.str.replace(r"_+", "_", regex=True).str.strip("_")
    return df

def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all", axis=0).dropna(how="all", axis=1)


# --- Page Config & Custom Styling ---
st.set_page_config(page_title="Uber Trips Data Studio", page_icon="🚘", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        color: #f8fafc;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & PyJanitor Cleanup ---
@st.cache_data
def load_and_clean(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.pipe(clean_names, remove_special=True).pipe(remove_empty)

    # Parse dates safely (drops the trailing 'Totals' row in data.csv)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["start_date"]).copy()

    # Feature engineering (NumPy + Datetime)
    df["solo_fecha"] = df["start_date"].dt.date
    df["duration_min"] = (df["end_date"] - df["start_date"]).dt.total_seconds() / 60.0
    df["duration_min"] = np.where(df["duration_min"] <= 0, np.nan, df["duration_min"])

    # StandardScaler (escaleno) on miles
    escaleno = StandardScaler()
    df["miles_zscore"] = escaleno.fit_transform(df[["miles"]])
    df["trip_type"] = np.where(df["miles_zscore"] > 2.0, "Outlier (>2σ)", "Standard")

    # TextBlob sentiment on purpose
    df["purpose_clean"] = df["purpose"].fillna("Unspecified")
    df["purpose_polarity"] = df["purpose_clean"].apply(lambda x: round(TextBlob(str(x)).sentiment.polarity, 3))

    return df

df_base = load_and_clean("./data.csv")

# --- Sidebar Controls ---
st.sidebar.header("Pipeline Filters")
selected_cats = st.sidebar.multiselect(
    "Category",
    options=df_base["category"].dropna().unique().tolist(),
    default=df_base["category"].dropna().unique().tolist()
)
z_thresh = st.sidebar.slider("Outlier Z-Score Threshold (StandardScaler)", 0.5, 5.0, 2.0, 0.25)

df_filtered = df_base[df_base["category"].isin(selected_cats)].copy()
df_filtered["is_outlier"] = df_filtered["miles_zscore"] >= z_thresh

# --- 2. RecordLinkage Blocking on 'solo_fecha' ---
indexador_fechas = recordlinkage.Index()
indexador_fechas.block("solo_fecha")
candidatos_fechas = indexador_fechas.index(df_filtered)

cubero = recordlinkage.Compare()
cubero.exact("start", "start", label="same_origin")
cubero.exact("stop", "stop", label="same_dest")
cubero.string("purpose_clean", "purpose_clean", method="jarowinkler", threshold=0.85, label="purpose_sim")
features = cubero.compute(candidatos_fechas, df_filtered)
strong_matches = features[features.sum(axis=1) >= 2]

# --- Header & KPI Cards ---
st.title("🚘 Trip Telemetry, Cleanup & Record Linkage Explorer")
st.caption("Powered by PyJanitor • StandardScaler • Missingno • RecordLinkage • TextBlob")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Clean Trips", f"{len(df_filtered):,}", f"{len(df_base) - len(df_filtered)} filtered out")
k2.metric("Total Distance", f"{df_filtered['miles'].sum():,.1f} mi", f"Avg {df_filtered['miles'].mean():.1f} mi/trip")
k3.metric("Distance Outliers", f"{df_filtered['is_outlier'].sum()}", f"Z-score ≥ {z_thresh}σ")
k4.metric("Same-Day Linked Pairs", f"{len(strong_matches):,}", f"From {len(candidatos_fechas):,} date-blocked pairs")

st.divider()

# --- Tabs for Visual Exploration ---
tab1, tab2, tab3 = st.tabs(["📈 Trip Trends & Outliers", "🔍 Missingno Audit", "🔗 RecordLinkage Inspector"])

with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Daily Miles Timeline (Grouped by `solo_fecha`)")
        daily = (
            df_filtered.groupby("solo_fecha", as_index=False)
            .agg(total_miles=("miles", "sum"), trips=("miles", "count"))
            .set_index("solo_fecha")
        )
        st.area_chart(daily["total_miles"], color="#38bdf8", height=320)

    with col_right:
        st.subheader("Miles by Trip Purpose")
        purpose_summary = (
            df_filtered.groupby("purpose_clean")["miles"]
            .sum()
            .sort_values(ascending=True)
        )
        st.bar_chart(purpose_summary, horizontal=True, color="#10b981", height=320)

    st.subheader("Distance vs. Duration — StandardScaler Outlier Detection")
    st.scatter_chart(
        df_filtered.dropna(subset=["duration_min"]),
        x="duration_min",
        y="miles",
        color="is_outlier",
        size="miles",
        height=340
    )

with tab2:
    st.subheader("Missing Value Matrix (`missingno` before imputation)")
    st.write("Notice how `purpose` carries ~43% missing entries while core GPS/timestamp fields remain intact:")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0f172a")
    msng.matrix(df_filtered[["start_date", "end_date", "category", "start", "stop", "miles", "purpose"]], ax=ax, sparkline=False, color=(0.22, 0.74, 0.97))
    ax.tick_params(colors="#e2e8f0")
    st.pyplot(fig)

with tab3:
    st.subheader("Same-Day Candidate Pairs (`indexador_fechas` + `cubero`)")
    st.write(f"Blocking on `solo_fecha` narrowed comparisons down to **{len(candidatos_fechas):,}** candidate pairs, yielding **{len(strong_matches):,}** multi-attribute matches:")
    if not strong_matches.empty:
        idx_1 = strong_matches.index.get_level_values(0)
        idx_2 = strong_matches.index.get_level_values(1)
        linked_view = pd.DataFrame({
            "Date": df_filtered.loc[idx_1, "solo_fecha"].values,
            "Trip A Route": (df_filtered.loc[idx_1, "start"] + " → " + df_filtered.loc[idx_1, "stop"]).values,
            "Trip B Route": (df_filtered.loc[idx_2, "start"] + " → " + df_filtered.loc[idx_2, "stop"]).values,
            "Miles A": df_filtered.loc[idx_1, "miles"].values,
            "Miles B": df_filtered.loc[idx_2, "miles"].values,
            "Purpose": df_filtered.loc[idx_1, "purpose_clean"].values,
            "TextBlob Polarity": df_filtered.loc[idx_1, "purpose_polarity"].values,
        })
        st.dataframe(linked_view, use_container_width=True, hide_index=True)
    else:
        st.info("No multi-attribute matches found under current filter selection.")
