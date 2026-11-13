import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
import recordlinkage
import missingno as msng
from textblob import TextBlob

# --- Drop-in PyJanitor Replacements ---
def clean_names(df: pd.DataFrame, remove_special: bool = True) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.astype(str).str.strip().str.lower()
    if remove_special:
        cols = cols.str.replace(r"[^a-z0-9_]", "_", regex=True)
    df.columns = cols.str.replace(r"_+", "_", regex=True).str.strip("_")
    return df

def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all", axis=0).dropna(how="all", axis=1)

# --- Page Config & Dark Theme Styling ---
st.set_page_config(page_title="NYC Taxi Parquet Studio", page_icon="🚕", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        color: #f8fafc;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #facc15 !important; }
</style>
""", unsafe_allow_html=True)

PAYMENT_MAP = {
    0: "Flex / E-Hail",
    1: "Credit Card",
    2: "Cash",
    3: "No Charge (Free)",
    4: "Dispute (Complaint)",
    5: "Unknown",
    6: "Voided Trip"
}

RATE_MAP = {
    1: "Standard Rate",
    2: "JFK Airport",
    3: "Newark Airport",
    4: "Nassau / Westchester",
    5: "Negotiated Fare",
    6: "Group Ride",
    99: "Unspecified"
}

# --- 1. Load & Clean Parquet ---
@st.cache_data
def load_parquet_data(path: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    df = raw.pipe(clean_names, remove_special=True).pipe(remove_empty)

    # Datetimes are already <M8[us]; coerce safely just in case
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime"]).copy()

    # Date & Duration Features
    df["solo_fecha"] = df["tpep_pickup_datetime"].dt.date
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0

    # Filter out invalid/corrupt meter rows using NumPy
    valid_mask = (
        (df["trip_distance"] > 0) &
        (df["total_amount"] > 0) &
        (df["duration_min"].between(0.5, 360))
    )
    df = df[valid_mask].copy()

    # Decode categorical IDs & compute tip %
    df["payment_label"] = df["payment_type"].map(PAYMENT_MAP).fillna("Other")
    df["rate_label"] = df["ratecodeid"].map(RATE_MAP).fillna("Missing RateCode")
    df["tip_pct"] = np.where(df["fare_amount"] > 0, (df["tip_amount"] / df["fare_amount"]) * 100.0, 0.0)
    df["has_cbd_fee"] = df["cbd_congestion_fee"].fillna(0) > 0

    # StandardScaler (escaleno) on distance & total fare
    escaleno = StandardScaler()
    df[["distance_zscore", "fare_zscore"]] = escaleno.fit_transform(
        df[["trip_distance", "total_amount"]].fillna(0)
    )

    # TextBlob sentiment on synthesized trip descriptor (e.g. "Dispute (Complaint) - JFK Airport")
    df["trip_descriptor"] = df["payment_label"] + " - " + df["rate_label"]
    unique_desc = {d: round(TextBlob(d).sentiment.polarity, 3) for d in df["trip_descriptor"].unique()}
    df["descriptor_polarity"] = df["trip_descriptor"].map(unique_desc)

    return df

# Auto-detect first .parquet file in current directory or use sidebar input
parquet_files = sorted(glob.glob("*.parquet"))
default_file = parquet_files[0] if parquet_files else "./yellow_tripdata_2023-01.parquet"

st.sidebar.header("Parquet Pipeline Controls")
file_path = st.sidebar.text_input("Parquet File Path", value=default_file)

try:
    df_base = load_parquet_data(file_path)
except Exception as e:
    st.error(f"Could not read Parquet file at `{file_path}`: {e}")
    st.stop()

selected_payments = st.sidebar.multiselect(
    "Payment Type",
    options=df_base["payment_label"].unique().tolist(),
    default=df_base["payment_label"].unique().tolist()
)
z_thresh = st.sidebar.slider("Outlier Z-Score Threshold (StandardScaler)", 1.0, 6.0, 3.0, 0.25)
linkage_sample_n = st.sidebar.slider("RecordLinkage Sample Cap (rows)", 1000, 25000, 5000, 1000)

df_filtered = df_base[df_base["payment_label"].isin(selected_payments)].copy()
df_filtered["is_outlier"] = (df_filtered["distance_zscore"] >= z_thresh) | (df_filtered["fare_zscore"] >= z_thresh)

# --- 2. RecordLinkage: Detect Potential Double-Billed / Duplicate Rides ---
df_link = df_filtered.head(linkage_sample_n).reset_index(drop=True)
indexador = recordlinkage.Index()
indexador.block(["solo_fecha", "pulocationid", "dolocationid"])
candidatos = indexador.index(df_link)

cubero = recordlinkage.Compare()
cubero.exact("payment_type", "payment_type", label="same_payment")
cubero.exact("passenger_count", "passenger_count", label="same_pax")
cubero.numeric("total_amount", "total_amount", method="gauss", offset=0.25, scale=1.0, label="fare_match")
cubero.numeric("trip_distance", "trip_distance", method="gauss", offset=0.1, scale=0.5, label="dist_match")

if len(candidatos) > 0:
    features = cubero.compute(candidatos, df_link)
    duplicate_pairs = features[features.sum(axis=1) >= 3.5]
else:
    duplicate_pairs = pd.DataFrame()

# --- Header & KPIs ---
st.title("🚕 TLC Yellow Taxi Parquet Telemetry & Audit Studio")
st.caption("PyJanitor Cleanup • NumPy Anomaly Filtering • StandardScaler • Missingno • RecordLinkage • TextBlob")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Valid Trips", f"{len(df_filtered):,}", f"CBD Fee Trips: {df_filtered['has_cbd_fee'].sum():,}")
k2.metric("Gross Revenue", f"${df_filtered['total_amount'].sum():,.2f}", f"Avg Tip: {df_filtered['tip_pct'].mean():.1f}%")
k3.metric("Fare/Dist Outliers", f"{df_filtered['is_outlier'].sum():,}", f"Z-score ≥ {z_thresh}σ")
k4.metric("Suspected Duplicate Charges", f"{len(duplicate_pairs):,}", f"In first {len(df_link):,} rows")

st.divider()

# --- Visual Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Fares, Tips & Outliers", "🔍 Missingno Null Audit", "🔗 RecordLinkage Double-Charge Check"])

with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Hourly Demand & Revenue (`pickup_hour`)")
        hourly = df_filtered.groupby("pickup_hour")["total_amount"].sum()
        st.area_chart(hourly, color="#facc15", height=310)

    with c2:
        st.subheader("Revenue by Payment Type")
        pay_rev = df_filtered.groupby("payment_label")["total_amount"].sum().sort_values()
        st.bar_chart(pay_rev, horizontal=True, color="#38bdf8", height=310)

    st.subheader("Trip Distance vs. Total Amount (Sampled 3,000 pts — Z-Score Outliers Highlighted)")
    scatter_df = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
    st.scatter_chart(
        scatter_df,
        x="trip_distance",
        y="total_amount",
        color="is_outlier",
        height=340
    )

with tab2:
    st.subheader("Missing Value Matrix (`missingno` on 2,500-row sample)")
    st.write("In TLC Parquet files, `passenger_count`, `ratecodeid`, `store_and_fwd_flag`, `congestion_surcharge`, and `airport_fee` often share identical missing blocks on Flex/e-hail rides:")
    audit_cols = [
        "vendorid", "tpep_pickup_datetime", "passenger_count", "trip_distance",
        "ratecodeid", "store_and_fwd_flag", "pulocationid", "dolocationid",
        "payment_type", "fare_amount", "tip_amount", "total_amount",
        "congestion_surcharge", "airport_fee", "cbd_congestion_fee"
    ]
    sample_msng = df_base[audit_cols].sample(min(2500, len(df_base)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    msng.matrix(sample_msng, ax=ax, sparkline=False, color=(0.98, 0.80, 0.08))
    ax.tick_params(colors="#e2e8f0", labelsize=8)
    st.pyplot(fig)

with tab3:
    st.subheader("Same-Day, Same-Zone Duplicate Ride Candidates (`cubero`)")
    st.write(f"Blocked on `['solo_fecha', 'pulocationid', 'dolocationid']` across **{len(df_link):,}** trips → **{len(candidatos):,}** zone-matched pairs → **{len(duplicate_pairs):,}** high-confidence duplicates:")
    if not duplicate_pairs.empty:
        i1 = duplicate_pairs.index.get_level_values(0)
        i2 = duplicate_pairs.index.get_level_values(1)
        dup_view = pd.DataFrame({
            "Pickup A": df_link.loc[i1, "tpep_pickup_datetime"].values,
            "Pickup B": df_link.loc[i2, "tpep_pickup_datetime"].values,
            "PU → DO Zone": (df_link.loc[i1, "pulocationid"].astype(str) + " → " + df_link.loc[i1, "dolocationid"].astype(str)).values,
            "Miles A": df_link.loc[i1, "trip_distance"].values,
            "Miles B": df_link.loc[i2, "trip_distance"].values,
            "Total A ($)": df_link.loc[i1, "total_amount"].values,
            "Total B ($)": df_link.loc[i2, "total_amount"].values,
            "Descriptor": df_link.loc[i1, "trip_descriptor"].values,
            "TextBlob Score": df_link.loc[i1, "descriptor_polarity"].values,
        })
        st.dataframe(dup_view, use_container_width=True, hide_index=True)
    else:
        st.info("No near-identical duplicate trips found in the sampled slice. Increase the sample slider in the sidebar to scan more rows.")
