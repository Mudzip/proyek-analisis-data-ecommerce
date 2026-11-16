import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Polygon
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MAIN, ACCENT, GREY, TEXT = "#2a78d6", "#eb6834", "#c9c8c2", "#52514e"

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": "#d9d8d3",
    "grid.color": "#ecebe7",
})

DELAY_BINS = [-np.inf, 0, 3, 7, 14, 30, np.inf]
DELAY_LABELS = ["Tepat waktu", "Telat 1-3 hari", "Telat 4-7 hari",
                "Telat 8-14 hari", "Telat 15-30 hari", "Telat >30 hari"]
SEGMENT_ORDER = ["Champions", "Loyal", "Baru – Nilai Tinggi", "Baru",
                 "Perlu Perhatian", "Berisiko – Nilai Tinggi", "Hibernating"]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "main_data.csv", parse_dates=["order_purchase_timestamp"])
    df["is_late"] = df["is_late"].astype(bool)
    return df


@st.cache_data
def load_geojson():
    with open(BASE_DIR / "brazil_states.geojson", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: menyiapkan DataFrame untuk tiap visualisasi
# ---------------------------------------------------------------------------
def create_order_level_df(df):
    return df.drop_duplicates("order_id")


def create_monthly_df(df):
    return (df.assign(month=df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
              .groupby("month")
              .agg(order_count=("order_id", "nunique"), revenue=("price", "sum"))
              .reset_index())


def create_category_df(df):
    category_df = (df.groupby("product_category_name_english")
                     .agg(revenue=("price", "sum"), order_count=("order_id", "nunique"),
                          avg_price=("price", "mean"))
                     .sort_values("revenue", ascending=False)
                     .reset_index())
    category_df["revenue_share_pct"] = category_df["revenue"] / category_df["revenue"].sum() * 100
    return category_df


def create_delay_df(order_df):
    reviewed = order_df.dropna(subset=["review_score", "delay_days"]).copy()
    reviewed["delay_group"] = pd.cut(reviewed["delay_days"], bins=DELAY_BINS, labels=DELAY_LABELS)
    reviewed["is_bad_review"] = reviewed["review_score"] <= 2
    delay_df = (reviewed.groupby("delay_group", observed=False)
                        .agg(order_count=("order_id", "count"), avg_review=("review_score", "mean"),
                             bad_review_pct=("is_bad_review", "mean")))
    delay_df["bad_review_pct"] *= 100
    return delay_df


def create_state_df(df, order_df):
    state_df = (df.groupby("customer_state")
                  .agg(customer_count=("customer_unique_id", "nunique"), revenue=("price", "sum"))
                  .join(order_df.groupby("customer_state")
                        .agg(avg_delivery_days=("delivery_time_days", "mean"),
                             late_pct=("is_late", "mean"), avg_review=("review_score", "mean"))))
    state_df["late_pct"] *= 100
    return state_df.sort_values("revenue", ascending=False)


def quantile_score(series, labels):
    """Skor kuantil 1–5 seperti di notebook; jika data terlalu sedikit/kembar, pakai peringkat."""
    try:
        return pd.qcut(series, q=5, labels=labels).astype(int)
    except ValueError:
        return pd.qcut(series.rank(method="first"), q=5, labels=labels).astype(int)


def create_rfm_df(df, reference_date):
    rfm_df = (df.groupby("customer_unique_id", as_index=False)
                .agg(last_order=("order_purchase_timestamp", "max"),
                     frequency=("order_id", "nunique"), monetary=("price", "sum")))
    rfm_df["recency"] = (reference_date - rfm_df["last_order"]).dt.days

    # Manual grouping: skor R & M berbasis kuantil, skor F berbasis aturan bisnis
    rfm_df["r_score"] = quantile_score(rfm_df["recency"], labels=[5, 4, 3, 2, 1])
    rfm_df["f_score"] = pd.cut(rfm_df["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)
    rfm_df["m_score"] = quantile_score(rfm_df["monetary"], labels=[1, 2, 3, 4, 5])

    conditions = [
        (rfm_df["f_score"] >= 2) & (rfm_df["r_score"] >= 4),
        (rfm_df["f_score"] >= 2),
        (rfm_df["r_score"] >= 4) & (rfm_df["m_score"] >= 4),
        (rfm_df["r_score"] >= 4),
        (rfm_df["r_score"] == 3),
        (rfm_df["m_score"] >= 4),
    ]
    rfm_df["segment"] = np.select(conditions, SEGMENT_ORDER[:6], default="Hibernating")
    return rfm_df


def polygon_centroid(ring):
    pts = np.array(ring)
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    return ((x + np.roll(x, -1)) * cross).sum() / (6 * area), ((y + np.roll(y, -1)) * cross).sum() / (6 * area)


def draw_choropleth(values, label, log=False):
    geo = load_geojson()
    fig, ax = plt.subplots(figsize=(7, 7))
    cm = plt.get_cmap("Blues")
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmin, vmax = vmin * 0.9, vmax * 1.1 + 1e-9
    norm = LogNorm(max(vmin, 1e-3), vmax) if log else Normalize(vmin, vmax)
    for feat in geo["features"]:
        uf = feat["properties"]["sigla"]
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        has_value = uf in values.index and pd.notna(values[uf])
        face = cm(norm(values[uf])) if has_value else "#f4f4f2"
        for poly in polys:
            ax.add_patch(Polygon(poly[0], closed=True, facecolor=face, edgecolor="white", linewidth=0.8))
        largest = max(polys, key=lambda p: abs(np.ptp(np.array(p[0]), axis=0).prod()))
        cx, cy = polygon_centroid(largest[0])
        dark = has_value and norm(values[uf]) > 0.6
        ax.text(cx, cy, uf, ha="center", va="center", fontsize=7, color="white" if dark else "#0b0b0b")
    ax.set_xlim(-75, -33)
    ax.set_ylim(-34.5, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    if log:
        cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: f"{v:,.0f}"))
    return fig


def fmt_brl(value):
    if value >= 1e6:
        return f"R${value / 1e6:,.2f} jt"
    if value >= 1e3:
        return f"R${value / 1e3:,.1f}K"
    return f"R${value:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar & filter
# ---------------------------------------------------------------------------
all_df = load_data()
# Periode analisis sama dengan notebook: 1 Jan 2017 – 31 Ags 2018
min_date = pd.Timestamp("2017-01-01").date()
max_date = pd.Timestamp("2018-08-31").date()

with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.caption("Proyek Analisis Data – Dicoding")
    date_range = st.date_input("Rentang waktu pembelian", value=[min_date, max_date],
                               min_value=min_date, max_value=max_date)
    states = sorted(all_df["customer_state"].dropna().unique())
    selected_states = st.multiselect("State pelanggan", options=states, default=[],
                                     placeholder="Semua state")
    st.divider()
    st.caption("Data: pesanan berstatus *delivered*, Jan 2017 – Ags 2018. "
               "Revenue = total harga barang (tanpa ongkir), dalam Real Brasil (R$).")

if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan akhir pada sidebar.")
    st.stop()
start_date, end_date = date_range

main_df = all_df[(all_df["order_purchase_timestamp"] >= pd.Timestamp(start_date))
                 & (all_df["order_purchase_timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))]
if selected_states:
    main_df = main_df[main_df["customer_state"].isin(selected_states)]

if main_df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Silakan ubah rentang waktu atau state.")
    st.stop()

order_df = create_order_level_df(main_df)

# ---------------------------------------------------------------------------
# Header & KPI
# ---------------------------------------------------------------------------
st.title("E-Commerce Public Dataset Dashboard")
st.caption(f"Periode {start_date:%d %b %Y} – {end_date:%d %b %Y}"
           + (f" · State: {', '.join(selected_states)}" if selected_states else " · Semua state"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pesanan", f"{order_df['order_id'].nunique():,}")
col2.metric("Total revenue", fmt_brl(main_df["price"].sum()))
col3.metric("Rata-rata review", f"{order_df['review_score'].mean():.2f} / 5")
col4.metric("Pesanan terlambat", f"{order_df['is_late'].mean():.1%}")

tab_trend, tab_cat, tab_delivery, tab_rfm, tab_geo = st.tabs(
    ["📈 Tren Penjualan", "🏷️ Kategori Produk", "🚚 Pengiriman & Review", "👥 RFM & Segmen", "🗺️ Geografis"])

# ---------------------------------------------------------------------------
# Tab 1 – Tren penjualan (Pertanyaan 1)
# ---------------------------------------------------------------------------
with tab_trend:
    monthly_df = create_monthly_df(main_df)
    st.subheader("Tren pesanan dan revenue bulanan")
    if len(monthly_df) < 2:
        st.info("Pilih rentang waktu minimal dua bulan untuk melihat tren.")
    peak = monthly_df.loc[monthly_df["revenue"].idxmax()]
    st.markdown(f"Revenue tertinggi pada **{peak['month']:%B %Y}**: **{fmt_brl(peak['revenue'])}** "
                f"dari **{int(peak['order_count']):,}** pesanan.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, col, title in [(axes[0], "order_count", "Jumlah pesanan per bulan"),
                           (axes[1], "revenue", "Revenue per bulan")]:
        ax.plot(monthly_df["month"], monthly_df[col], color=MAIN, linewidth=2, marker="o", markersize=5)
        ax.scatter(peak["month"], peak[col], s=90, color=ACCENT, zorder=5, edgecolor="white", linewidth=2)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"R${x / 1e3:,.0f}K"))
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat tabel data bulanan"):
        st.dataframe(monthly_df.assign(month=monthly_df["month"].dt.strftime("%Y-%m")),
                     hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 – Kategori produk (Pertanyaan 2)
# ---------------------------------------------------------------------------
with tab_cat:
    category_df = create_category_df(main_df)
    n_top = st.slider("Jumlah kategori yang ditampilkan", min_value=5, max_value=15, value=10)
    top_share = category_df.head(n_top)["revenue_share_pct"].sum()
    st.subheader(f"{n_top} kategori teratas menyumbang {top_share:.1f}% revenue")

    top_df = category_df.head(n_top).iloc[::-1]
    bottom_df = category_df.tail(n_top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * n_top + 2))
    axes[0].barh(top_df["product_category_name_english"], top_df["revenue"],
                 color=[GREY] * (len(top_df) - 1) + [MAIN], height=0.7)
    for y, (rev, share) in enumerate(zip(top_df["revenue"], top_df["revenue_share_pct"])):
        axes[0].text(rev, y, f" {fmt_brl(rev)} ({share:.1f}%)", va="center", fontsize=9, color=TEXT)
    axes[0].set_title("Revenue tertinggi")
    axes[0].set_xlim(0, top_df["revenue"].max() * 1.45)
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: fmt_brl(x)))
    axes[0].xaxis.set_major_locator(mtick.MaxNLocator(4))

    axes[1].barh(bottom_df["product_category_name_english"], bottom_df["revenue"],
                 color=[ACCENT] + [GREY] * (len(bottom_df) - 1), height=0.7)
    for y, rev in enumerate(bottom_df["revenue"]):
        axes[1].text(rev, y, f" {fmt_brl(rev)}", va="center", fontsize=9, color=TEXT)
    axes[1].set_title("Revenue terendah")
    axes[1].set_xlim(0, bottom_df["revenue"].max() * 1.35)
    axes[1].xaxis.set_major_locator(mtick.MaxNLocator(4))
    for ax in axes:
        ax.grid(axis="y", visible=False)
    plt.tight_layout(w_pad=4)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat tabel semua kategori"):
        st.dataframe(category_df, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 – Pengiriman & review (Pertanyaan 3)
# ---------------------------------------------------------------------------
with tab_delivery:
    delay_df = create_delay_df(order_df)
    reviewed = order_df.dropna(subset=["review_score"])
    on_time = reviewed.loc[~reviewed["is_late"], "review_score"].mean()
    late = reviewed.loc[reviewed["is_late"], "review_score"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Skor rata-rata – tepat waktu", f"{on_time:.2f}")
    c2.metric("Skor rata-rata – terlambat", f"{late:.2f}" if pd.notna(late) else "–",
              delta=f"{late - on_time:.2f} poin" if pd.notna(late) else None)
    c3.metric("Rata-rata lama pengiriman", f"{order_df['delivery_time_days'].mean():.1f} hari")

    st.subheader("Makin lama terlambat, makin rendah review score")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    labels = [lbl.replace(" ", "\n", 1) for lbl in delay_df.index.astype(str)]
    colors = [MAIN] + [ACCENT] * (len(delay_df) - 1)
    axes[0].bar(labels, delay_df["avg_review"].fillna(0), color=colors, width=0.65)
    for i, (score, n) in enumerate(zip(delay_df["avg_review"], delay_df["order_count"])):
        if pd.notna(score):
            axes[0].text(i, score + 0.06, f"{score:.2f}", ha="center", fontsize=10, color=TEXT)
        axes[0].text(i, 0.12, f"n={n:,}", ha="center", fontsize=8, color="white")
    axes[0].set_ylim(0, 5)
    axes[0].set_title("Rata-rata review score")
    axes[1].bar(labels, delay_df["bad_review_pct"].fillna(0), color=colors, width=0.65)
    for i, pct in enumerate(delay_df["bad_review_pct"]):
        if pd.notna(pct):
            axes[1].text(i, pct + 1.2, f"{pct:.0f}%", ha="center", fontsize=10, color=TEXT)
    axes[1].set_ylim(0, 100)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_title("Persentase ulasan buruk (skor 1–2)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Kelompok keterlambatan dibuat dengan teknik binning (pd.cut) pada selisih tanggal "
               "diterima dengan tanggal estimasi.")

# ---------------------------------------------------------------------------
# Tab 4 – RFM & segmentasi
# ---------------------------------------------------------------------------
with tab_rfm:
    reference_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    if main_df["customer_unique_id"].nunique() < 5:
        st.info("Data pelanggan terlalu sedikit untuk analisis RFM. Perluas filter.")
        st.stop()
    rfm_df = create_rfm_df(main_df, reference_date)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata recency", f"{rfm_df['recency'].mean():.0f} hari")
    c2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f} pesanan")
    c3.metric("Rata-rata monetary", fmt_brl(rfm_df["monetary"].mean()))

    st.subheader("Pelanggan terbaik berdasarkan parameter RFM")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, data, col, title in [
        (axes[0], rfm_df.nsmallest(5, "recency"), "recency", "Recency (hari)"),
        (axes[1], rfm_df.nlargest(5, "frequency"), "frequency", "Frequency (pesanan)"),
        (axes[2], rfm_df.nlargest(5, "monetary"), "monetary", "Monetary (R$)"),
    ]:
        short_id = data["customer_unique_id"].str[:8] + "…"
        ax.barh(short_id.iloc[::-1], data[col].iloc[::-1], color=MAIN, height=0.6)
        for y, v in enumerate(data[col].iloc[::-1]):
            ax.text(v, y, f" {v:,.0f}", va="center", fontsize=9, color=TEXT)
        ax.set_xlim(0, max(data[col].max() * 1.3, 1))
        ax.xaxis.set_major_locator(mtick.MaxNLocator(4, integer=True))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Segmentasi pelanggan (manual grouping)")
    segment_df = (rfm_df.groupby("segment")
                        .agg(customer_count=("customer_unique_id", "count"), revenue=("monetary", "sum"))
                        .reindex(SEGMENT_ORDER).fillna(0))
    segment_df["customer_pct"] = segment_df["customer_count"] / segment_df["customer_count"].sum() * 100
    segment_df["revenue_pct"] = segment_df["revenue"] / segment_df["revenue"].sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    plot_df = segment_df.iloc[::-1]
    colors = [ACCENT if s == "Berisiko – Nilai Tinggi" else MAIN for s in plot_df.index]
    xmax = max(segment_df["customer_pct"].max(), segment_df["revenue_pct"].max()) * 1.2
    for ax, col, title in [(axes[0], "customer_pct", "Porsi pelanggan"),
                           (axes[1], "revenue_pct", "Porsi revenue")]:
        ax.barh(plot_df.index, plot_df[col], color=colors, height=0.65)
        for y, v in enumerate(plot_df[col]):
            ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9, color=TEXT)
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(axis="y", visible=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Aturan segmentasi"):
        st.markdown("""
| Segmen | Aturan |
|---|---|
| Champions | Belanja ≥ 2 kali dan R-score ≥ 4 |
| Loyal | Belanja ≥ 2 kali dan R-score ≤ 3 |
| Baru – Nilai Tinggi | Belanja 1 kali, R-score ≥ 4, M-score ≥ 4 |
| Baru | Belanja 1 kali, R-score ≥ 4, M-score ≤ 3 |
| Perlu Perhatian | Belanja 1 kali, R-score = 3 |
| Berisiko – Nilai Tinggi | Belanja 1 kali, R-score ≤ 2, M-score ≥ 4 |
| Hibernating | Belanja 1 kali, R-score ≤ 2, M-score ≤ 3 |

R-score dan M-score adalah kuantil 1–5 (5 = terbaik) dari recency dan monetary.
""")

# ---------------------------------------------------------------------------
# Tab 5 – Geospatial
# ---------------------------------------------------------------------------
with tab_geo:
    state_df = create_state_df(main_df, order_df)
    metric_options = {
        "Revenue (R$)": ("revenue", True),
        "Jumlah pelanggan": ("customer_count", True),
        "Rata-rata lama pengiriman (hari)": ("avg_delivery_days", False),
        "Persentase pesanan terlambat (%)": ("late_pct", False),
        "Rata-rata review score": ("avg_review", False),
    }
    metric_label = st.selectbox("Metrik peta", list(metric_options))
    metric_col, use_log = metric_options[metric_label]

    left, right = st.columns([3, 2])
    with left:
        label = metric_label + (" – skala log" if use_log else "")
        map_fig = draw_choropleth(state_df[metric_col], label, log=use_log)
        st.pyplot(map_fig)
        plt.close(map_fig)
    with right:
        st.markdown(f"**Peringkat state – {metric_label}**")
        ranked = state_df[[metric_col]].sort_values(metric_col, ascending=False)
        ranked.columns = [metric_label]
        st.dataframe(ranked.style.format("{:,.2f}"), height=470)

    st.subheader("Persebaran lokasi pelanggan")
    points = (main_df.drop_duplicates("customer_unique_id")
                     .dropna(subset=["customer_lat", "customer_lng"])
                     .rename(columns={"customer_lat": "lat", "customer_lng": "lon"})[["lat", "lon"]])
    st.map(points, size=2000, color="#2a78d6aa")

st.caption("Copyright © 2026 · Adam Himawan · Proyek Analisis Data Dicoding")
