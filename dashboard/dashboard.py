import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
# Semua grafik memakai Altair (Vega-Lite) yang digambar langsung di browser,
# sehingga tidak bergantung pada file gambar sementara di server.
BASE_DIR = Path(__file__).resolve().parent
MAIN, ACCENT, GREY, LABEL = "#2a78d6", "#eb6834", "#b4b3ad", "#808080"

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
alt.data_transformers.disable_max_rows()

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


@st.cache_data
def state_centroids():
    rows = []
    for feat in load_geojson()["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        largest = max(polys, key=lambda p: abs(np.ptp(np.array(p[0]), axis=0).prod()))
        lon, lat = polygon_centroid(largest[0])
        rows.append({"state": feat["properties"]["sigla"], "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


def fmt_brl(value):
    if value >= 1e6:
        return f"R${value / 1e6:,.2f} jt"
    if value >= 1e3:
        return f"R${value / 1e3:,.1f}K"
    return f"R${value:,.0f}"


# ---------------------------------------------------------------------------
# Helper: grafik Altair
# ---------------------------------------------------------------------------
BRL_AXIS = "'R$' + format(datum.value / 1000, ',.0f') + 'K'"


def show_chart(chart):
    """Tampilkan grafik selebar container (kompatibel Streamlit lama & baru)."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def line_chart(df, field, title, axis_label_expr=None, tooltip_format=",.0f"):
    peak_month = df.loc[df["revenue"].idxmax(), "month"]
    data = df.assign(is_peak=df["month"] == peak_month)
    axis = alt.Axis(labelExpr=axis_label_expr) if axis_label_expr else alt.Axis(format=",.0f")
    base = alt.Chart(data).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)),
        y=alt.Y(f"{field}:Q", title=title, axis=axis),
        tooltip=[alt.Tooltip("month:T", title="Bulan", format="%B %Y"),
                 alt.Tooltip(f"{field}:Q", title=title, format=tooltip_format)],
    )
    line = base.mark_line(color=MAIN, strokeWidth=2, point=alt.OverlayMarkDef(color=MAIN, size=45))
    peak = base.transform_filter(alt.datum.is_peak).mark_point(color=ACCENT, filled=True, size=160, opacity=1)
    return (line + peak).properties(height=260, title=title)


def hbar_chart(df, cat_col, val_col, x_title, label_col, tooltip,
               highlight=(), highlight_color=MAIN, base_color=GREY):
    """Bar horizontal berurutan; kategori di `highlight` diberi warna penekanan."""
    order = df[cat_col].tolist()
    x_max = float(df[val_col].max()) if len(df) else 1.0
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat_col}:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X(f"{val_col}:Q", title=x_title, scale=alt.Scale(domain=[0, (x_max or 1.0) * 1.35])),
        tooltip=tooltip,
    )
    if highlight:
        color = alt.condition(alt.FieldOneOfPredicate(field=cat_col, oneOf=list(highlight)),
                              alt.value(highlight_color), alt.value(base_color))
    else:
        color = alt.value(base_color)
    bars = base.mark_bar(cornerRadiusEnd=3).encode(color=color)
    labels = base.mark_text(align="left", dx=4, color=LABEL, fontSize=11).encode(text=f"{label_col}:N")
    return (bars + labels).properties(height=max(34 * len(df), 200))


def vbar_chart(df, cat_col, val_col, y_title, y_domain, label_col, tooltip):
    base = alt.Chart(df).encode(
        x=alt.X(f"{cat_col}:N", sort=df[cat_col].tolist(), title="Keterlambatan dari estimasi",
                axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=120,
                              labelExpr="replace(datum.label, 'Telat ', '')")),
        y=alt.Y(f"{val_col}:Q", title=y_title, scale=alt.Scale(domain=y_domain)),
        tooltip=tooltip,
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.condition(alt.FieldOneOfPredicate(field=cat_col, oneOf=[DELAY_LABELS[0]]),
                            alt.value(MAIN), alt.value(ACCENT)))
    labels = base.mark_text(dy=-8, color=LABEL, fontSize=12).encode(text=f"{label_col}:N")
    return (bars + labels).properties(height=320)


def choropleth(state_values, title, log=False):
    geo = alt.Data(values=load_geojson()["features"])
    values_df = pd.DataFrame({"customer_state": state_values.index, "value": state_values.values})
    if log:
        values_df = values_df[values_df["value"] > 0]

    background = alt.Chart(geo).mark_geoshape(fill="#e9e8e4", stroke="white", strokeWidth=0.7)
    colored = (alt.Chart(geo)
               .mark_geoshape(stroke="white", strokeWidth=0.7)
               .transform_lookup(lookup="properties.sigla",
                                 from_=alt.LookupData(values_df, "customer_state", ["value"]))
               .transform_filter("isValid(datum.value)")
               .encode(color=alt.Color("value:Q", title=title,
                                       scale=alt.Scale(type="log" if log else "linear", scheme="blues")),
                       tooltip=[alt.Tooltip("properties.name:N", title="State"),
                                alt.Tooltip("properties.sigla:N", title="Kode"),
                                alt.Tooltip("value:Q", title=title, format=",.2f")]))
    halo = (alt.Chart(state_centroids())
            .mark_text(fontSize=9, fontWeight="bold", stroke="#1f1f1f", strokeWidth=2.5, color="#1f1f1f")
            .encode(longitude="lon:Q", latitude="lat:Q", text="state:N"))
    labels = halo.mark_text(fontSize=9, fontWeight="bold", color="white")
    return (background + colored + halo + labels).project("mercator").properties(height=560)


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
                f"dari **{int(peak['order_count']):,}** pesanan (titik oranye).")

    show_chart(line_chart(monthly_df, "order_count", "Jumlah pesanan"))
    show_chart(line_chart(monthly_df, "revenue", "Revenue (R$)",
                          axis_label_expr=BRL_AXIS, tooltip_format=",.2f"))

    with st.expander("Lihat tabel data bulanan"):
        st.dataframe(monthly_df.assign(month=monthly_df["month"].dt.strftime("%Y-%m")), hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 – Kategori produk (Pertanyaan 2)
# ---------------------------------------------------------------------------
with tab_cat:
    category_df = create_category_df(main_df)
    n_top = st.slider("Jumlah kategori yang ditampilkan", min_value=5, max_value=15, value=10)
    top_share = category_df.head(n_top)["revenue_share_pct"].sum()
    st.subheader(f"{n_top} kategori teratas menyumbang {top_share:.1f}% revenue")

    category_df["label"] = [f"{fmt_brl(r)} ({s:.1f}%)"
                            for r, s in zip(category_df["revenue"], category_df["revenue_share_pct"])]
    tooltip = [alt.Tooltip("product_category_name_english:N", title="Kategori"),
               alt.Tooltip("revenue:Q", title="Revenue (R$)", format=",.2f"),
               alt.Tooltip("order_count:Q", title="Jumlah pesanan", format=","),
               alt.Tooltip("revenue_share_pct:Q", title="Kontribusi (%)", format=".2f")]
    top_df = category_df.head(n_top)
    bottom_df = category_df.tail(n_top).iloc[::-1].copy()
    bottom_df["label"] = bottom_df["revenue"].map(fmt_brl)
    cat = "product_category_name_english"

    left, right = st.columns(2)
    with left:
        st.markdown("**Revenue tertinggi** (biru = kategori nomor 1)")
        show_chart(hbar_chart(top_df, cat, "revenue", "Revenue (R$)", "label", tooltip,
                              highlight=[top_df[cat].iloc[0]], highlight_color=MAIN))
    with right:
        st.markdown("**Revenue terendah** (oranye = kategori paling kecil)")
        show_chart(hbar_chart(bottom_df, cat, "revenue", "Revenue (R$)", "label", tooltip,
                              highlight=[bottom_df[cat].iloc[0]], highlight_color=ACCENT))

    with st.expander("Lihat tabel semua kategori"):
        st.dataframe(category_df.drop(columns="label"), hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 – Pengiriman & review (Pertanyaan 3)
# ---------------------------------------------------------------------------
with tab_delivery:
    delay_df = create_delay_df(order_df).reset_index()
    delay_df["delay_group"] = delay_df["delay_group"].astype(str)
    reviewed = order_df.dropna(subset=["review_score"])
    on_time = reviewed.loc[~reviewed["is_late"], "review_score"].mean()
    late = reviewed.loc[reviewed["is_late"], "review_score"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Skor rata-rata – tepat waktu", f"{on_time:.2f}" if pd.notna(on_time) else "–")
    c2.metric("Skor rata-rata – terlambat", f"{late:.2f}" if pd.notna(late) else "–",
              delta=f"{late - on_time:.2f} poin" if pd.notna(late) and pd.notna(on_time) else None)
    c3.metric("Rata-rata lama pengiriman", f"{order_df['delivery_time_days'].mean():.1f} hari")

    st.subheader("Makin lama terlambat, makin rendah review score")
    delay_df["score_label"] = delay_df["avg_review"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    delay_df["bad_label"] = delay_df["bad_review_pct"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "")
    tooltip = [alt.Tooltip("delay_group:N", title="Kelompok"),
               alt.Tooltip("order_count:Q", title="Jumlah pesanan", format=","),
               alt.Tooltip("avg_review:Q", title="Rata-rata skor", format=".2f"),
               alt.Tooltip("bad_review_pct:Q", title="Ulasan buruk (%)", format=".1f")]

    left, right = st.columns(2)
    with left:
        st.markdown("**Rata-rata review score (1–5)**")
        show_chart(vbar_chart(delay_df, "delay_group", "avg_review", "Rata-rata review score",
                              [0, 5], "score_label", tooltip))
    with right:
        st.markdown("**Persentase ulasan buruk (skor 1–2)**")
        show_chart(vbar_chart(delay_df, "delay_group", "bad_review_pct", "Ulasan buruk (%)",
                              [0, 100], "bad_label", tooltip))
    st.caption("Kelompok keterlambatan dibuat dengan teknik binning (pd.cut) pada selisih tanggal "
               "diterima dengan tanggal estimasi. Biru = tepat waktu, oranye = terlambat.")

# ---------------------------------------------------------------------------
# Tab 4 – RFM & segmentasi
# ---------------------------------------------------------------------------
with tab_rfm:
    reference_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    if main_df["customer_unique_id"].nunique() < 5:
        st.info("Data pelanggan terlalu sedikit untuk analisis RFM. Perluas filter.")
    else:
        rfm_df = create_rfm_df(main_df, reference_date)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rata-rata recency", f"{rfm_df['recency'].mean():.0f} hari")
        c2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f} pesanan")
        c3.metric("Rata-rata monetary", fmt_brl(rfm_df["monetary"].mean()))

        st.subheader("Pelanggan terbaik berdasarkan parameter RFM")
        cols = st.columns(3)
        for col, data, field, title in [
            (cols[0], rfm_df.nsmallest(5, "recency"), "recency", "Recency (hari)"),
            (cols[1], rfm_df.nlargest(5, "frequency"), "frequency", "Frequency (pesanan)"),
            (cols[2], rfm_df.nlargest(5, "monetary"), "monetary", "Monetary (R$)"),
        ]:
            data = data.assign(customer=data["customer_unique_id"].str[:8] + "…",
                               label=data[field].map(lambda v: f"{v:,.0f}"))
            with col:
                st.markdown(f"**{title}**")
                show_chart(hbar_chart(data, "customer", field, title, "label",
                                      [alt.Tooltip("customer_unique_id:N", title="ID pelanggan"),
                                       alt.Tooltip(f"{field}:Q", title=title, format=",.2f")],
                                      base_color=MAIN))

        st.subheader("Segmentasi pelanggan (manual grouping)")
        segment_df = (rfm_df.groupby("segment")
                            .agg(customer_count=("customer_unique_id", "count"), revenue=("monetary", "sum"))
                            .reindex(SEGMENT_ORDER).fillna(0).rename_axis("segment").reset_index())
        segment_df["customer_pct"] = segment_df["customer_count"] / segment_df["customer_count"].sum() * 100
        segment_df["revenue_pct"] = segment_df["revenue"] / segment_df["revenue"].sum() * 100
        segment_df["customer_label"] = segment_df["customer_pct"].map(lambda v: f"{v:.1f}%")
        segment_df["revenue_label"] = segment_df["revenue_pct"].map(lambda v: f"{v:.1f}%")
        tooltip = [alt.Tooltip("segment:N", title="Segmen"),
                   alt.Tooltip("customer_count:Q", title="Jumlah pelanggan", format=","),
                   alt.Tooltip("revenue:Q", title="Revenue (R$)", format=",.2f")]
        at_risk = ["Berisiko – Nilai Tinggi"]

        left, right = st.columns(2)
        with left:
            st.markdown("**Porsi pelanggan (%)**")
            show_chart(hbar_chart(segment_df, "segment", "customer_pct", "% pelanggan", "customer_label",
                                  tooltip, highlight=at_risk, highlight_color=ACCENT, base_color=MAIN))
        with right:
            st.markdown("**Porsi revenue (%)**")
            show_chart(hbar_chart(segment_df, "segment", "revenue_pct", "% revenue", "revenue_label",
                                  tooltip, highlight=at_risk, highlight_color=ACCENT, base_color=MAIN))
        st.caption("Oranye = segmen *Berisiko – Nilai Tinggi*: sedikit pelanggan tetapi porsi revenue besar.")

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
        st.markdown(f"**Peta per state – {metric_label}**" + (" (skala log)" if use_log else ""))
        show_chart(choropleth(state_df[metric_col].dropna(), metric_label, log=use_log))
    with right:
        st.markdown(f"**Peringkat state – {metric_label}**")
        ranked = state_df[[metric_col]].sort_values(metric_col, ascending=False)
        ranked.columns = [metric_label]
        st.dataframe(ranked.style.format("{:,.2f}"), height=520)

    st.subheader("Persebaran lokasi pelanggan")
    points = (main_df.drop_duplicates("customer_unique_id")
                     .dropna(subset=["customer_lat", "customer_lng"])
                     .rename(columns={"customer_lat": "lat", "customer_lng": "lon"})[["lat", "lon"]])
    st.map(points, size=2000, color="#2a78d6aa")

st.caption("Copyright © 2026 · Adam Himawan · Proyek Analisis Data Dicoding")
