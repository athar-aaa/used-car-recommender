from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import FEATURE_LABELS, load_vehicle_data
from src.recommender import UsedCarRecommender, UserPreferences


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Used Car Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css() -> None:
    css = (PROJECT_ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_vehicle_data().copy()


@st.cache_resource
def get_model(data: pd.DataFrame) -> UsedCarRecommender:
    return UsedCarRecommender().fit(data)


def rupiah(value: int | float) -> str:
    return "Rp" + f"{int(value):,}".replace(",", ".")


def number_id(value: int | float) -> str:
    return f"{int(value):,}".replace(",", ".")


def page_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-kicker">{html.escape(kicker)}</div>
          <h2>{html.escape(title)}</h2>
          <div class="section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_preferences_form(data: pd.DataFrame) -> tuple[UserPreferences | None, int]:
    price_min = int(data["price_rp"].min() // 5_000_000 * 5_000_000)
    price_max = int((data["price_rp"].max() // 5_000_000 + 1) * 5_000_000)
    default_low = max(price_min, 100_000_000)
    default_high = min(price_max, 300_000_000)

    with st.form("preference_form"):
        st.markdown("#### Preferensi utama")
        budget = st.slider(
            "Rentang anggaran",
            min_value=price_min,
            max_value=price_max,
            value=(default_low, default_high),
            step=5_000_000,
            format="Rp%d",
        )
        first, second = st.columns(2)
        with first:
            year_min = st.slider(
                "Tahun minimum",
                int(data["year"].min()),
                int(data["year"].max()),
                max(2017, int(data["year"].min())),
            )
        with second:
            mileage_max = st.slider(
                "Kilometer maksimum",
                10_000,
                int((data["mileage_km"].max() // 10_000 + 1) * 10_000),
                100_000,
                step=10_000,
                format="%d km",
            )

        brands = st.multiselect(
            "Merek pilihan",
            sorted(data["brand"].unique()),
            placeholder="Semua merek",
        )
        first, second = st.columns(2)
        with first:
            transmission_choice = st.selectbox(
                "Transmisi",
                ["Tidak ada preferensi", *sorted(data["transmission"].unique())],
            )
            locations = sorted(data.loc[data["location"].ne("Unknown"), "location"].unique())
            location_choice = st.selectbox(
                "Lokasi", ["Tidak ada preferensi", *locations]
            )
        with second:
            plate_choice = st.selectbox(
                "Jenis pelat",
                ["Tidak ada preferensi", *sorted(data["plate_type"].unique())],
            )
            instalment_max = st.number_input(
                "Batas cicilan per bulan (0 berarti tanpa batas)",
                min_value=0,
                max_value=int(data["instalment_rp"].max()),
                value=0,
                step=250_000,
            )

        st.markdown("#### Fitur yang diinginkan")
        feature_columns = st.columns(2)
        selected_features: list[str] = []
        for index, (feature, label) in enumerate(FEATURE_LABELS.items()):
            with feature_columns[index % 2]:
                if st.checkbox(label, key=f"feature_{feature}"):
                    selected_features.append(feature)

        n_results = st.segmented_control(
            "Jumlah rekomendasi", options=[5, 8, 10], default=5
        )
        submitted = st.form_submit_button(
            "Cari rekomendasi", use_container_width=True
        )

    if not submitted:
        return None, int(n_results or 5)

    preferences = UserPreferences(
        budget_min=int(budget[0]),
        budget_max=int(budget[1]),
        year_min=int(year_min),
        mileage_max=int(mileage_max),
        brands=tuple(brands),
        transmission=None if transmission_choice == "Tidak ada preferensi" else transmission_choice,
        location=None if location_choice == "Tidak ada preferensi" else location_choice,
        plate_type=None if plate_choice == "Tidak ada preferensi" else plate_choice,
        instalment_max=int(instalment_max) or None,
        desired_features=tuple(selected_features),
    )
    return preferences, int(n_results or 5)


def render_car_card(
    row: pd.Series, reasons: list[str], preferences: UserPreferences
) -> None:
    location = (
        "Lokasi tidak tersedia"
        if str(row["location"]) == "Unknown"
        else str(row["location"])
    )
    reason_html = "".join(
        f"<div class='reason'>{html.escape(reason)}</div>" for reason in reasons
    )
    feature_status_html = ""
    if preferences.desired_features:
        statuses = []
        for feature in preferences.desired_features:
            available = bool(row[feature])
            status_class = "feature-available" if available else "feature-unavailable"
            status_text = "Tersedia" if available else "Tidak tersedia"
            statuses.append(
                f'<div class="feature-status {status_class}">'
                f'<span class="feature-name">{html.escape(FEATURE_LABELS[feature])}</span>'
                f'<span class="feature-value">{status_text}</span>'
                "</div>"
            )
        feature_status_html = (
            "<div class='feature-section'><div class='feature-title'>Fitur pilihan</div>"
            f"<div class='feature-grid'>{''.join(statuses)}</div></div>"
        )
    st.markdown(
        f"""
        <article class="car-card">
          <div class="car-card-top">
            <div>
              <div class="car-name">{html.escape(str(row['display_name']))}</div>
              <div class="car-price">{rupiah(row['price_rp'])}</div>
            </div>
            <div class="score">{row['match_score']:.1f}% cocok</div>
          </div>
          <div class="car-meta">
            {int(row['year'])} &nbsp;·&nbsp; {number_id(row['mileage_km'])} km
            &nbsp;·&nbsp; {html.escape(str(row['transmission']))}
            &nbsp;·&nbsp; {html.escape(location)}
          </div>
          {reason_html}
          {feature_status_html}
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_comparison(results: pd.DataFrame) -> None:
    if results.empty:
        return
    page_header(
        "Perbandingan",
        "Bandingkan pilihan teratas",
        "Pilih hingga tiga kendaraan untuk melihat atribut utamanya secara berdampingan.",
    )
    choices = st.multiselect(
        "Kendaraan yang dibandingkan",
        results["display_name"].tolist(),
        max_selections=3,
        placeholder="Pilih kendaraan",
    )
    if not choices:
        return
    selected = results[results["display_name"].isin(choices)].copy()
    table = selected.set_index("display_name")[[
        "match_score",
        "price_rp",
        "year",
        "mileage_km",
        "transmission",
        "location",
        "instalment_rp",
    ]].T
    table.index = [
        "Skor kecocokan",
        "Harga",
        "Tahun",
        "Kilometer",
        "Transmisi",
        "Lokasi",
        "Cicilan per bulan",
    ]
    for car in table.columns:
        table.loc["Skor kecocokan", car] = f"{float(table.loc['Skor kecocokan', car]):.1f}%"
        table.loc["Harga", car] = rupiah(table.loc["Harga", car])
        table.loc["Kilometer", car] = number_id(table.loc["Kilometer", car]) + " km"
        table.loc["Cicilan per bulan", car] = rupiah(table.loc["Cicilan per bulan", car])
    st.dataframe(table, use_container_width=True)


def render_recommendation(data: pd.DataFrame, model: UsedCarRecommender) -> None:
    page_header(
        "Pencarian",
        "Temukan kendaraan yang paling sesuai",
        "Anggaran, tahun, dan kilometer menjadi batas utama. Preferensi lainnya membantu KNN menyusun peringkat kandidat.",
    )
    form_column, result_column = st.columns([0.9, 1.35], gap="large")
    with form_column:
        preferences, n_results = build_preferences_form(data)
        if preferences is not None:
            with st.spinner("Menghitung kemiripan kendaraan..."):
                results, relaxed = model.recommend(preferences, n_results)
            st.session_state["results"] = results
            st.session_state["preferences"] = preferences
            st.session_state["relaxed"] = relaxed

    with result_column:
        st.markdown("#### Hasil rekomendasi")
        results = st.session_state.get("results")
        saved_preferences = st.session_state.get("preferences")
        if results is None:
            st.markdown(
                "<div class='notice'>Isi preferensi lalu pilih Cari rekomendasi. Hasil terbaik akan tampil di area ini.</div>",
                unsafe_allow_html=True,
            )
        elif results.empty:
            st.warning("Belum ada kendaraan yang cukup dekat dengan batas tersebut. Coba perluas anggaran atau kilometer maksimum.")
        else:
            if st.session_state.get("relaxed"):
                st.markdown(
                    "<div class='notice'>Kandidat tepat belum mencukupi. Sistem memperluas anggaran 15%, menurunkan batas tahun dua tahun, dan menaikkan batas kilometer 25%.</div>",
                    unsafe_allow_html=True,
                )
            for _, row in results.iterrows():
                render_car_card(
                    row,
                    model.explanation(row, saved_preferences),
                    saved_preferences,
                )

    results = st.session_state.get("results")
    if isinstance(results, pd.DataFrame) and not results.empty:
        render_comparison(results)


load_css()
data = get_data()
model = get_model(data)

st.markdown(
    """
    <div class="brandbar">
      <div class="brandname">Used Car Recommender</div>
      <div class="brandmeta">Content-based KNN</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_recommendation(data, model)
