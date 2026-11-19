
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_FILE = Path("gaussian_nb_burnout_bundle.joblib")
st.set_page_config(page_title="Klasifikasi Burnout", layout="centered")
st.title("Sistem Klasifikasi Tingkat Burnout")
st.caption("Model Gaussian Naive Bayes — prototipe penelitian")

if not MODEL_FILE.exists():
    st.error("Letakkan gaussian_nb_burnout_bundle.joblib di folder yang sama.")
    st.stop()

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["feature_names"]
defaults = bundle["default_values"]

st.warning(
    "Hasil bukan diagnosis psikologis. Dataset penelitian saat ini bersifat "
    "sintetis sehingga aplikasi hanya untuk demonstrasi teknis."
)

with st.form("form_prediksi"):
    values = {}
    for feature in features:
        values[feature] = st.number_input(
            feature.replace("_", " "),
            value=float(defaults.get(feature, 0.0)),
            step=0.1,
            format="%.2f"
        )
    submitted = st.form_submit_button("Proses Klasifikasi")

if submitted:
    X_input = pd.DataFrame([values], columns=features)
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    classes = model.named_steps["model"].classes_

    st.subheader("Hasil Klasifikasi")
    st.metric("Kategori prediksi", prediction)
    probability_df = pd.DataFrame({
        "Kategori": classes,
        "Probabilitas": probabilities
    })
    st.dataframe(probability_df, use_container_width=True)
    st.bar_chart(probability_df.set_index("Kategori"))
    st.info(
        "Probabilitas adalah keluaran model berdasarkan data pelatihan, "
        "bukan ukuran klinis atau diagnosis individu."
    )
