import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Klasifikasi Burnout",
    page_icon="📊",
    layout="wide"
)

MODEL_CANDIDATES = [
    Path("gaussian_nb_burnout_bundle.joblib"),
    Path("models/gaussian_nb_burnout_bundle.joblib"),
]
ID_COLUMN = "ID_Responden"
TARGET_COLUMN_DEFAULT = "Tingkat_Burnout"

st.title("Sistem Klasifikasi Tingkat Burnout")
st.caption("Gaussian Naive Bayes | Prediksi batch melalui Excel")

st.warning(
    "Aplikasi ini merupakan prototipe penelitian, bukan alat diagnosis psikologis "
    "atau medis. Gunakan data secara bertanggung jawab dan lindungi identitas responden."
)

@st.cache_resource
def load_bundle(model_path: str):
    return joblib.load(model_path)

model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    st.error(
        "File gaussian_nb_burnout_bundle.joblib tidak ditemukan. Letakkan file model "
        "di folder yang sama dengan streamlit_app.py atau di folder models/."
    )
    st.stop()

try:
    bundle = load_bundle(str(model_path))
    model = bundle["model"]
    features = list(bundle["feature_names"])
    defaults = bundle.get("default_values", {})
    target_name = bundle.get("target_name", TARGET_COLUMN_DEFAULT)
    classes = list(model.named_steps["model"].classes_)
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.info(
        "Pastikan versi scikit-learn pada requirements.txt sama dengan versi saat "
        "model joblib dibuat. Model yang disediakan dibuat menggunakan scikit-learn 1.6.1."
    )
    st.stop()

st.success("Model berhasil dimuat.")
model_data_status = str(bundle.get("data_status", ""))
if "SINTETIS" in model_data_status.upper() or "SYNTHETIC" in model_data_status.upper():
    st.warning(
        "Model yang digunakan dilatih menggunakan dataset sintetis, bukan data hasil "
        "pengumpulan lapangan. Hasil prediksi hanya mendemonstrasikan fungsi teknis "
        "dan tidak dapat dijadikan kesimpulan tingkat burnout karyawan sebenarnya."
    )
with st.expander("Informasi model dan fitur"):
    st.write("**Algoritma:** Gaussian Naive Bayes")
    st.write(f"**Jumlah fitur:** {len(features)}")
    st.write("**Fitur:**")
    st.code(", ".join(features))
    st.write(f"**Kelas:** {', '.join(map(str, classes))}")
    st.write(f"**Status dataset model:** {bundle.get('data_status', 'Tidak dicantumkan')}")
    st.write(f"**Nama dataset pelatihan:** {bundle.get('dataset_filename', 'Tidak dicantumkan')}")

def read_input(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded)
    if suffix == ".csv":
        return pd.read_csv(uploaded)
    raise ValueError("Format file harus XLSX, XLS, atau CSV.")

def validate_data(raw):
    errors = []
    if raw.empty:
        return None, ["File tidak memiliki baris data."], pd.DataFrame()

    missing_columns = [f for f in features if f not in raw.columns]
    if missing_columns:
        return None, [
            "Kolom fitur wajib tidak ditemukan: " + ", ".join(missing_columns)
        ], pd.DataFrame()

    checked = raw.copy()
    issue_list = []

    for feature in features:
        original = checked[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        for idx in checked.index[invalid]:
            issue_list.append({
                "Baris_Excel": int(raw.index.get_loc(idx)) + 2,
                "ID_Responden": checked.loc[idx, ID_COLUMN] if ID_COLUMN in checked.columns else str(idx),
                "Kolom": feature,
                "Masalah": "Nilai bukan numerik"
            })
        checked[feature] = converted

    for idx in checked.index:
        for feature in features:
            value = checked.loc[idx, feature]
            if pd.isna(value):
                issue_list.append({
                    "Baris_Excel": int(raw.index.get_loc(idx)) + 2,
                    "ID_Responden": checked.loc[idx, ID_COLUMN] if ID_COLUMN in checked.columns else str(idx),
                    "Kolom": feature,
                    "Masalah": "Nilai kosong atau tidak valid"
                })
            elif not np.isfinite(float(value)):
                issue_list.append({
                    "Baris_Excel": int(raw.index.get_loc(idx)) + 2,
                    "ID_Responden": checked.loc[idx, ID_COLUMN] if ID_COLUMN in checked.columns else str(idx),
                    "Kolom": feature,
                    "Masalah": "Nilai tidak finite"
                })

    issues = pd.DataFrame(issue_list).drop_duplicates() if issue_list else pd.DataFrame(
        columns=["Baris_Excel", "ID_Responden", "Kolom", "Masalah"]
    )
    if not issues.empty:
        errors.append("Terdapat nilai kosong atau tidak valid. Perbaiki data yang ditandai sebelum klasifikasi.")
        return None, errors, issues

    return checked, errors, issues

def predict_batch(data):
    X = data[features].copy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = data.copy()
    result["Prediksi_Burnout"] = predictions
    for index, class_name in enumerate(classes):
        safe_class = str(class_name).strip().replace(" ", "_")
        result[f"Probabilitas_{safe_class}"] = probabilities[:, index]
    result["Keyakinan_Model_Persen"] = probabilities.max(axis=1) * 100
    return result

def create_excel(result, summary, descriptive=None, report=None, matrix=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Hasil_Prediksi", index=False)
        summary.to_excel(writer, sheet_name="Ringkasan_Kategori", index=False)
        if descriptive is not None:
            descriptive.to_excel(writer, sheet_name="Kesesuaian_Deskriptif", index=False)
        if report is not None:
            report.to_excel(writer, sheet_name="Evaluasi_Test", index=False)
        if matrix is not None:
            matrix.to_excel(writer, sheet_name="Confusion_Matrix")
    output.seek(0)
    return output.getvalue()

tab_batch, tab_single = st.tabs(["📁 Upload Excel (Batch)", "✍️ Prediksi Satu Responden"])

with tab_batch:
    st.subheader("Klasifikasi seluruh responden")
    st.write(
        "Unggah Excel yang berisi banyak responden. Aplikasi membaca seluruh baris, "
        "memvalidasi sembilan fitur, lalu memprediksi kategori dan probabilitas secara bersamaan."
    )
    uploaded = st.file_uploader("Pilih dataset Excel atau CSV", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        try:
            raw = read_input(uploaded)
            st.write(f"**File:** {uploaded.name} · **Baris:** {len(raw)} · **Kolom:** {len(raw.columns)}")
            with st.expander("Pratinjau dataset", expanded=True):
                st.dataframe(raw.head(10), use_container_width=True)

            valid_data, validation_errors, issues = validate_data(raw)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                if not issues.empty:
                    st.dataframe(issues, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Unduh daftar data yang perlu diperbaiki",
                        issues.to_csv(index=False).encode("utf-8-sig"),
                        "Data_Perlu_Diperbaiki.csv",
                        "text/csv"
                    )
            else:
                st.success(f"Validasi berhasil. {len(valid_data)} baris siap diproses.")
                if ID_COLUMN in valid_data.columns:
                    duplicate_ids = valid_data[ID_COLUMN].duplicated(keep=False)
                    if duplicate_ids.any():
                        st.warning(
                            f"Ada {int(duplicate_ids.sum())} baris dengan ID berulang. "
                            "Periksa ID responden sebelum menggunakan hasil."
                        )

                if st.button("🚀 Proses Seluruh Data", type="primary", use_container_width=True):
                    try:
                        st.session_state["batch_result"] = predict_batch(valid_data)
                        st.session_state["batch_filename"] = uploaded.name
                    except Exception as exc:
                        st.error(f"Prediksi gagal dijalankan: {exc}")

        except Exception as exc:
            st.error(f"File tidak dapat dibaca: {exc}")

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        st.divider()
        st.subheader("Hasil klasifikasi")
        st.caption(f"Sumber data: {st.session_state.get('batch_filename', '-')}")
        st.metric("Jumlah responden diproses", f"{len(result)}")

        counts = (
            result["Prediksi_Burnout"]
            .value_counts()
            .reindex(classes, fill_value=0)
            .rename_axis("Kategori")
            .reset_index(name="Jumlah_Responden")
        )
        counts["Persentase"] = counts["Jumlah_Responden"] / len(result) * 100

        metric_cols = st.columns(len(classes))
        for col, (_, row) in zip(metric_cols, counts.iterrows()):
            col.metric(
                str(row["Kategori"]),
                f"{int(row['Jumlah_Responden'])} responden",
                f"{row['Persentase']:.2f}%"
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**Distribusi hasil prediksi**")
            st.bar_chart(counts.set_index("Kategori")["Jumlah_Responden"])
        with right:
            st.markdown("**Ringkasan kategori**")
            st.dataframe(counts, use_container_width=True, hide_index=True)

        st.markdown("**Hasil prediksi per responden**")
        st.dataframe(result, use_container_width=True, hide_index=True)

        descriptive = None
        report_df = None
        confusion_df = None

        if target_name in result.columns and result[target_name].notna().any():
            st.divider()
            st.subheader("Pemeriksaan terhadap label aktual")
            labelled = result[result[target_name].notna()].copy()
            actual = labelled[target_name].astype(str).str.strip()
            predicted = labelled["Prediksi_Burnout"].astype(str).str.strip()
            agreement = (actual.str.casefold() == predicted.str.casefold()).mean()

            descriptive = pd.DataFrame([{
                "Jumlah_Baris_Berlabel": len(labelled),
                "Jumlah_Prediksi_Sesuai": int((actual.str.casefold() == predicted.str.casefold()).sum()),
                "Persentase_Kesesuaian_Deskriptif": agreement * 100,
                "Catatan": "Bukan metrik generalisasi jika data yang sama digunakan saat pelatihan."
            }])
            st.metric("Kesesuaian deskriptif dengan label pada file", f"{agreement * 100:.2f}%")
            st.info(
                "Kesesuaian pada file yang diunggah tidak otomatis merupakan akurasi pengujian. "
                "Jika file ini juga digunakan untuk melatih model, hasilnya tidak independen "
                "dan tidak boleh dilaporkan sebagai performa generalisasi."
            )

            split_col = next(
                (c for c in ["Data_Split", "Split", "Subset", "Jenis_Data"] if c in labelled.columns),
                None
            )
            if split_col:
                split_values = labelled[split_col].astype(str).str.strip().str.casefold()
                test_mask = split_values.isin(["test", "testing", "uji", "holdout", "data uji"])
                test_rows = labelled.loc[test_mask].copy()
            else:
                test_rows = labelled.iloc[0:0].copy()

            if len(test_rows) > 0:
                from sklearn.metrics import classification_report, confusion_matrix
                y_true = test_rows[target_name].astype(str).str.strip()
                y_pred = test_rows["Prediksi_Burnout"].astype(str).str.strip()
                label_order = [str(c) for c in classes]
                report = classification_report(
                    y_true, y_pred, labels=label_order,
                    output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report).transpose().reset_index(names="Kategori_Metrik")
                cm = confusion_matrix(y_true, y_pred, labels=label_order)
                confusion_df = pd.DataFrame(
                    cm,
                    index=[f"Aktual_{c}" for c in label_order],
                    columns=[f"Prediksi_{c}" for c in label_order]
                )
                st.success(f"Data test/holdout teridentifikasi: {len(test_rows)} baris.")
                st.markdown("**Classification report pada data test/holdout**")
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                st.markdown("**Confusion matrix pada data test/holdout**")
                st.dataframe(confusion_df, use_container_width=True)
            else:
                st.warning(
                    "Kolom label aktual tersedia, tetapi tidak ada baris yang ditandai sebagai "
                    "test/holdout. Confusion matrix dan classification report tidak dihitung "
                    "agar data pelatihan tidak keliru dianggap sebagai data uji independen."
                )

        excel_bytes = create_excel(result, counts, descriptive, report_df, confusion_df)
        st.download_button(
            "⬇️ Unduh seluruh hasil dalam Excel",
            data=excel_bytes,
            file_name="Hasil_Klasifikasi_Burnout_50_Responden.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

with tab_single:
    st.subheader("Prediksi satu responden")
    with st.form("single_prediction_form"):
        values = {}
        columns = st.columns(2)
        for i, feature in enumerate(features):
            with columns[i % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    value=float(defaults.get(feature, 0.0)),
                    step=0.1,
                    format="%.2f",
                    key=f"single_{feature}"
                )
        submitted = st.form_submit_button("Proses Klasifikasi", use_container_width=True)

    if submitted:
        X_single = pd.DataFrame([values], columns=features)
        prediction = model.predict(X_single)[0]
        probabilities = model.predict_proba(X_single)[0]
        st.metric("Kategori prediksi", str(prediction))
        probability_df = pd.DataFrame({
            "Kategori": classes,
            "Probabilitas": probabilities,
            "Persentase": probabilities * 100
        })
        st.dataframe(probability_df, use_container_width=True, hide_index=True)
        st.bar_chart(probability_df.set_index("Kategori")["Probabilitas"])

st.caption(
    "Model yang digunakan adalah model tersimpan. Pengunggahan dataset tidak melatih ulang "
    "model. Evaluasi performa harus menggunakan data uji independen yang tidak digunakan "
    "dalam pelatihan atau pemilihan model."
)
