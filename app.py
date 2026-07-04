"""
Ứng dụng Đánh giá Rủi ro Tín dụng theo Khung 5C
Mô hình: Logistic Regression (chuyển đổi từ notebook mohinh.ipynb)
"""

import streamlit as st

# LỆNH STREAMLIT ĐẦU TIÊN
st.set_page_config(
    layout="wide",
    page_title="Đánh giá Rủi ro Tín dụng - Mô hình 5C",
    page_icon="🏦",
)

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =========================================================================
# CẤU HÌNH BIẾN (trích xuất từ notebook mohinh.ipynb + dữ liệu 5c.csv)
# =========================================================================
FEATURE_GROUPS = {
    "Tính cách (Character)": ["TC1", "TC2", "TC3", "TC4", "TC5"],
    "Năng lực (Capacity)": ["NL1", "NL2", "NL3", "NL4"],
    "Điều kiện (Conditions)": ["DK1", "DK2", "DK3", "DK4", "DK5"],
    "Vốn (Capital)": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "Tài sản đảm bảo (Collateral)": ["TS1", "TS2", "TS3", "TS4"],
}
X_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]
TARGET_COLUMN = "PD"
TARGET_LABELS = {0: "Không rủi ro", 1: "Có rủi ro"}
REQUIRED_COLUMNS = X_COLUMNS + [TARGET_COLUMN]


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu CSV dùng chung cho toàn app (nhận bytes để hashable)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    return df


# =========================================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# =========================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu khảo sát (CSV)",
        type=["csv"],
        help="Tệp CSV chứa 24 cột khảo sát theo khung 5C (TC, NL, DK, V, TS) và cột nhãn PD.",
    )

    st.subheader("Tham số mô hình AI")
    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test size)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình (notebook gốc dùng 0.2).",
    )
    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=32,
        step=1,
        help="Giá trị khởi tạo ngẫu nhiên để đảm bảo tái lập kết quả (notebook gốc dùng 32).",
    )
    C = st.slider(
        "Hệ số điều chuẩn C",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="C càng nhỏ thì điều chuẩn (regularization) càng mạnh, giúp giảm overfitting. Mặc định sklearn = 1.0.",
    )

    with st.expander("Tham số nâng cao"):
        max_iter = st.number_input(
            "Số vòng lặp tối đa (max_iter)",
            min_value=100,
            max_value=2000,
            value=100,
            step=50,
            help="Số vòng lặp tối đa để thuật toán hội tụ. Mặc định sklearn = 100.",
        )
        solver = st.selectbox(
            "Solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Thuật toán tối ưu hóa dùng để huấn luyện Logistic Regression. Mặc định sklearn = lbfgs.",
        )

    st.divider()
    train_button = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# =========================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# =========================================================================
st.title("🏦 Ứng dụng Đánh giá Rủi ro Tín dụng theo Khung 5C")
st.caption(
    "Ứng dụng sử dụng mô hình **Hồi quy Logistic (Logistic Regression)** để dự báo rủi ro tín dụng "
    "của khách hàng dựa trên 24 tiêu chí khảo sát (thang điểm 1-5) thuộc 5 nhóm: "
    "Tính cách, Năng lực, Điều kiện, Vốn, Tài sản đảm bảo. Đầu vào kỳ vọng là tệp CSV có cấu trúc "
    "tương tự tệp dữ liệu mẫu (5c.csv)."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu CSV ở thanh bên để bắt đầu.")
    st.stop()

try:
    df = load_data(uploaded_file.getvalue())
except Exception as e:
    st.error(f"❌ Không thể đọc tệp dữ liệu. Vui lòng kiểm tra định dạng CSV. Chi tiết lỗi: {e}")
    st.stop()

if df.empty:
    st.error("❌ Tệp dữ liệu rỗng. Vui lòng tải lên tệp khác.")
    st.stop()

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** — {df.shape[0]} dòng, {df.shape[1]} cột")
st.divider()

# =========================================================================
# KHỐI HUẤN LUYỆN — chạy khi bấm nút, lưu vào session_state
# =========================================================================
if train_button:
    try:
        X = df[X_COLUMNS]
        y = df[TARGET_COLUMN]

        if y.nunique() < 2:
            st.error("❌ Biến mục tiêu PD chỉ có 1 lớp duy nhất, không thể huấn luyện mô hình phân loại.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

            model = LogisticRegression(
                C=C,
                max_iter=int(max_iter),
                solver=solver,
                random_state=int(random_state),
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            st.session_state["model"] = model
            st.session_state["preprocessor"] = None  # notebook gốc không dùng scaler/encoder
            st.session_state["feature_columns"] = X_COLUMNS
            st.session_state["results"] = {
                "y_test": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            st.session_state["data_medians"] = X.median()
            st.session_state["data_ranges"] = {
                c: (int(X[c].min()), int(X[c].max())) for c in X_COLUMNS
            }
            st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định'.")
    except Exception as e:
        st.error(f"❌ Lỗi khi huấn luyện mô hình: {e}")

# =========================================================================
# TABS: TP3, TP4, TP5, TP6
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Tổng quan dữ liệu",
        "📈 Trực quan hóa dữ liệu",
        "🎯 Kết quả huấn luyện & kiểm định",
        "🔮 Sử dụng mô hình",
    ]
)

# -------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# -------------------------------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", f"{df.shape[0]:,}")
    col2.metric("Số cột", df.shape[1])
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col3.metric("Dung lượng tệp", f"{file_size_mb:.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến trong mô hình")
    st.caption("Chỉ hiển thị thống kê của 24 biến đầu vào (X) và biến mục tiêu PD (y).")
    st.dataframe(df[REQUIRED_COLUMNS].describe(), use_container_width=True)

# -------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# -------------------------------------------------------------------
with tab2:
    st.subheader("Phân phối biến mục tiêu (PD)")
    target_counts = (
        df[TARGET_COLUMN].map(TARGET_LABELS).value_counts().reset_index()
    )
    target_counts.columns = ["Nhãn", "Số lượng"]
    fig_target = px.bar(
        target_counts,
        x="Nhãn",
        y="Số lượng",
        color="Nhãn",
        text="Số lượng",
        title="Phân phối rủi ro tín dụng (PD)",
        color_discrete_map={"Không rủi ro": "#2ca02c", "Có rủi ro": "#d62728"},
    )
    fig_target.update_layout(height=350, showlegend=False)

    st.caption(
        "Có 24 biến đầu vào — chọn tối đa 3 biến để hiển thị cùng biến mục tiêu (tổng cộng 4 biểu đồ, bố trí lưới 2x2)."
    )
    default_vars = [v for v in ["TC1", "NL1", "V1"] if v in X_COLUMNS]
    selected_vars = st.multiselect(
        "Chọn biến đầu vào để trực quan hóa",
        options=X_COLUMNS,
        default=default_vars,
        max_selections=3,
        help="Các biến khảo sát là dữ liệu rời rạc (thang điểm 1-5) nên được trực quan hóa bằng biểu đồ cột.",
    )

    charts = [fig_target]
    for var in selected_vars:
        counts = df[var].value_counts().sort_index().reset_index()
        counts.columns = [var, "Số lượng"]
        fig_var = px.bar(
            counts,
            x=var,
            y="Số lượng",
            text="Số lượng",
            title=f"Phân phối biến {var}",
        )
        fig_var.update_layout(height=350)
        charts.append(fig_var)

    row1 = st.columns(2)
    row2 = st.columns(2)
    slots = row1 + row2
    for i, fig in enumerate(charts[:4]):
        with slots[i]:
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# -------------------------------------------------------------------
with tab3:
    if "results" not in st.session_state:
        st.info("👈 Vui lòng bấm nút 'Huấn luyện mô hình' ở thanh bên để xem kết quả.")
    else:
        res = st.session_state["results"]
        y_test, y_pred, y_proba = res["y_test"], res["y_pred"], res["y_proba"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        st.subheader("Chỉ tiêu kiểm định mô hình (trên tập test)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Dự báo: Không rủi ro", "Dự báo: Có rủi ro"],
                y=["Thực tế: Không rủi ro", "Thực tế: Có rủi ro"],
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            st.subheader("Đường cong ROC")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})")
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray"), name="Ngẫu nhiên",
                )
            )
            fig_roc.update_layout(
                height=380,
                xaxis_title="Tỷ lệ dương tính giả (FPR)",
                yaxis_title="Tỷ lệ dương tính thật (TPR)",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Báo cáo phân loại chi tiết")
        report = classification_report(
            y_test, y_pred,
            target_names=["Không rủi ro", "Có rủi ro"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        with st.expander("📐 Hệ số hồi quy (mức độ ảnh hưởng của từng tiêu chí)"):
            model = st.session_state["model"]
            coef_df = pd.DataFrame(
                {"Tiêu chí": X_COLUMNS, "Hệ số (coefficient)": model.coef_[0]}
            ).sort_values("Hệ số (coefficient)", ascending=False)
            fig_coef = px.bar(
                coef_df, x="Hệ số (coefficient)", y="Tiêu chí", orientation="h",
                title="Ảnh hưởng của từng tiêu chí đến khả năng rủi ro",
            )
            fig_coef.update_layout(height=600)
            st.plotly_chart(fig_coef, use_container_width=True)

# -------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# -------------------------------------------------------------------
with tab4:
    if "model" not in st.session_state:
        st.info("👈 Vui lòng bấm nút 'Huấn luyện mô hình' ở thanh bên trước khi sử dụng.")
    else:
        model = st.session_state["model"]
        feature_columns = st.session_state["feature_columns"]
        medians = st.session_state["data_medians"]
        ranges = st.session_state["data_ranges"]

        mode = st.radio(
            "Chọn chế độ dự báo",
            ["Nhập trực tiếp", "Tải tệp dữ liệu"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp":
            with st.form("predict_form"):
                st.markdown("Nhập điểm khảo sát (thang điểm 1-5) cho từng tiêu chí:")
                input_values = {}
                for group_name, cols in FEATURE_GROUPS.items():
                    st.markdown(f"**{group_name}**")
                    widget_cols = st.columns(len(cols))
                    for i, col_name in enumerate(cols):
                        lo, hi = ranges[col_name]
                        with widget_cols[i]:
                            input_values[col_name] = st.number_input(
                                col_name,
                                min_value=lo,
                                max_value=hi,
                                value=int(round(medians[col_name])),
                                step=1,
                                help=f"Điểm khảo sát tiêu chí {col_name} (khoảng {lo}-{hi})",
                            )
                submitted = st.form_submit_button(
                    "🔮 Dự báo", type="primary", use_container_width=True
                )

            if submitted:
                X_new = pd.DataFrame([input_values])[feature_columns]
                pred = model.predict(X_new)[0]
                proba = model.predict_proba(X_new)[0]
                label = TARGET_LABELS.get(pred, str(pred))

                if pred == 1:
                    st.error(f"⚠️ Kết quả dự báo: **{label}**")
                else:
                    st.success(f"✅ Kết quả dự báo: **{label}**")

                p1, p2 = st.columns(2)
                p1.metric("Xác suất không có rủi ro", f"{proba[0] * 100:.2f}%")
                p2.metric("Xác suất có rủi ro", f"{proba[1] * 100:.2f}%")

        else:
            st.markdown(
                f"Tải lên tệp CSV có đúng **{len(feature_columns)} cột**: "
                f"`{', '.join(feature_columns)}`"
            )
            batch_file = st.file_uploader(
                "Tệp dữ liệu cần dự báo (theo cấu trúc X_test)",
                type=["csv"],
                key="batch_predict_uploader",
            )

            if batch_file is not None:
                try:
                    new_df = pd.read_csv(batch_file)
                    new_df.columns = new_df.columns.str.strip()
                except Exception as e:
                    st.error(f"❌ Không thể đọc tệp dữ liệu. Chi tiết lỗi: {e}")
                    new_df = None

                if new_df is not None:
                    if new_df.empty:
                        st.error("❌ Tệp dữ liệu rỗng.")
                    else:
                        missing = [c for c in feature_columns if c not in new_df.columns]
                        if missing:
                            st.error(f"❌ Tệp thiếu các cột bắt buộc: {', '.join(missing)}")
                        else:
                            X_new = new_df[feature_columns]
                            preds = model.predict(X_new)
                            probas = model.predict_proba(X_new)[:, 1]

                            result_df = new_df.copy()
                            result_df["Dự báo"] = [
                                TARGET_LABELS.get(p, str(p)) for p in preds
                            ]
                            result_df["Xác suất rủi ro (%)"] = (probas * 100).round(2)

                            st.subheader("Kết quả dự báo")
                            with st.container(height=400):
                                st.dataframe(result_df, use_container_width=True)

                            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "⬇️ Tải kết quả CSV",
                                data=csv_bytes,
                                file_name="ket_qua_du_bao.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
