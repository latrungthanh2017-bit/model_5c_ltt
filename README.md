# 🏦 Ứng dụng Đánh giá Rủi ro Tín dụng theo Khung 5C

Ứng dụng Streamlit được chuyển đổi từ notebook `mohinh.ipynb`. Ứng dụng huấn luyện mô hình
**Logistic Regression** (scikit-learn) để dự báo biến mục tiêu **PD** (0 = không rủi ro,
1 = có rủi ro) dựa trên 24 câu trả lời khảo sát theo thang đo Likert 1–5, thuộc khung **5C**
trong thẩm định tín dụng:

| Nhóm | Cột | Ý nghĩa |
|---|---|---|
| Tính cách (Character) | `TC1`–`TC5` | |
| Năng lực (Capacity) | `NL1`–`NL4` | |
| Điều kiện (Conditions) | `DK1`–`DK5` | |
| Vốn (Capital) | `V1`–`V6` | |
| Tài sản đảm bảo (Collateral) | `TS1`–`TS4` | |

Notebook gốc **không** dùng scaler/encoder — 24 biến khảo sát được đưa thẳng vào
`LogisticRegression()` (tham số mặc định), chia tập train/test với `test_size=0.2`,
`random_state=32`. Ứng dụng tái hiện đúng pipeline này và cho phép người dùng điều chỉnh
thêm các siêu tham số.

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Chạy ứng dụng

```bash
streamlit run app.py
```

> Khuyến nghị dùng **Streamlit ≥ 1.38** để đảm bảo hỗ trợ đầy đủ `st.container(height=...)`,
> `st.multiselect(max_selections=...)` và các thành phần bố cục dùng trong ứng dụng.

## 3. Cấu trúc file dữ liệu đầu vào

Tệp CSV cần có tối thiểu 25 cột sau (đúng cấu trúc tệp mẫu `5c.csv`, 150 dòng khảo sát):

- **24 biến đầu vào (X):** `TC1`–`TC5`, `NL1`–`NL4`, `DK1`–`DK5`, `V1`–`V6`, `TS1`–`TS4`
  (giá trị nguyên, thang điểm 1–5)
- **1 biến mục tiêu (y):** `PD` — nhãn nhị phân (`0` = không rủi ro, `1` = có rủi ro)

**Ghi chú:** tệp mẫu còn có thêm cột `Dấu thời gian` và `NN`. Hai cột này **không** được
notebook gốc đưa vào mô hình, nên ứng dụng cũng không dùng để huấn luyện hay mô tả thống kê
mô hình (chỉ xuất hiện trong bảng xem dữ liệu thô ở tab "Tổng quan dữ liệu").

Ở tab "Sử dụng mô hình" — chế độ tải tệp hàng loạt, tệp CSV chỉ cần chứa đúng 24 cột biến
đầu vào (X), không bắt buộc có cột `PD`.

## 4. Mô tả các tab

- **⚙️ Sidebar — Cấu hình & Tải dữ liệu:** tải tệp CSV; chọn tỷ lệ tập test và `random_state`
  (mặc định đúng theo notebook gốc: 0.2 và 32); điều chỉnh hệ số điều chuẩn `C`; tham số nâng cao
  `max_iter`, `solver` (mặc định đúng giá trị mặc định của scikit-learn vì notebook gốc dùng
  `LogisticRegression()` không chỉnh tham số). Nút **"🚀 Huấn luyện mô hình"** là điểm duy nhất
  kích hoạt việc train.
- **📊 Tổng quan dữ liệu:** kích thước dữ liệu (số dòng/cột/dung lượng), xem nhanh dữ liệu thô,
  thống kê mô tả (`describe()`) cho 24 biến đầu vào và biến mục tiêu `PD`.
- **📈 Trực quan hóa dữ liệu:** biểu đồ cột phân phối biến mục tiêu `PD`, cùng tối đa 3 biến đầu
  vào do người dùng chọn (mặc định `TC1`, `NL1`, `V1` — đại diện 3 nhóm), bố trí lưới 2×2.
- **🎯 Kết quả huấn luyện & kiểm định:** Accuracy, Precision, Recall, F1-score, ROC-AUC, ma trận
  nhầm lẫn, đường cong ROC, báo cáo phân loại chi tiết, và hệ số hồi quy (mức độ ảnh hưởng của
  từng tiêu chí) — chỉ hiển thị sau khi huấn luyện.
- **🔮 Sử dụng mô hình:** hai chế độ dự báo trên mô hình đã huấn luyện —
  1. **Nhập trực tiếp:** nhập điểm khảo sát cho từng tiêu chí (form theo nhóm 5C), trả về nhãn
     dự báo và xác suất rủi ro/không rủi ro (giống logic `predict_proba` trong notebook gốc).
  2. **Tải tệp dữ liệu:** tải CSV chứa đúng 24 cột đầu vào, kiểm tra thiếu cột, dự báo hàng loạt
     và tải kết quả về dưới dạng CSV (mã hóa `utf-8-sig`).

## 5. Ghi chú kỹ thuật

- Mô hình, tập kết quả kiểm định và các cột đặc trưng được lưu trong `st.session_state` nên khi
  chuyển qua lại giữa các tab, ứng dụng **không huấn luyện lại**.
- Ứng dụng không sử dụng `localStorage`/`sessionStorage` (không áp dụng cho kiến trúc Streamlit).
- Nếu tệp CSV tải lên thiếu cột bắt buộc, sai định dạng, hoặc rỗng, ứng dụng sẽ hiển thị thông
  báo lỗi rõ ràng thay vì crash.
