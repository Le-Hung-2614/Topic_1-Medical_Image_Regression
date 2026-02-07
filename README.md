# Topic_1-Medical-Image-Regression - Tumor Size Prediction using Deep Learning
> Dự đoán kích thước khối u phổi trên ảnh X-quang sử dụng mạng nơ-ron tích chập (CNN).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Giới thiệu (Overview)
Dự án này tập trung giải quyết bài toán **Hồi quy (Regression)** trong y tế: Dự đoán chính xác đường kính khối u (đơn vị mm) từ ảnh chụp X-quang lồng ngực. Thay vì chỉ phân loại (có bệnh/không bệnh), mô hình cung cấp thông tin định lượng để hỗ trợ bác sĩ chẩn đoán mức độ nghiêm trọng.

**Thách thức:**
- Dữ liệu hạn chế (Small Dataset ~115 ảnh).
- Sự mất cân bằng dữ liệu (Data Imbalance) giữa các kích thước u.
- Nhiễu ảnh y tế và sự đa dạng về vị trí/hình dạng khối u.

## 🛠️ Phương pháp & Kỹ thuật (Methodology)

### 1. Xử lý dữ liệu (Data Preprocessing)
- **Hợp nhất dữ liệu:** Kết hợp thông tin tọa độ Pixel và PixelSpacing để tính kích thước thực (mm).
- **Stratified Group Split:** Chia tập Train/Val/Test đảm bảo cân bằng phân phối kích thước u, tránh hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các bệnh nhân.
- **Data Augmentation:** Áp dụng các kỹ thuật: Horizontal Flip, Rotation, Affine Translation, Gaussian Blur để tăng cường dữ liệu.

### 2. Kiến trúc Mô hình (Model Architecture)
Sử dụng **ResNet18** làm xương sống (Backbone) với chiến lược **Transfer Learning**:
- **Backbone:** ResNet18 (Pretrained on ImageNet).
- **Regression Head (Custom):**
  - Flatten -> Linear (512 -> 256) -> ReLU -> Dropout (0.5) -> Linear (1).
  - Output: 1 giá trị thực (Scalar) đại diện cho kích thước mm.
- **Loss Function:** L1 Loss (MAE) để tối ưu hóa sai số tuyệt đối.

### 3. Huấn luyện (Training Strategy)
- **Fine-tuning:** Unfreeze toàn bộ các lớp (Full Unfreeze) để mô hình học lại các đặc trưng y tế mức thấp.
- **Optimizer:** Adam với Learning Rate nhỏ ($1e-5$) và Weight Decay ($1e-3$) để chống Overfitting.
- **Scheduler:** ReduceLROnPlateau.

## 📊 Kết quả Thực nghiệm (Experimental Results)

Mô hình đạt được kết quả khả quan trên tập Test độc lập:

| Metric | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **8.44 mm** | Sai số trung bình tuyệt đối |
| **MPAE (Mean % Error)** | **~25%** | Sai số phần trăm trung bình |
| **RMSE (Root Mean Sq. Error)** | 11.2 mm | Độ lệch chuẩn của sai số |

### Demo Dự đoán
*(Kết quả trực quan trên một ca bệnh cụ thể)*

![Prediction Demo](images/demo_prediction.png)

> **Nhận xét:** Với khối u thực tế 15.3mm, mô hình dự đoán 11.7mm (Lệch 3.6mm). Đây là mức sai số chấp nhận được trong sàng lọc sơ bộ.

## 📈 Phân tích Biểu đồ (Analysis)

### 1. Biểu đồ Loss (Train vs Val)
Cho thấy mô hình hội tụ tốt, không bị Overfitting nặng nhờ Dropout và Data Augmentation.
![Loss Chart](images/loss_chart.png)

### 2. Biểu đồ Tương quan (Scatter Plot)
Mô hình dự đoán tốt các khối u kích thước nhỏ và trung bình (< 40mm). Các khối u quá lớn (Outliers) có xu hướng bị dự đoán thấp hơn thực tế.
![Scatter Plot](images/scatter_plot.png)

## 📂 Cấu trúc Thư mục (Project Structure)
├── data/ # Thư mục chứa dữ liệu (Không public) 
├── models/ # Chứa file trọng số (.pth) │ 
├── tumor_model_mae.pth # Best Model 
├── notebooks/ # Jupyter Notebooks phân tích & train 
├── images/ # Hình ảnh kết quả để hiển thị README 
├── requirements.txt # Các thư viện cần thiết 
├── train.py # Script huấn luyện chính 
├── evaluate.py # Script đánh giá kiểm thử └── README.md # Tài liệu dự án