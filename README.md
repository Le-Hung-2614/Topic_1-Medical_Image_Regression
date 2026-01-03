# Topic_1-Medical-Image-Regression

## Medical_Image_Regression/
│
├── data/                        # Chứa dữ liệu (Metadata)
│   ├── raw/                     # Dữ liệu thô (tải từ link) - KHÔNG UPLOAD ẢNH LÊN GIT
│   │   ├── images/              # Thư mục chứa 100k ảnh (hoặc link shortcut tới chỗ lưu ảnh)
│   │   └── Data_Entry_2017.csv  # File CSV chứa nhãn (Age, Labels,...)
│   └── processed/               # Dữ liệu đã làm sạch (nếu có tạo file csv mới sau khi lọc tuổi)
│       ├── train.csv            # File csv đã chia tập Train
│       ├── val.csv              # File csv đã chia tập Validation
│       └── test.csv             # File csv đã chia tập Test
│
├── notebooks/                   # Các file Jupyter Notebook để chạy thử nghiệm nhanh
│   ├── 01_EDA.ipynb             # Phân tích dữ liệu: Vẽ biểu đồ phân bố tuổi, lọc nhiễu
│   └── 02_Demo_Test.ipynb       # Load model đã train để test thử trên vài ảnh
│
├── src/                         # Source code chính (Logic của dự án)
│   ├── __init__.py
│   ├── config.py                # File cấu hình (Hyperparameters: LR, Batch Size, Epochs...)
│   ├── dataset.py               # Class xử lý dữ liệu (Custom Dataset & DataLoader)
│   ├── model.py                 # Định nghĩa kiến trúc CNN + GAP + Linear
│   ├── train.py                 # File chính để chạy huấn luyện (Training Loop)
│   ├── evaluate.py              # File chạy đánh giá (tính MAE/MSE trên tập Test)
│   └── utils.py                 # Các hàm phụ trợ (vẽ biểu đồ loss, tính metric...)
│
├── saved_models/                # Nơi lưu file trọng số model (.pth hoặc .h5)
│   ├── cnn_gap_mse.pth          # Model chạy với MSE loss
│   └── cnn_gap_mae.pth          # Model chạy với MAE loss
│
├── results/                     # Kết quả đầu ra để đưa vào báo cáo
│   ├── loss_curve_mse.png       # Biểu đồ Loss
│   ├── prediction_scatter.png   # Biểu đồ phân tán (Thực tế vs Dự đoán)
│   └── logs.txt                 # File log ghi lại quá trình train
│
├── requirements.txt             # Các thư viện cần thiết (torch, pandas, numpy, scikit-learn...)
└── README.md                    # Hướng dẫn cách chạy code cho giáo viên/bạn bè
