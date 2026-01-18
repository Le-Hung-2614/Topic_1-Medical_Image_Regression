import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# SỬA 1: Trỏ ra thư mục cha "images" (nơi chứa cả images_001, images_002...)
RAW_IMG_DIR = os.path.join(ROOT_DIR, "data", "raw", "images") 
RAW_CSV_PATH = os.path.join(ROOT_DIR, "data", "raw", "Data_Entry_2017_v2020.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

# --- PHẦN 1: HÀM XỬ LÝ & CHIA DỮ LIỆU ---
def prepare_and_save_splits():
    # Tạo folder processed nếu chưa có
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    # Nếu đã có file processed, dùng luôn không cần quét lại
    if os.path.exists(train_path):
        print(f"[INFO] Da tim thay du lieu da xu ly tai: {PROCESSED_DIR}")
        return train_path, val_path, test_path

    print("[INFO] Dang quet toan bo thu muc anh (001, 002, 003...)...")
    
    # SỬA 2: Quét toàn bộ thư mục con để tìm đường dẫn ảnh
    image_paths = {} # Dictionary: {Tên file: Đường dẫn đầy đủ}
    for root, dirs, files in os.walk(RAW_IMG_DIR):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                image_paths[file] = full_path
    
    print(f"[INFO] Tim thay tong cong {len(image_paths)} anh trong tat ca thu muc.")

    # Đọc CSV gốc
    df = pd.read_csv(RAW_CSV_PATH)
    
    # SỬA 3: Map đường dẫn thực tế vào DataFrame
    # Tạo cột 'path' mới, chứa đường dẫn đầy đủ đến ảnh
    df['path'] = df['Image Index'].map(image_paths)
    
    # Loại bỏ những dòng mà không tìm thấy ảnh (ví dụ ảnh ở images_012 chưa tải)
    df = df.dropna(subset=['path'])
    
    # Lọc tuổi hợp lệ
    df = df[(df['Patient Age'] > 0) & (df['Patient Age'] < 100)]
    
    print(f"[INFO] So luong anh hop le sau khi loc: {len(df)}")

    # Chia tập dữ liệu
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Lưu file (Lưu cả cột 'path' để Dataset class dùng luôn)
    print("[INFO] Dang luu file vao folder data/processed/...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_path, val_path, test_path

# --- PHẦN 2: CLASS DATASET ---
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # SỬA 4: Lấy đường dẫn trực tiếp từ cột 'path' đã lưu
        img_path = self.dataframe.iloc[idx]['path']
        age = self.dataframe.iloc[idx]['Patient Age']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, float(age)

# --- PHẦN 3: DATA LOADER ---
def get_data_loaders(batch_size=32):
    train_csv, val_csv, test_csv = prepare_and_save_splits()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Class Dataset mới không cần truyền RAW_IMG_DIR nữa vì đường dẫn đã có trong CSV
    train_loader = DataLoader(ChestXrayDataset(train_csv, transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ChestXrayDataset(val_csv, transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ChestXrayDataset(test_csv, transform), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# --- PHẦN 4: TEST CODE ---
if __name__ == "__main__":
    # QUAN TRỌNG: Xóa file cũ để code tạo lại file mới có chứa đường dẫn của cả 002, 003
    processed_folder = os.path.join(ROOT_DIR, "data", "processed")
    if os.path.exists(os.path.join(processed_folder, "train.csv")):
        print("!!!! Phat hien du lieu cu. Dang xoa de quet lai toan bo folder...")
        import shutil
        shutil.rmtree(processed_folder)

    train_dl, val_dl, test_dl = get_data_loaders()
    print(f"\nDa load thanh cong du lieu tu tat ca cac folder con!")
    print(f"So batch train: {len(train_dl)}")