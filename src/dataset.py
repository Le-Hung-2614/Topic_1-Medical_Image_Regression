import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import src.config as config

# --- CLASS DATASET ---
class AgeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Đường dẫn đến file train.csv/val.csv đã clean.
            transform (callable, optional): Các bước xử lý ảnh (Resize, Augment...).
        """
        # Đọc file CSV
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Lấy thông tin từ dòng thứ idx
        row = self.df.iloc[idx]
        img_path = row['path']        # Lấy địa chỉ ảnh trực tiếp (không cần map nữa)
        age = row['Patient Age']      # Lấy nhãn tuổi

        # 2. Đọc ảnh
        try:
            # Mở ảnh và convert sang RGB (ResNet yêu cầu 3 kênh màu)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Nếu file lỗi (dù rất hiếm vì đã clean), in cảnh báo
            print(f"LỖI ĐỌC ẢNH: {img_path} | Lỗi: {e}")
            # Trả về ảnh đen để code không bị crash (nhưng sẽ báo lỗi để biết)
            image = Image.new('RGB', (224, 224))

        # 3. Áp dụng Transform (Resize, Augmentation...)
        if self.transform:
            image = self.transform(image)

        # 4. Trả về: Ảnh (Tensor) và Tuổi (Tensor float)
        return image, torch.tensor([float(age)])

# --- HÀM TẠO DATALOADER (HỖ TRỢ AUGMENTATION) ---
def get_dataloaders(augment=False, batch_size=None):
    """
    Tạo DataLoader cho Train, Val, Test.
    Args:
        augment (bool): Nếu True -> Bật chế độ Data Augmentation (Xoay, Lật...).
        batch_size (int): Nếu không truyền vào thì lấy từ config.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    print(f"Đang tạo DataLoader... (Augmentation={augment})")

    # 1. Cấu hình Transform
    # Transform chuẩn cho Val/Test (Chỉ Resize + Normalize)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transform cho Train (Có thể thêm Augmentation)
    if augment:
        print("Kích hoạt: RandomFlip, Rotation, ColorJitter...")
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # Lật ảnh ngẫu nhiên
            transforms.RandomRotation(degrees=15),  # Xoay ảnh +/- 15 độ
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Chỉnh sáng tối
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform # Không augment thì giống val

    # 2. Khởi tạo Dataset từ các file CSV đã clean
    # Lưu ý: Đảm bảo src.config trỏ đúng vào folder 'processed'
    train_ds = AgeDataset(config.TRAIN_CSV, transform=train_transform)
    val_ds = AgeDataset(config.VAL_CSV, transform=val_transform)
    test_ds = AgeDataset(config.TEST_CSV, transform=val_transform)

    # 3. Tạo DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Đã load xong: Train({len(train_ds)}), Val({len(val_ds)}), Test({len(test_ds)})")
    
    return train_loader, val_loader, test_loader
