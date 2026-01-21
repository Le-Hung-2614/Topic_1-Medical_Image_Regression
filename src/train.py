# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import sys

# Fix Unicode encoding issue for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import hàm load dữ liệu từ file dataset.py của bạn
# Lưu ý: file dataset.py phải nằm cùng thư mục với file này
from dataset import get_data_loaders

def build_model():
    """
    Tải model ResNet50 đã pre-train và sửa lớp cuối cho bài toán Regression (1 đầu ra)
    """
    print("[INFO] Dang tai model ResNet50 (pretrained)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Đóng băng các lớp đầu (Feature Extractor) để train nhanh hơn (tùy chọn)
    # Nếu muốn train kỹ hơn thì comment 2 dòng dưới lại
    for param in model.parameters():
        param.requires_grad = False

    # Thay đổi lớp Fully Connected cuối cùng
    # ResNet50 có in_features = 2048. Ta đổi out_features = 1 (dự đoán 1 số thực là Tuổi)
    num_ftrs = model.fc.in_features
    
    # Thêm một vài lớp để học tốt hơn
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1) # Output ra 1 giá trị duy nhất
    )
    
    return model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    since = time.time()

    val_mae_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 1000.0 # Khởi tạo sai số lớn nhất có thể

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Mỗi epoch đều có pha Train và pha Validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_mae = 0.0
            total_samples = 0

            # Lặp qua từng batch dữ liệu
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1) # Reshape labels thành (Batch_size, 1)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize chỉ khi ở pha training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Thống kê
                running_loss += loss.item() * inputs.size(0)
                # Tính MAE (Mean Absolute Error) để dễ hình dung sai số bao nhiêu tuổi
                mae = torch.abs(outputs - labels).sum()
                running_mae += mae.item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_mae = running_mae / total_samples

            print(f'{phase} Loss (MSE): {epoch_loss:.4f} | MAE (Sai so tuoi): {epoch_mae:.4f}')

            # Deep copy model nếu kết quả tốt hơn
            if phase == 'val' and epoch_mae < best_mae:
                best_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"--> Tim thay model tot hon! Sai so trung binh: {best_mae:.2f} tuoi")
            
            if phase == 'val':
                val_mae_history.append(epoch_mae)

    time_elapsed = time.time() - since
    print(f'\nHuan luyen hoan tat trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val MAE: {best_mae:.4f}')

    # Load lại weight tốt nhất
    model.load_state_dict(best_model_wts)
    return model, val_mae_history

def main():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dang su dung thiet bi: {device}")

    # 2. Load dữ liệu (Sử dụng hàm từ dataset.py của bạn)
    # Batch size có thể giảm xuống 16 nếu bị lỗi tràn RAM/VRAM
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 3. Khởi tạo model
    model = build_model()
    model = model.to(device)

    # 4. Định nghĩa Loss Function và Optimizer
    # Bài toán Regression nên dùng MSELoss (Mean Squared Error) để train
    criterion = nn.MSELoss() 
    
    # Chỉ optimize các tham số có requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 5. Bắt đầu train
    model, history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)

    # 6. Lưu model
    torch.save(model.state_dict(), "age_prediction_model.pth")
    print("[INFO] Da luu model tai 'age_prediction_model.pth'")

if __name__ == "__main__":
    main()