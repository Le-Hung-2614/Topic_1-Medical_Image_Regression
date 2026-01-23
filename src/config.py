import os

# Đường dẫn gốc
INPUT_ROOT = "/kaggle/input/data"
WORKING_DIR = "/kaggle/working"

# Đường dẫn đến dữ liệu ĐÃ CLEAN (Quan trọng!)
PROCESSED_DIR = os.path.join(WORKING_DIR, "data/processed")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV = os.path.join(PROCESSED_DIR, "val.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

# Nơi lưu model
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "saved_models/cnn_model.pth")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda"
