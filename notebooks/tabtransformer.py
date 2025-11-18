import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score # Thêm để đánh giá
from tab_transformer_pytorch import TabTransformer

# Hàm để set seed cho mọi thứ
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # nếu dùng nhiều GPU
    np.random.seed(seed)
    random.seed(seed)
    # Đảm bảo các thuật toán CUDA chạy giống hệt nhưng có thể làm chậm quá trình huấn luyện
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

class HeartDataset(Dataset):
    def __init__(self, x_cat, x_num, y):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_num[idx], self.y[idx].unsqueeze(-1)

# --- Tải dữ liệu ---
data = pd.read_csv("../data/heart.csv")

# --- Xác định features ---
features_cols = [col for col in data.columns if col not in ["HeartDisease", "RestingBP", "RestingECG"]]
target = data["HeartDisease"]

categorical_features = []
numerical_features = []
for col in features_cols:
    if len(data[col].unique()) > 6:
        numerical_features.append(col)
    else:
        categorical_features.append(col)

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# --- Tiền xử lý ---
categories_cardinalities = [] 
label_encoders = {}
df = data.copy(deep=True) 

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    categories_cardinalities.append(len(le.classes_))

print(f"Cardinalities: {categories_cardinalities}")

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# --- Tách X, y ---
y = target.values
x_cat = df[categorical_features].values
x_num = df[numerical_features].values

# --- Tách Train/Test ---
x_cat_train, x_cat_test, x_num_train, x_num_test, y_train, y_test = train_test_split(
    x_cat, x_num, y, test_size=0.2, random_state=42
)

# --- Tạo Dataset & DataLoader ---
train_dataset = HeartDataset(x_cat_train, x_num_train, y_train)
test_dataset = HeartDataset(x_cat_test, x_num_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Khởi tạo Model & Các tham số ---
num_continuous = len(numerical_features)
output_dim = 1
embed_dim = 32
num_heads = 4
num_layers = 4
learning_rate = 1e-3
num_epochs = 20

### KHỞI TẠO MODEL ###
model = TabTransformer(
    categories = categories_cardinalities,
    num_continuous = num_continuous,
    dim = embed_dim,
    depth = num_layers,
    heads = num_heads,
    dim_out = output_dim,
    mlp_hidden_mults = (4, 2) # Cấu hình MLP head (ví dụ: [dim*4, dim*2])
)
####################################

# Loss Function và Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# --- Vòng lặp Training ---
print("Bắt đầu huấn luyện...")
model.train() 
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_cat, batch_num, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_cat, batch_num)
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

print("Huấn luyện hoàn tất!")

# --- Đánh giá Model ---
print("Bắt đầu đánh giá...")
model.eval() 

all_preds_scores = []
all_preds_classes = []
all_targets = []

with torch.no_grad(): 
    for batch_cat, batch_num, batch_y in test_loader:
        outputs = model(batch_cat, batch_num)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds_scores.extend(probs.cpu().numpy())
        all_preds_classes.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

all_preds_classes = np.array(all_preds_classes).flatten()
all_targets = np.array(all_targets).flatten()
all_preds_scores = np.array(all_preds_scores).flatten()

accuracy = (all_preds_classes == all_targets).mean()
auc_score = roc_auc_score(all_targets, all_preds_scores)

print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
print(f'Test ROC-AUC Score: {auc_score:.4f}\n')
print("Classification Report:")
print(classification_report(all_targets, all_preds_classes, target_names=['No Disease (0)', 'Disease (1)']))