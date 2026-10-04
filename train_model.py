import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ========================
# 1. CONFIG
# ========================
DATA_DIR = "data"   # your dataset folder
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ========================
# 2. TRANSFORMS (VERY IMPORTANT)
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ========================
# 3. DATASET
# ========================
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_data.classes)

# ========================
# 4. MODEL (BEST: ResNet50)
# ========================
model = models.resnet50(pretrained=True)

# Freeze layers (important for small dataset)
for param in model.parameters():
    param.requires_grad = False

# Change final layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model = model.to(device)

# ========================
# 5. LOSS + OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ========================
# 6. TRAINING LOOP
# ========================
best_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- VALIDATION ----
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"Loss: {train_loss:.4f} | Val Accuracy: {acc:.4f}")

    # SAVE BEST MODEL
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model Saved!")

print("\n🎯 Training Complete. Best Accuracy:", best_acc)