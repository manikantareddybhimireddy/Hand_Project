import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image

from sklearn.metrics import f1_score, classification_report

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder(root="Dataset_Object_Detection/train", transform=transform)
val_dataset = datasets.ImageFolder(root="Dataset_Object_Detection/val", transform=transform)
test_dataset = datasets.ImageFolder(root="Dataset_Object_Detection/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)


# Load pre-trained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification

model = model.to(device)

print(torch.cuda.is_available())  # Should print True
print(next(model.parameters()).device)  # Should print cuda:0

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.backends.cudnn.benchmark = True

scaler = torch.cuda.amp.GradScaler()

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed Precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Training loop
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "hand_xray_model.pth") # Model Name
print("Model saved successfully!")

# Load the saved model for testing
loaded_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
loaded_model.fc = nn.Linear(num_ftrs, 2)
loaded_model.load_state_dict(torch.load("hand_xray_model.pth")) 
loaded_model = loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully!")

# Function to test with a new image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = loaded_model(image)
        _, predicted = torch.max(output, 1)
    
    class_names = ["Hand X-ray", "Other"]
    return class_names[predicted.item()]

# Example usage
image_path = "image.jpg"  # Change this to your image path
result = predict_image(image_path)
print(f"Prediction: {result}")
