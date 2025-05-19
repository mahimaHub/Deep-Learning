import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parameters
DATA_DIR = r"C:\DL_project\dataset"
BATCH_SIZE = 8
EPOCHS = 25
IMG_SIZE = 224
NUM_CLASSES = 3
device = torch.device("cpu")

class_labels = ["high_fall", "medium_fall", "low_fall"]

# Augmented Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomPerspective(0.4),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load and Stratified Split
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
targets = [label for _, label in full_dataset.samples]
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weights
targets_array = np.array(targets)
class_weights = compute_class_weight('balanced', classes=np.unique(targets_array), y=targets_array)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Model: EfficientNet-B3 + classifier
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Training Loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {running_loss:.4f} - Accuracy: {train_accuracy:.2%}")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2%}")
    scheduler.step(val_accuracy)

# Save
torch.save(model.state_dict(), "efficientnet_precipitation_model.pth")
print("✅ Model saved.")

# =====================
# ✅ Evaluation
# =====================
model.eval()
y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.numpy())
        y_scores.extend(probs.cpu().numpy())

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)
print("\nLabeled Confusion Matrix:")
for i, row in enumerate(cm):
    print(f"{class_labels[i]:<15} → {row}")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar(cax)
ax.set_xticks(range(len(class_labels)))
ax.set_yticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45)
ax.set_yticklabels(class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, roc_auc = {}, {}, {}
y_true_bin = np.eye(NUM_CLASSES)[y_true]
y_scores = np.array(y_scores)

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()
print("📈 Plots saved: confusion_matrix.png, roc_curves.png")
