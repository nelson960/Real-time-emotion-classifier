import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch_ema import ExponentialMovingAverage
from timm.data.mixup import Mixup
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ----------------- Settings -----------------
DATA_DIR = "/Users/nelson/py/paper_impl/emotion_classifier/fane_dataset"
SAVE_DIR = "models"
NUM_CLASSES = 9
IMG_SIZE = 256
EPOCHS = 50
BATCH_SIZE = 32
GRAD_ACCUM = 4
USE_EMA = True
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ----------------- Custom Loss -----------------
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta, gamma, class_counts):
        super().__init__()
        class_counts = torch.tensor(class_counts, dtype=torch.float32).to(device)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * NUM_CLASSES
        self.class_weights = weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ----------------- Temperature Scaling -----------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(model, val_loader):
    scaler = TemperatureScaler().to(device)
    model.eval().to(device)
    logits_list, labels_list = [], []

    if ema:
        ema.store()
        ema.copy_to(model.parameters())

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits_list.append(model(x))
            labels_list.append(y)

    if ema:
        ema.restore()

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    print(f"\U0001f321️ Calibrated Temp: {scaler.temperature.item():.4f}")
    return scaler

# ----------------- Data Transforms -----------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ----------------- Class Weights + Loss -----------------
train_targets = np.array(train_ds.targets)
class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
criterion = ClassBalancedFocalLoss(beta=0.999, gamma=1.5, class_counts=class_counts)

print("Train classes:", train_ds.classes)
print("Validation classes:", val_ds.classes)

assert train_ds.classes == val_ds.classes, "Class mismatch between train and validation sets!"

# ----------------- Model -----------------
model = create_model("mobilevitv2_100", pretrained=True)
feature_dim = model.get_classifier().in_features
model.reset_classifier(NUM_CLASSES)

for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
model.to(device)

# ----------------- Optimizer and Scheduler -----------------
optimizer = torch.optim.AdamW([
    {"params": model.get_classifier().parameters(), "lr": 3e-4, "initial_lr": 3e-4, "max_lr": 3e-4, "min_lr": 3e-5}
])
steps_per_epoch = len(train_loader) // GRAD_ACCUM
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, pct_start=0.3, cycle_momentum=False
)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995) if USE_EMA else None
mixup_fn = Mixup(
    mixup_alpha=0.2, 
    cutmix_alpha=0.2, 
    prob=0.5,
    num_classes=NUM_CLASSES
)

# ----------------- Unfreezing Helper -----------------
def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True

# ----------------- Training Loop -----------------
best_f1 = 0
best_model_path = None
for epoch in range(EPOCHS):
    print(f"\n\U0001f4c5 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    optimizer.zero_grad()

    if epoch == 0:
        print("Unfreezing Stage 4 (deepest)")
        unfreeze(model.stages[4])
        optimizer.add_param_group({"params": model.stages[4].parameters(), "lr": 1e-4, "initial_lr": 1e-4, "max_lr":1e-4, "min_lr":1e-5})
    if epoch == 1:
        print("Unfreezing Stage 3")
        unfreeze(model.stages[3])
        optimizer.add_param_group({"params": model.stages[3].parameters(), "lr": 1e-4, "initial_lr": 1e-4, "max_lr":1e-4, "min_lr":1e-5})
    if epoch == 3:
        print("Unfreezing Stage 2")
        unfreeze(model.stages[2])
        optimizer.add_param_group({"params": model.stages[2].parameters(), "lr": 1e-4, "initial_lr": 1e-4, "max_lr":1e-4, "min_lr":1e-5})
    if epoch == 5:
        print("Unfreezing Stage 1 (earliest layers)")
        unfreeze(model.stages[1])
        optimizer.add_param_group({"params": model.stages[1].parameters(), "lr": 5e-5, "initial_lr": 5e-5, "max_lr": 5e-5, "min_lr": 5e-6})

    running_loss = 0.0
    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"\U0001fa9d️ Train {epoch+1:02d}")):
        x, y = x.to(device), y.to(device)
        x, y = mixup_fn(x, y)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        running_loss += loss.item()

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

    
    avg_loss = running_loss / len(train_loader)

    if ema:
        ema.store()
        ema.copy_to(model.parameters())

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Validate {epoch+1:02d}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds += logits.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()

    if ema:
        ema.restore()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    print(f"✅ Epoch {epoch+1:02d} | 🔁 LR: {scheduler.get_last_lr()[0]:.2e} | 🎯 Acc: {acc:.4f} | 🧠 F1: {f1:.4f} | 📉 Loss: {avg_loss:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        if ema:
            ema.store()
            ema.copy_to(model.parameters())
            best_model_path = os.path.join(SAVE_DIR, f"best_model_ema_{epoch+1}.pth")  # ⬅️ store exact path
            torch.save(model.state_dict(), best_model_path)
            ema.restore()
        else:
            best_model_path = os.path.join(SAVE_DIR, f"best_model_{epoch+1}.pth")  # ⬅️ store exact path
            torch.save(model.state_dict(), best_model_path)
        print(f"🔥 Saved best model at epoch {epoch+1} with F1 = {f1:.4f}")

# ----------------- Final Evaluation -----------------
if best_model_path is not None and os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    calibrate_temperature(model, val_loader)

    print("\n📊 Final Classification Report:")
    # Evaluate with the best model...
else:
    print("❌ No best model was saved during training.")

print("Final Classification Report:")
print(classification_report(targets, preds, target_names=val_ds.classes))
cm = confusion_matrix(targets, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_ds.classes, yticklabels=val_ds.classes)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

