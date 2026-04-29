# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import relu
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import torchinfo


# %%
DATA_PATH = "/Volumes/LabData/BMED6460/beetle-master/data/"
IMAGE_DIR = DATA_PATH + "images/development/output/beetle-patches/img_patches-256"
MASK_DIR = DATA_PATH + "images/development/output/beetle-patches/mask_patches-256"
OUTPUT_DIR = DATA_PATH + "images/development/output/beetle-patches/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

LABEL_MAP = {
    "unannotated": 0,
    "other": 1,
    "non-invasive epithelium": 2,
    "invasive epithelium": 3,
    "necrosis": 4
}
NUM_CLASSES = len(LABEL_MAP)


BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-4
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# From observed distribution
observed = np.array([7.7, 42.1, 27.9, 16.9])  # classes 1,2,3,4
weights  = 1.0 / observed
weights  = weights / weights.mean()
print(weights)

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
full_weights  = torch.cat([torch.tensor([0.0]).to(DEVICE), class_weights])



# %%

COLORS = {
    0: (0, 0, 0),          # unannotated (black)
    1: (173, 216, 230),    # other (light blue)
    2: (255, 255, 0),      # non-invasive epithelium (yellow)
    3: (255, 105, 180),    # invasive epithelium (pink)
    4: (128, 0, 128)       # necrosis (purple)
}

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        rgb[mask == cls] = color

    return rgb


# %%
# MODEL
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # ---- Encoder ----
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(128, 256)

        # ---- Decoder ----
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.dropout = nn.Dropout2d(0.3)
        self.out_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # 32
        e2 = self.enc2(self.pool(e1))  # 64
        e3 = self.enc3(self.pool(e2))  # 128

        b = self.bottleneck(self.pool(e3))  # 256

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(self.dropout(d1))



# %%
# DATASET
class HistopathologyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size=256, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def random_crop(self, img, mask):
        h, w = img.shape[:2]
        cs = self.crop_size

        if h < cs or w < cs:
            img = cv2.resize(img, (cs, cs))
            mask = cv2.resize(mask, (cs, cs), interpolation=cv2.INTER_NEAREST)
            return img, mask

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)

        img = img[y:y+cs, x:x+cs]
        mask = mask[y:y+cs, x:x+cs]

        return img, mask
    
    def _augment(self, img_t, mask_t):
        """
        img_t : C*H*W float tensor
        mask_t: H*W   long  tensor
        All ops are label-safe (same transform applied to both).
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img_t  = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t.unsqueeze(0)).squeeze(0)

        # Random vertical flip
        if random.random() > 0.5:
            img_t  = TF.vflip(img_t)
            mask_t = TF.vflip(mask_t.unsqueeze(0)).squeeze(0)

        # Random 90° rotation
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            img_t  = torch.rot90(img_t,  k, dims=[1, 2])
            mask_t = torch.rot90(mask_t, k, dims=[0, 1])

        # Color jitter — image only (stain variation simulation)
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1)
        img_t = jitter(img_t)

        return img_t, mask_t

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Failed to load index {idx}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5]*3, [0.5]*3)
        mask  = torch.from_numpy(mask).long()

        if self.augment:
            image, mask = self._augment(image, mask)  # consistent names

        return image, mask
    
# %%
# LOSSES
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)

        total, count = 0.0, 0
        for cls in range(1, self.num_classes):
            pred   = probs[:, cls] * valid
            target = ((targets == cls) & valid).float()

            inter = (pred * target).sum()
            denom = pred.sum() + target.sum()

            total += (2 * inter + self.smooth) / (denom + self.smooth)
            count += 1

        return 1 - total / count


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight_tensor=None):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=0, weight=ce_weight_tensor)
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)
    
# %%
# # METRICS
# def dice_per_class(logits, targets, num_classes):
#     preds = torch.argmax(logits, dim=1)

#     dice_stats = {}

#     for cls in range(1, num_classes):
#         pred = (preds == cls)
#         target = (targets == cls)

#         inter = (pred & target).sum().item()
#         union = pred.sum().item() + target.sum().item()

#         dice_stats[cls] = (inter, union)

#     return dice_stats

# %%
# DATA PREP
# Ignore non-image files
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png") and not f.startswith(".")
]
all_images, all_masks = [], []

for f in image_files:
    img_path = os.path.join(IMAGE_DIR, f)
    mask_path = os.path.join(MASK_DIR, f)

    if os.path.exists(mask_path):
        all_images.append(img_path)
        all_masks.append(mask_path)

print(f"Found {len(all_images)} valid pairs")



# --- use 50% of data ---
subset_fraction = 0.5
num_samples = int(len(all_images) * subset_fraction)

indices = list(range(len(all_images)))
random.seed(42)
random.shuffle(indices)

selected_idx = indices[:num_samples]

all_images = [all_images[i] for i in selected_idx]
all_masks  = [all_masks[i] for i in selected_idx]

print(f"Using {len(all_images)} samples (50% of dataset)")

# --- use 50% of data ---



train_img, val_img, train_mask, val_mask = train_test_split(
    all_images, all_masks, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    HistopathologyDataset(train_img, train_mask, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    HistopathologyDataset(val_img, val_mask, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
# %%
def pixel_accuracy(preds, masks):
    # ignore background class (0)
    valid = (masks != 0)

    # if no foreground pixels exist, skip this batch
    if valid.sum() == 0:
        return 0.0

    correct = (preds == masks) & valid

    return correct.sum().item() / valid.sum().item()

def dice_per_class(preds, targets, num_classes):
    dice_scores = {}

    for cls in range(1, num_classes):  # ignore 0
        pred = (preds == cls)
        target = (targets == cls)

        intersection = (pred & target).sum().item()
        union = pred.sum().item() + target.sum().item()

        if union == 0:
            dice_scores[cls] = None  # class not present
        else:
            dice_scores[cls] = (2 * intersection) / union

    return dice_scores


# %%
# =========================
# TRAINING
# =========================
model = TinyUNet(NUM_CLASSES).to(DEVICE)
criterion = CombinedLoss(NUM_CLASSES, ce_weight_tensor=full_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# %%

train_losses = []
val_losses = []

train_mean_dice_list = []
val_mean_dice_list = []

train_dice_history = {cls: [] for cls in range(1, NUM_CLASSES)}
val_dice_history = {cls: [] for cls in range(1, NUM_CLASSES)}

for epoch in range(EPOCHS):

    # =========================
    # TRAIN
    # =========================
    model.train()
    train_loss = 0

    train_dice_totals = {cls: 0 for cls in range(1, NUM_CLASSES)}
    train_counts = {cls: 0 for cls in range(1, NUM_CLASSES)}

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for imgs, masks in train_pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # ---- Dice ----
        preds = torch.argmax(outputs, dim=1)
        dice_scores = dice_per_class(preds, masks, NUM_CLASSES)

        for cls, score in dice_scores.items():
            if score is not None:
                train_dice_totals[cls] += score
                train_counts[cls] += 1

        # mean dice for progress bar
        valid_scores = [v for v in dice_scores.values() if v is not None]
        mean_dice = np.mean(valid_scores) if len(valid_scores) > 0 else 0

        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{mean_dice:.4f}"
        })

    train_loss /= len(train_loader)

    # compute epoch train dice
    train_dice_avg = {
        cls: train_dice_totals[cls] / train_counts[cls]
        if train_counts[cls] > 0 else 0
        for cls in train_dice_totals
    }
    mean_train_dice = np.mean(list(train_dice_avg.values()))

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    val_loss = 0

    val_dice_totals = {cls: 0 for cls in range(1, NUM_CLASSES)}
    val_counts = {cls: 0 for cls in range(1, NUM_CLASSES)}

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

    visual_samples_saved = 0
    max_vis = 5

    with torch.no_grad():
        for imgs, masks in val_pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            # ---- Dice ----
            preds = torch.argmax(outputs, dim=1)
            dice_scores = dice_per_class(preds, masks, NUM_CLASSES)
            
            # if epoch == 0:
            #     unique, cnts = torch.unique(preds, return_counts=True)
            #     print("Predicted classes:", dict(zip(unique.tolist(), cnts.tolist())))

            for cls, score in dice_scores.items():
                if score is not None:
                    val_dice_totals[cls] += score
                    val_counts[cls] += 1

            # mean dice for tqdm
            valid_scores = [v for v in dice_scores.values() if v is not None]
            mean_dice = np.mean(valid_scores) if len(valid_scores) > 0 else 0

            val_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{mean_dice:.4f}"
            })

            # =========================
            # VISUALIZATION
            # =========================
            if visual_samples_saved < max_vis:
                for i in range(imgs.size(0)):

                    img = imgs[i].cpu().permute(1, 2, 0).numpy()
                    img = (img * 0.5) + 0.5

                    gt_mask = masks[i].cpu().numpy()
                    pred_mask = preds[i].cpu().numpy()

                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                    ax[0].imshow(img)
                    ax[0].set_title("Input")
                    ax[0].axis("off")

                    ax[1].imshow(mask_to_rgb(gt_mask))
                    ax[1].set_title("Ground Truth")
                    ax[1].axis("off")

                    ax[2].imshow(mask_to_rgb(pred_mask))
                    ax[2].set_title("Prediction")
                    ax[2].axis("off")

                    save_path = os.path.join(
                        VIS_DIR,
                        f"epoch_{epoch+1}_sample_{visual_samples_saved}.png"
                    )

                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)

                    visual_samples_saved += 1

                    if visual_samples_saved >= max_vis:
                        break

    val_loss /= len(val_loader)

    # compute epoch val dice
    val_dice_avg = {
        cls: val_dice_totals[cls] / val_counts[cls]
        if val_counts[cls] > 0 else 0
        for cls in val_dice_totals
    }
    mean_val_dice = np.mean(list(val_dice_avg.values()))

    # Step scheduler
    scheduler.step(val_loss)

    # =========================
    # LOGGING
    # =========================
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_mean_dice_list.append(mean_train_dice)
    val_mean_dice_list.append(mean_val_dice)

    for cls in train_dice_avg:
        train_dice_history[cls].append(train_dice_avg[cls])

    for cls in val_dice_avg:
        val_dice_history[cls].append(val_dice_avg[cls])

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Dice: {mean_train_dice:.4f} | Val Dice: {mean_val_dice:.4f}")

    print("Train Dice per class:")
    for cls in train_dice_avg:
        print(f"  Class {cls}: {train_dice_avg[cls]:.4f}")

    print("Val Dice per class:")
    for cls in val_dice_avg:
        print(f"  Class {cls}: {val_dice_avg[cls]:.4f}")


# %%
# SAVE MODEL
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "unet_histopathology.pth"))
print("Training complete.")

import matplotlib.pyplot as plt

epochs = range(1, EPOCHS + 1)

# =========================
# LOSS PLOT
# =========================
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.show()


# =========================
# DICE PLOT (REPLACES ACCURACY)
# =========================
plt.figure()
plt.plot(epochs, train_mean_dice_list, label="Train Dice")
plt.plot(epochs, val_mean_dice_list, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Training vs Validation Dice")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "dice_curve.png"))
plt.show()

# %%

# =========================
# PER-CLASS DICE PLOT
# =========================
CLASS_NAMES = {1: "other", 2: "non-inv epi", 3: "inv epi", 4: "necrosis"}
CLASS_COLORS = {1: "blue", 2: "orange", 3: "green", 4: "red"}

plt.figure()

for cls in train_dice_history:
    color = CLASS_COLORS[cls]
    plt.plot(epochs, train_dice_history[cls], linestyle="--", color=color, label=f"Train {CLASS_NAMES[cls]}")
    plt.plot(epochs, val_dice_history[cls],   linestyle="-",  color=color, label=f"Val {CLASS_NAMES[cls]}")

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Per-Class Dice Scores")
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1, 1),  # places legend to the right of the plot
    fontsize=7
)
plt.tight_layout()  # prevents the legend from being clipped
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_dice.png"))
plt.show()
# %%


summary = torchinfo.summary(model, input_size=(1, 3, 256, 256))
print(summary)
# %%
