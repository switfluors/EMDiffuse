import os
import h5py
from tifffile import imwrite
from tqdm import tqdm
import numpy as np

# --- Load data ---
filepath = "./datasets/Training_Perlin160k_Pperlin50k_MatchNorm.mat"
with h5py.File(filepath, "r") as f:
    gt_data = f["GTspt"][:]        # Shape: (N, 16, 128)
    input_data = f["sptimg4"][:]   # Shape: (N, 16, 128)

# --- Helper: normalize to uint16 ---
def normalize_to_uint16(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max() + 1e-8  # Avoid divide-by-zero
    return (img * 65535).astype(np.uint16)

# --- Prepare output directories ---
base_dir = os.path.join(os.path.dirname(filepath), "Spec_dataset_160k")
gt_dir = os.path.join(base_dir, "train_gt")
wf_dir = os.path.join(base_dir, "train_wf")

os.makedirs(gt_dir, exist_ok=True)
os.makedirs(wf_dir, exist_ok=True)

# --- Save GT images ---
print("Saving ground truth images...")
for i in tqdm(range(len(gt_data)), desc="GT"):
    folder_id = i // 1000  # Folder grouping: 1000 images per folder
    subdir = os.path.join(gt_dir, str(folder_id), "Spec__4w_04")
    os.makedirs(subdir, exist_ok=True)

    img = normalize_to_uint16(gt_data[i, :, :].transpose(1, 0))  # Transpose to (128, 16)
    imwrite(os.path.join(subdir, f"{i}.tif"), img)

# --- Save Input WF images ---
print("Saving input (WF) images...")
for i in tqdm(range(len(input_data)), desc="WF"):
    folder_id = i // 1000
    subdir = os.path.join(wf_dir, str(folder_id), "Spec__4w_04")
    os.makedirs(subdir, exist_ok=True)

    img = normalize_to_uint16(input_data[i, :, :].transpose(1, 0))
    imwrite(os.path.join(subdir, f"{i}.tif"), img)

print("✅ All images saved.")