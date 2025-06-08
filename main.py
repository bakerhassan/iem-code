import torch, argparse, time
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
from metrics import *
from models import *
from datasets import *

parser = argparse.ArgumentParser(description='Inpainting Error Maximization')
parser.add_argument('data_path', type=str, default='/lustre/cniel/data/AI4Shipwrecks')
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--batch-size', type=int, default=1020)
parser.add_argument('--iters', type=int, default=150)
parser.add_argument('--sigma', type=float, default=5.0)
parser.add_argument('--kernel-size', type=int, default=11)
parser.add_argument('--reps', type=int, default=2)
parser.add_argument('--lmbda', type=float, default=0.001)
parser.add_argument('--scale-factor', type=int, default=1)
parser.add_argument('--device',  type=str, default='cuda')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(args.size, transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(args.size),
    transforms.ToTensor()
])
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------ Dataset ------------------

class SonarDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx])).astype(np.float32)
        label = np.array(Image.open(self.label_paths[idx])).astype(np.float32)

        p98 = np.percentile(image, 98)
        image = np.clip(image / p98, 0, 1)

        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        label = torch.from_numpy(label).unsqueeze(0)  # [1, H, W]
        return image, label

# ------------------ Data Paths ------------------

train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")

train_images = sorted([os.path.join(train_dir, "images", f) for f in os.listdir(os.path.join(train_dir, "images"))])
train_labels = sorted([os.path.join(train_dir, "labels", f) for f in os.listdir(os.path.join(train_dir, "labels"))])

test_images = sorted([os.path.join(test_dir, "images", f) for f in os.listdir(os.path.join(test_dir, "images"))])
test_labels = sorted([os.path.join(test_dir, "labels", f) for f in os.listdir(os.path.join(test_dir, "labels"))])

train_loader = DataLoader(SonarDataset(train_images, train_labels), batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(SonarDataset(test_images, test_labels), batch_size=1, shuffle=False)

# ------------------ Modules ------------------

inpainter = Inpainter(args.sigma, args.kernel_size, args.reps, args.scale_factor).to(args.device)
boundary = Boundary().to(args.device)

# ------------------ Training Loop ------------------

start_time = time.time()
for batch_idx, (x, seg) in enumerate(train_loader):
    print("Training Batch {}/{}".format(batch_idx + 1, len(train_loader)))
    x, seg = x.to(args.device), seg.to(args.device)

    B, _, H, W = x.shape

    mask = torch.nn.Parameter(torch.zeros(B, 1, H, W).to(args.device))
    init_start1, init_end1 = H // 5, H - H // 5
    init_start2, init_end2 = W // 5, W - W // 5

    mask.data[:, :, init_start1:init_end1, init_start2:init_end2].fill_(1.0)

    for i in range(args.iters):
        foreground = x * mask
        background = x * (1 - mask)

        pred_foreground = inpainter(background, (1 - mask))
        pred_background = inpainter(foreground, mask)

        inp_error = neg_coeff_constraint(x, mask, pred_foreground, pred_background)
        mask_diversity = diversity(x, mask, foreground, background)

        total_loss = inp_error - args.lmbda * mask_diversity
        total_loss.sum().backward()

        with torch.no_grad():
            grad = mask.grad.data
            update_bool = boundary(mask) * (grad != 0)
            mask.data[update_bool] = (grad[update_bool] > 0).float()
            grad.zero_()
            mask.data = (F.avg_pool2d(mask, 3, 1, 1, divisor_override=1) >= 4).float()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.1f} seconds")

# ------------------ Testing Loop with Visualization ------------------

os.makedirs("iem_outputs", exist_ok=True)

start_time = time.time()
for batch_idx, (x, seg) in enumerate(test_loader):
    print("Testing Batch {}/{}".format(batch_idx + 1, len(test_loader)))
    x, seg = x.to(args.device), seg.to(args.device)

    _, _, H, W = x.shape

    mask = torch.nn.Parameter(torch.zeros(1, 1, H, W).to(args.device))
    init_start1, init_end1 = H // 5, H - H // 5
    init_start2, init_end2 = W // 5, W - W // 5
    mask.data[:, :, init_start1:init_end1, init_start2:init_end2].fill_(1.0)

    for i in range(args.iters):
        foreground = x * mask
        background = x * (1 - mask)

        pred_foreground = inpainter(background, (1 - mask))
        pred_background = inpainter(foreground, mask)

        inp_error = neg_coeff_constraint(x, mask, pred_foreground, pred_background)
        mask_diversity = diversity(x, mask, foreground, background)

        total_loss = inp_error - args.lmbda * mask_diversity
        total_loss.sum().backward()

        with torch.no_grad():
            grad = mask.grad.data
            update_bool = boundary(mask) * (grad != 0)
            mask.data[update_bool] = (grad[update_bool] > 0).float()
            grad.zero_()
            mask.data = (F.avg_pool2d(mask, 3, 1, 1, divisor_override=1) >= 4).float()

            acc, iou, miou, dice = compute_performance(mask, seg)
            print(f"\tIter {i:>3}: InpError {inp_error.mean():.3f} IoU {iou:.3f} DICE {dice:.3f}")

    # ------------------ Save Composite Image ------------------

    input_img = x[0, 0].cpu().numpy()
    true_mask = seg[0, 0].cpu().numpy()
    est_mask = mask[0, 0].cpu().numpy()
    est_thresh = (est_mask >= 0.5).astype(np.float32)

    p98 = np.percentile(input_img, 98)
    norm_input = np.clip(input_img / p98, 0, 1)
    input_colored = cm.get_cmap('BuPu_r')(norm_input)[:, :, :3]
    input_colored = (input_colored * 255).astype(np.uint8)

    def to_rgb(mask):
        return (np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8)

    composite = np.concatenate([
        input_colored,
        to_rgb(true_mask),
        to_rgb(est_mask),
        to_rgb(est_thresh)
    ], axis=1)

    save_path = os.path.join("iem_outputs", f"test_{batch_idx}.png")
    Image.fromarray(composite).save(save_path)

end_time = time.time()
print(f"Testing completed in {end_time - start_time:.1f} seconds")
