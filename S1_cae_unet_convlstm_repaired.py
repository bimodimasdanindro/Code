# train_semi_cae_unet_convlstm.py
# ------------------------------------------------------------
# Semi-supervised cloud motion forecasting:
# U-Net + (Bi)ConvLSTM with reconstruction + segmentation heads
# Time-series 5-fold CV with purge to avoid leakage.
# ------------------------------------------------------------
import os, io, json, math, gzip, pickle, argparse, shutil, time, random, platform
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ----------------------------
# Utils: I/O & reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image_grid(rgb_list, path, ncols=None):
    """
    rgb_list: list of np.uint8 HxWx3
    """
    if not rgb_list:
        return
    h, w, _ = rgb_list[0].shape
    n = len(rgb_list)
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    canvas = np.zeros((nrows*h, ncols*w, 3), np.uint8)
    for i, img in enumerate(rgb_list):
        r, c = i // ncols, i % ncols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w, :] = img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def tensor_to_uint8(img_t: torch.Tensor, mean, std):
    """
    Inverse of (x - mean) / std when x was originally in [0,1].
    img_t: [3,H,W] torch tensor (normalized)
    mean, std: iterable of 3 floats (the same used to normalize)
    """
    x = img_t.detach().cpu().float()           # [3,H,W]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
    x = x * std_t + mean_t                     # back to [0,1]
    x = x.clamp(0, 1) * 255.0
    return x.byte().permute(1,2,0).numpy()


def load_any_pickle(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    with open(p, "rb") as fh:
        head = fh.read(2)
    def _pl(fobj):
        try:
            return pickle.load(fobj)
        except Exception:
            fobj.seek(0)
            return pickle.load(fobj, encoding="latin1")
    if head == b"\x1f\x8b":  # gzip
        with gzip.open(p, "rb") as f:
            return _pl(f)
    else:
        with open(p, "rb") as f:
            return _pl(f)


# --------------------------------------
# Datasets: from asi_seq rows -> tensors
# --------------------------------------
def _imread_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _ensure_np_img(img_like, fallback_path):
    if isinstance(img_like, np.ndarray):
        return img_like
    if fallback_path is not None:
        return _imread_rgb(fallback_path)
    return None

def _resize_img(img, size_hw):
    H, W = size_hw
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

def _resize_mask(mask, size_hw):
    H, W = size_hw
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

def _img_to_01(img: np.ndarray) -> np.ndarray:
    """Return float32 in [0,1]. If uint8 or max>1.5, scale /255."""
    x = img.astype(np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if img.dtype == np.uint8 or mx > 1.5:
        x /= 255.0
    return np.clip(x, 0.0, 1.0)

def _apply_sky_mask_to_img(img01: np.ndarray, sky_mask) -> np.ndarray:
    """
    img01: float32 [H,W,3] in [0,1]
    sky_mask: [H,W] or [H,W,1], any dtype; >0 means sky. Auto-resized to match img.
    """
    if not isinstance(sky_mask, np.ndarray):
        return img01

    H, W = img01.shape[:2]
    m = sky_mask

    # squeeze channel if [H,W,1]
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]

    # resize if needed
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    # boolean mask
    m = (m > 0)

    # quick exits (just for speed)
    if m.all():
        return img01
    if not m.any():
        return np.zeros_like(img01, dtype=np.float32)

    # broadcast over channels
    return img01 * m[..., None].astype(np.float32)

def _to_tensor_img(img01):
    # img01 must be float32 in [0,1]
    return torch.from_numpy(img01.transpose(2,0,1))  # [3,H,W]

def _to_tensor_mask(mask):
    return torch.from_numpy(mask.astype(np.int64))  # [-1,0..C-1]

def stack_window(row,
                 out_hw=(512, 512),
                 apply_sky_to_inputs=True,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5),
                 fill_missing_with=0.0,
                 input_colorspace=None,
                 **_ignored_kwargs):
    """
    Build [T,3,H,W] image tensor, [T,H,W] labels, [T,H,W] sky, and ISO timestamps.
    Accepts optional `input_colorspace` ('RGB'|'HSV'|'BGR'|None). If provided, images
    are converted to RGB before resizing/normalization.
    """
    imgs     = row["images"]
    segs     = row.get("seg_masks", [None] * len(imgs))
    sky_list = row.get("sky_masks", [None] * len(imgs))
    fpaths   = row.get("filepaths", [None] * len(imgs))
    ts_raw   = row["timestamps"]

    Ht, Wt = out_hw
    X_list, Y_list, SKY_list = [], [], []
    has_any_label = False

    cs = (str(input_colorspace).upper() if input_colorspace is not None else None)

    for im_like, seg_like, sky_like, fp in zip(imgs, segs, sky_list, fpaths):
        # -------- IMAGE: load/convert to RGB --------
        im = _ensure_np_img(im_like, fp)
        if isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] == 3 and cs:
            if cs == "HSV":
                im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
            elif cs == "BGR":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if im is None:
            img01 = np.full((Ht, Wt, 3), float(fill_missing_with), np.float32)
            sky_resized = None
        else:
            im = _resize_img(im, (Ht, Wt))   # expects RGB uint8
            img01 = _img_to_01(im)           # → float32 [0,1]

            # -------- SKY mask (optional) --------
            if isinstance(sky_like, np.ndarray):
                sky_resized = _resize_mask(sky_like, (Ht, Wt))
                sky_resized = (sky_resized > 0).astype(np.uint8)
            else:
                sky_resized = None

            if apply_sky_to_inputs and (sky_resized is not None):
                img01 = _apply_sky_mask_to_img(img01, sky_resized)

        # -------- LABELS --------
        if isinstance(seg_like, np.ndarray):
            lab = _resize_mask(seg_like, (Ht, Wt)).astype(np.int16)
            has_any_label = True
        else:
            lab = np.full((Ht, Wt), -1, dtype=np.int16)

        X_list.append(_to_tensor_img(img01))         # [3,H,W] float in [0,1]
        Y_list.append(_to_tensor_mask(lab))          # [H,W] long (-1 ok)
        SKY_list.append(torch.from_numpy(sky_resized) if isinstance(sky_resized, np.ndarray) else None)

    # stack time
    X = torch.stack(X_list, 0)                        # [T,3,H,W]
    Y = torch.stack(Y_list, 0) if has_any_label else None

    # always return SKY (fill missing with ones = all-sky)
    if all(s is None for s in SKY_list):
        SKY = torch.ones((X.shape[0], Ht, Wt), dtype=torch.uint8)
    else:
        SKY = torch.stack(
            [s if s is not None else torch.ones((Ht, Wt), dtype=torch.uint8) for s in SKY_list],
            0
        )  # [T,H,W]

    # normalize with provided mean/std
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).clamp_min(1e-6)
    X = (X - mean_t) / std_t

    # timestamps as strings so DataLoader can collate
    ts = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in ts_raw]
    return X, Y, SKY, ts


def apply_temporal_mask(X, k_range=(1,4), strategy="zero", token=None):
    T = X.shape[0]
    k = np.random.randint(k_range[0], k_range[1]+1) if k_range[1] > 0 else 0
    idx = np.random.choice(T, size=k, replace=False) if k > 0 else np.array([],dtype=int)
    mask_time = torch.zeros(T, dtype=torch.bool)
    if k > 0: mask_time[idx] = True
    X_in = X.clone()
    if k > 0:
        if strategy == "zero" or token is None:
            X_in[mask_time] = 0.0
        else:
            X_in[mask_time] = token
    return X_in, mask_time


class ConvLSTMDataset(Dataset):
    def __init__(self, df, out_hw=(512,512), k_range=(1,4),
                 apply_sky_to_inputs=True, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                 mask_in_val=False, input_colorspace="auto"):
        self.df = df
        self.out_hw = out_hw
        self.k_range = k_range
        self.apply_sky = apply_sky_to_inputs
        self.mean, self.std = mean, std
        self.mask_in_val = mask_in_val   # if False, no temporal masking (val)
        self.input_cs = input_colorspace

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        X, Y, SKY, ts = stack_window(row, out_hw=self.out_hw,
                                     apply_sky_to_inputs=self.apply_sky,
                                     mean=self.mean, std=self.std,
                                     input_colorspace=self.input_cs)

        # make sure Y is a tensor (all -1 if unlabeled)
        if Y is None:
            Y = torch.full((X.shape[0], self.out_hw[0], self.out_hw[1]),
                           fill_value=-1, dtype=torch.long)

        # temporal masking
        if self.k_range != (0,0) and self.mask_in_val:
            X_in, mask_time = apply_temporal_mask(X, k_range=self.k_range, strategy="zero")
        else:
            X_in, mask_time = X.clone(), torch.zeros(X.shape[0], dtype=torch.bool)

        sample = {
            "x_in": X_in,           # [T,3,H,W]
            "y_img": X,             # [T,3,H,W]
            "y_seg": Y,             # [T,H,W]
            "mask_time": mask_time, # [T]
            "sky": SKY,             # [T,H,W] uint8
            "timestamps": ts        # list[str]
        }
        return sample


# -----------------------------------------
# Time-series CV with Purge (no information leakage)
# -----------------------------------------
def purged_time_split(seq_df, val_start_time, val_end_time, L=10, extra_embargo_min=0):
    assert isinstance(seq_df.index, pd.DatetimeIndex), "seq_df must be indexed by start-time"
    assert "end" in seq_df.columns, "seq_df must have an 'end' column"
    embargo = pd.Timedelta(minutes=max(L-1, 0) + int(extra_embargo_min))
    train_mask = seq_df["end"] <= (pd.Timestamp(val_start_time) - embargo)
    val_mask   = (seq_df.index >= (pd.Timestamp(val_start_time) + embargo)) & \
                 (seq_df.index <= (pd.Timestamp(val_end_time)   - embargo))
    train_seq = seq_df.loc[train_mask].copy()
    val_seq   = seq_df.loc[val_mask].copy()
    return train_seq, val_seq

def ts_cv_splits_5(seq_df, n_splits=5, L=10, extra_embargo_min=0):
    df = seq_df.sort_index()
    n  = len(df)
    if n < (n_splits + 1):
        raise ValueError(f"Not enough windows ({n}) for {n_splits} splits.")
    test_size = max(n // (n_splits + 1), 1)
    splits = []
    for k in range(1, n_splits + 1):
        val_start_idx = k * test_size
        val_end_idx   = min((k + 1) * test_size - 1, n - 1)
        val_start_time = df.index[val_start_idx]
        val_end_time   = df.index[val_end_idx]
        tr, va = purged_time_split(df, val_start_time, val_end_time, L=L, extra_embargo_min=extra_embargo_min)
        if len(tr) and len(va):
            splits.append((tr, va, dict(fold=k, val_start=str(val_start_time), val_end=str(val_end_time))))
    if not splits:
        raise RuntimeError("No valid CV splits produced (dataset too short vs embargo).")
    return splits


# ------------------
# Model: ConvLSTM U-Net
# ------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, bias=True):
        super().__init__()
        pad = k//2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, k, padding=pad, bias=bias)
        self.hid_ch = hid_ch

    def forward(self, x, hx):
        h, c = hx
        z = torch.cat([x, h], dim=1)
        gates = self.conv(z)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, b, h, w, device):
        return (torch.zeros(b, self.hid_ch, h, w, device=device),
                torch.zeros(b, self.hid_ch, h, w, device=device))

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, n_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.fwd = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch, k) for i in range(n_layers)])
        if bidirectional:
            self.bwd = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch, k) for i in range(n_layers)])

    def forward(self, x_seq):  # x_seq: [B,T,C,H,W]
        B,T,C,H,W = x_seq.shape
        device = x_seq.device

        # forward
        h_f = x_seq
        for cell in self.fwd:
            h, c = cell.init_state(B, H, W, device)
            outs = []
            for t in range(T):
                h, c = cell(h_f[:,t], (h,c))
                outs.append(h)
            h_f = torch.stack(outs, dim=1)  # [B,T,Hid,H,W]

        if not self.bidirectional:
            return h_f  # [B,T,Hid,H,W]

        # backward
        h_b = x_seq.flip(1)
        for cell in self.bwd:
            h, c = cell.init_state(B, H, W, device)
            outs = []
            for t in range(T):
                h, c = cell(h_b[:,t], (h,c))
                outs.append(h)
            h_b = torch.stack(outs, dim=1)
        hs_b = h_b.flip(1)

        # concat on channel dim (C = dim=2 for [B,T,C,H,W])
        return torch.cat([h_f, hs_b], dim=2)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet_ConvLSTM(nn.Module):
    def __init__(self, in_ch=3, base=32, classes=4, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

        # Encoder (framewise)
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)

        # Bottleneck temporal (ConvLSTM on deepest features)
        in_lstm = base*4
        hid = base*4
        self.temporal = ConvLSTM(in_ch=in_lstm, hid_ch=hid, k=3, n_layers=1, bidirectional=bidirectional)
        bott_out_ch = hid if not bidirectional else hid*2
        self.bott_reduce = nn.Conv2d(bott_out_ch, hid, 1)  # unify channels for decoder

        # Decoder (per time step)
        self.up2 = nn.ConvTranspose2d(hid, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base + base, base)

        # Heads
        self.recon_head = nn.Conv2d(base, 3, 1)
        self.seg_head   = nn.Conv2d(base, classes, 1)

    def forward(self, x):  # x: [B,T,3,H,W]
        B,T,C,H,W = x.shape

        # Encode per frame
        e1_list, e2_list, e3_list = [], [], []
        for t in range(T):
            f = x[:,t]
            e1 = self.enc1(f)         # [B,base,H,W]
            p1 = self.pool1(e1)       # [B,base,H/2,W/2]
            e2 = self.enc2(p1)        # [B,2b,H/2,W/2]
            p2 = self.pool2(e2)       # [B,2b,H/4,W/4]
            e3 = self.enc3(p2)        # [B,4b,H/4,W/4]
            e1_list.append(e1); e2_list.append(e2); e3_list.append(e3)

        E3 = torch.stack(e3_list, dim=1)  # [B,T,4b,H/4,W/4]

        # Temporal ConvLSTM on deepest feature
        Hs = self.temporal(E3)            # [B,T,4b( x2 if bi),H/4,W/4]

        # Reduce to decoder channels and decode per timestep
        Yt = []
        for t in range(T):
            h = Hs[:,t]
            h = self.bott_reduce(h)                 # [B,4b,H/4,W/4]
            u2 = self.up2(h)                        # [B,2b,H/2,W/2]
            d2 = self.dec2(torch.cat([u2, e2_list[t]], dim=1))  # [B,2b,H/2,W/2]
            u1 = self.up1(d2)                       # [B,b,H,W]
            d1 = self.dec1(torch.cat([u1, e1_list[t]], dim=1))  # [B,b,H,W]
            recon = self.recon_head(d1)             # [B,3,H,W]
            seg   = self.seg_head(d1)               # [B,C,H,W]
            Yt.append((recon, seg))

        recon_seq = torch.stack([y[0] for y in Yt], dim=1)  # [B,T,3,H,W]
        seg_seq   = torch.stack([y[1] for y in Yt], dim=1)  # [B,T,C,H,W]
        return recon_seq, seg_seq


# -----------------------
# Losses & simple metrics
# -----------------------
def mse_time_mask_only(y_pred, y_true, mask_time):
    """
    MSE over **masked timesteps only** (True=masked).
    If there are no masked frames in the batch, returns 0 (safe).
    """
    if mask_time.dim() == 1:
        mask_time = mask_time[None, :].expand(y_pred.shape[0], -1)  # [B,T]
    m = mask_time.bool().float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1,1]
    diff2 = (y_pred - y_true) ** 2
    num = (m * diff2).sum()
    den = (m.sum() * y_pred.shape[2] * y_pred.shape[3] * y_pred.shape[4]).clamp_min(1.0)
    return num / den

def mse_all_frames(y_pred, y_true):
    """Standard MSE over all frames."""
    return ((y_pred - y_true) ** 2).mean()

def cross_entropy_ignore(logits, y, ignore_index=-1):
    # logits: [B,T,C,H,W], y: [B,T,H,W]
    B,T,C,H,W = logits.shape
    return F.cross_entropy(logits.view(B*T, C, H, W),
                           y.view(B*T, H, W),
                           ignore_index=ignore_index,
                           reduction='mean')

@torch.no_grad()
def psnr_from_mse(mse_val: float):
    if mse_val <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse_val)

@torch.no_grad()
def pixel_acc_ignore(logits, y, ignore_index=-1):
    B,T,C,H,W = logits.shape
    pred = logits.argmax(dim=2)  # [B,T,H,W]
    valid = (y != ignore_index)
    correct = (pred == y) & valid
    tot = valid.sum().item()
    if tot == 0: return 0.0
    return correct.sum().item() / tot


# =========================
# Build loaders (Windows-safe)
# =========================
def build_loaders(train_df, val_df, out_hw=(256,256), bs=2, num_workers=0, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), **kwargs):
    if platform.system().lower().startswith("win"):
        num_workers = 0  # avoid spawn/pickle errors

    tr_ds = ConvLSTMDataset(train_df, out_hw=out_hw, k_range=(1,4),
                            apply_sky_to_inputs=True, mean=mean, std=std, mask_in_val=True)
    va_ds = ConvLSTMDataset(val_df,   out_hw=out_hw, k_range=(0,0),
                            apply_sky_to_inputs=True, mean=mean, std=std, mask_in_val=False)

    pin = torch.cuda.is_available()
    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                    num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    va = DataLoader(va_ds, batch_size=bs, shuffle=False,
                    num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    return tr, va


# -----------------------
# Training / evaluation
# -----------------------
def estimate_mean_std(df,
                      n=2048,
                      out_hw=(512, 512),
                      apply_sky=True,
                      sky_only=True):
    """
    Estimate dataset mean/std for input normalization.
    """
    if len(df) == 0:
        return np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)

    idxs = np.linspace(0, len(df) - 1, num=min(n, len(df)), dtype=int)

    sums  = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0

    Ht, Wt = int(out_hw[0]), int(out_hw[1])

    for i in idxs:
        row     = df.iloc[i]
        imgs    = row["images"]
        fpaths  = row.get("filepaths", [None] * len(imgs))
        skylist = row.get("sky_masks", [None] * len(imgs))

        for img_like, fp, sky in zip(imgs, fpaths, skylist):
            im = _ensure_np_img(img_like, fp)
            if im is None:
                continue

            # resize and to [0,1]
            im = _resize_img(im, (Ht, Wt))
            im = _img_to_01(im)  # float32 [0,1], HxWx3

            sky_mask_resized = None
            if apply_sky and isinstance(sky, np.ndarray):
                if sky.shape[:2] != (Ht, Wt):
                    sky_mask_resized = _resize_mask(sky, (Ht, Wt))
                else:
                    sky_mask_resized = sky

            if apply_sky and sky_mask_resized is not None:
                m = (sky_mask_resized > 0)
                if sky_only:
                    nn = int(m.sum())
                    if nn == 0:
                        continue
                    px = im[m]  # [nn, 3]
                    sums  += px.sum(0)
                    sums2 += (px ** 2).sum(0)
                    count += nn
                else:
                    im = im * m[..., None].astype(im.dtype)
                    px = im.reshape(-1, 3)
                    sums  += px.sum(0)
                    sums2 += (px ** 2).sum(0)
                    count += px.shape[0]
            else:
                px = im.reshape(-1, 3)
                sums  += px.sum(0)
                sums2 += (px ** 2).sum(0)
                count += px.shape[0]

    if count == 0:
        return np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)

    mean = (sums / count).astype(np.float32)
    var  = (sums2 / count) - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std  = np.maximum(std, 1e-3)  # avoid zeros
    return mean, std


def train_one_epoch(model, loaders, optimizer, device,
                    lambda_seg: float = 0.5,
                    scaler: torch.amp.GradScaler | None = None,
                    amp_enabled: bool | None = None):
    """
    Train for 1 epoch on loaders['train'].
    Reconstruction loss is computed **only on masked timesteps**.
    """
    model.train()
    tr_loader = loaders["train"]
    use_amp = ((amp_enabled if amp_enabled is not None else (scaler is not None))
               and (device.type == "cuda"))

    rec_meter = 0.0
    seg_meter = 0.0
    n = 0

    pbar = tqdm(tr_loader, total=len(tr_loader), desc="train", leave=False)
    for batch in pbar:
        x_in   = batch["x_in"].to(device, non_blocking=True)      # [B,T,3,H,W]
        y_img  = batch["y_img"].to(device, non_blocking=True)     # [B,T,3,H,W]
        y_seg  = batch["y_seg"].to(device, non_blocking=True)     # [B,T,H,W]
        mtime  = batch["mask_time"].to(device, non_blocking=True) # [B,T]

        optimizer.zero_grad(set_to_none=True)

        try:
            if use_amp and (scaler is not None):
                with torch.amp.autocast('cuda', enabled=True):
                    recon_pred, seg_logits = model(x_in)
                    l_rec = mse_time_mask_only(recon_pred, y_img, mtime)
                    if (y_seg != -1).any() and (seg_logits is not None):
                        l_seg = cross_entropy_ignore(seg_logits, y_seg, ignore_index=-1)
                    else:
                        l_seg = l_rec.new_zeros(())
                    loss  = l_rec + lambda_seg * l_seg
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_pred, seg_logits = model(x_in)
                l_rec = mse_time_mask_only(recon_pred, y_img, mtime)
                if (y_seg != -1).any() and (seg_logits is not None):
                    l_seg = cross_entropy_ignore(seg_logits, y_seg, ignore_index=-1)
                else:
                    l_seg = l_rec.new_zeros(())
                loss  = l_rec + lambda_seg * l_seg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            rec_meter += float(l_rec.detach().item())
            seg_meter += float(l_seg.detach().item())
            n += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print("[WARN] OOM on this batch — skipping.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        pbar.set_postfix(rec=rec_meter / max(1, n), seg=seg_meter / max(1, n))

    return rec_meter / max(1, n), seg_meter / max(1, n)


@torch.no_grad()
def evaluate(model, loaders, device, lambda_seg=0.5, amp_enabled=False):
    """
    Validation: when there are no masked frames (typical), compute
    recon loss over **all frames** to get meaningful PSNR (not 99).
    If a batch contains masked frames (rare in val), use masked-only.
    """
    model.eval()
    va_loader = loaders["val"]

    use_amp = bool(amp_enabled) and (device.type == "cuda")

    rec_losses, seg_losses, psnrs, accs = [], [], [], []
    for batch in va_loader:
        x_in  = batch["x_in"].to(device)
        y_img = batch["y_img"].to(device)
        y_seg = batch["y_seg"].to(device)
        mtime = batch["mask_time"].to(device)

        if use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                recon_pred, seg_logits = model(x_in)
        else:
            recon_pred, seg_logits = model(x_in)

        if batch["mask_time"].any():
            l_rec = mse_time_mask_only(recon_pred, y_img, mtime)
            psnr_val = psnr_from_mse(float(l_rec.item()))
        else:
            l_rec = mse_all_frames(recon_pred, y_img)
            psnr_val = psnr_from_mse(float(l_rec.item()))

        rec_losses.append(float(l_rec.item()))
        psnrs.append(psnr_val)

        if (y_seg != -1).any() and (seg_logits is not None):
            l_seg = cross_entropy_ignore(seg_logits, y_seg, ignore_index=-1)
            seg_losses.append(float(l_seg.item()))
            accs.append(pixel_acc_ignore(seg_logits, y_seg, ignore_index=-1))

    def _mean(xs, default=0.0): return float(np.mean(xs)) if xs else float(default)
    return dict(rec=_mean(rec_losses), seg=_mean(seg_losses),
                psnr=_mean(psnrs), acc=_mean(accs))


def dump_samples(model, batch, outdir, tag, mean, std, device, grid_cols=5):
    """
    Save 2x5 grids:
      - inputs_raw (the unmasked ground-truth frames)
      - inputs_masked (what the model receives)
      - recons: masked frames = model recon; unmasked = passthrough
      - seg (if logits exist)
    """
    model.eval()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    x_in  = batch["x_in"][:1].to(device)   # [1,T,3,H,W] normalized + masked in some timesteps
    y_img = batch["y_img"][:1].to(device)  # [1,T,3,H,W] normalized full target
    mtime = batch["mask_time"][:1].to(device)  # [1,T] bool

    with torch.no_grad():
        recon_pred, seg_logits = model(x_in)

    xin = x_in[0]              # [T,3,H,W]
    yi  = y_img[0]             # [T,3,H,W]
    rp  = recon_pred[0]        # [T,3,H,W]
    T   = rp.shape[0]

    # Build recon visualization: masked -> recon, unmasked -> passthrough GT
    mask5d = mtime.bool().view(1, -1, 1, 1, 1).expand_as(recon_pred[:1])  # [1,T,3,H,W]
    recon_vis = torch.where(mask5d, recon_pred[:1], y_img[:1])[0]  # [T,3,H,W]

    def _to_u8(t): return tensor_to_uint8(t, mean, std)

    save_image_grid([_to_u8(yi[t])  for t in range(T)], Path(outdir)/f"{tag}_inputs_raw.png",    ncols=grid_cols)
    save_image_grid([_to_u8(xin[t]) for t in range(T)], Path(outdir)/f"{tag}_inputs_masked.png", ncols=grid_cols)
    save_image_grid([_to_u8(recon_vis[t]) for t in range(T)], Path(outdir)/f"{tag}_recons.png",  ncols=grid_cols)

    # segmentation preview (only if logits exist)
    if seg_logits is not None:
        seg = seg_logits.argmax(2)[0].detach().cpu().numpy()  # [T,H,W]
        pal = np.array([[0,0,0],[70,130,180],[255,215,0],[255,69,0]], np.uint8)
        seg_rgb = [pal[seg[t]].reshape(seg[t].shape+(3,)) for t in range(T)]
        save_image_grid(seg_rgb, Path(outdir)/f"{tag}_seg.png", ncols=grid_cols)


# ---------- index coercion ----------
def _first_ts(x):
    if isinstance(x, (list, tuple)) and len(x) > 0: return x[0]
    try: return x[0]
    except Exception: return pd.NaT

def _last_ts(x):
    if isinstance(x, (list, tuple)) and len(x) > 0: return x[-1]
    try: return x[-1]
    except Exception: return pd.NaT

def coerce_asi_seq_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Build DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "start" in df.columns:
        idx = pd.to_datetime(df["start"], errors="coerce")
    elif "timestamps" in df.columns:
        starts = pd.to_datetime(df["timestamps"].apply(_first_ts), errors="coerce")
        df["start"] = starts
        idx = starts
    else:
        raise ValueError("No 'start' or 'timestamps' to form DatetimeIndex.")

    # Ensure 'end' exists
    if "end" not in df.columns:
        if "timestamps" in df.columns:
            df["end"] = pd.to_datetime(df["timestamps"].apply(_last_ts), errors="coerce")
        else:
            df["end"] = idx

    # Finalize
    df["_idx"] = idx
    df = df.dropna(subset=["_idx"]).set_index("_idx", drop=True)
    if "_idx" in df.columns: df = df.drop(columns=["_idx"])
    df.index.name = "start"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    print("[DATA] windows:", len(df))
    print("[DATA] time range:", df.index.min(), "→", df["end"].max())
    return df


def try_resume_fold(fold_dir: Path, resume_from: str,
                    model: nn.Module, optim: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler | None, device: torch.device):
    """
    Returns (start_epoch:int, best_val:float or None). Loads states if resume file exists.
    """
    if resume_from is None:
        return 1, None

    ckpt_path = fold_dir / f"{resume_from}.pt"
    if not ckpt_path.exists():
        print(f"[RESUME] {ckpt_path} not found; starting fresh.")
        return 1, None

    print(f"[RESUME] Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    if "optim" in ckpt and ckpt["optim"] is not None:
        try:
            optim.load_state_dict(ckpt["optim"])
        except Exception as e:
            print(f"[RESUME][WARN] Optimizer state not loaded ({e}); continuing with fresh optimizer.")

    if scaler is not None and ("scaler" in ckpt) and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[RESUME][WARN] GradScaler state not loaded ({e}); AMP scaler will continue fresh.")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    metrics = ckpt.get("metrics", {})
    best_val = None
    if metrics:
        best_val = float(metrics.get("rec", 0.0)) + float(metrics.get("seg", 0.0))*float(ckpt.get("args", {}).get("lambda_seg", 0.5))

    print(f"[RESUME] Continuing at epoch {start_epoch}")
    return start_epoch, best_val


# --------------
# Main training
# --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="Dataset/asi_seq_color.pkl.gz")
    ap.add_argument("--exp",  type=str, default="semi_cae_unet_convlstm")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--embargo-extra", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-h", type=int, default=512)
    ap.add_argument("--out-w", type=int, default=512)
    ap.add_argument("--lambda-seg", type=float, default=0.5)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume-from", type=str, choices=["last", "best", None],
                    default=None, help="Resume each fold from its 'last' or 'best' checkpoint if present.")
    args = ap.parse_args()

    # --- derived from CLI ---
    OUT_HW = (args.out_h, args.out_w)

    # --- seeds / device / AMP ---
    set_seed(args.seed)
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    DEVICE   = torch.device("cuda" if use_cuda else "cpu")
    scaler   = torch.amp.GradScaler('cuda') if use_cuda else None
    amp_enabled = use_cuda
    print(f"[DEVICE] {DEVICE} | cuda_available={use_cuda}")

    # --- run dir ---
    out_root = Path("runs")/args.exp
    if out_root.exists():
        print(f"[INFO] Writing into existing {out_root} (will append).")
    out_root.mkdir(parents=True, exist_ok=True)

    # --- load data ---
    print(f"[LOAD] {args.data}")
    asi_seq = load_any_pickle(args.data)
    asi_seq = coerce_asi_seq_index(asi_seq)

    print("[DATA] windows:", len(asi_seq))
    print("[DATA] time range:", asi_seq.index.min(), "→", asi_seq["end"].max())
    print("[DATA] columns:", list(asi_seq.columns))

    # --- splits ---
    splits = ts_cv_splits_5(asi_seq, n_splits=args.folds, L=args.L, extra_embargo_min=args.embargo_extra)

    # --- global mean/std from train of first fold at correct size ---
    mean, std = estimate_mean_std(splits[0][0], out_hw=OUT_HW, apply_sky=False)
    print("[NORM] mean:", np.round(mean,4), "std:", np.round(std,4))

    # Windows-safe: force single-process loading to avoid spawn/pickle errors
    effective_workers = 0 if platform.system().lower().startswith("win") else max(0, int(args.num_workers))

    # --- per-fold training ---
    for (train_seq, val_seq, info) in splits:
        fold_id = info["fold"]
        fold_dir = out_root/f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir/"fold_info.json","w") as f:
            json.dump({k: (str(v) if "time" in k else v) for k,v in info.items()}, f, indent=2)

        print(f"\n[Fold {fold_id}] val {info['val_start']} → {info['val_end']} | "
              f"train windows: {len(train_seq)} | val windows: {len(val_seq)}")

        # --- loaders ---
        tr_loader, va_loader = build_loaders(
            train_seq, val_seq,
            out_hw=OUT_HW, bs=args.bs,
            num_workers=effective_workers, mean=mean, std=std
        )
        loaders = {"train": tr_loader, "val": va_loader}
        
        # --- model / optim ---
        model = UNet_ConvLSTM(in_ch=3, base=32, classes=4,
                              bidirectional=args.bidirectional).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        start_epoch, best_val = try_resume_fold(fold_dir, args.resume_from, model, optim, scaler, DEVICE)
        if best_val is None:
            best_val = float("inf")

        # --- sample before training (guard for empty train fold) ---
        try:
            first_batch = next(iter(tr_loader))
        except StopIteration:
            print(f"[WARN] Fold {fold_id} has empty training loader. Skipping training loop for this fold.")
            try:
                vb = next(iter(va_loader))
                dump_samples(model, vb, fold_dir/"samples", tag="init", mean=mean, std=std, device=DEVICE, grid_cols=5)
            except StopIteration:
                pass
            continue

        dump_samples(model, first_batch, fold_dir/"samples", tag="init", mean=mean, std=std, device=DEVICE, grid_cols=5)

        # --- train loop ---
        log_path = fold_dir / "log.jsonl"

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            # ---- TRAIN ----
            rec_tr, seg_tr = train_one_epoch(
                model, loaders, optim, DEVICE,
                lambda_seg=args.lambda_seg,
                scaler=scaler,
                amp_enabled=amp_enabled
            )

            # ---- VALIDATE ----
            metrics = evaluate(
                model, loaders, DEVICE,
                lambda_seg=args.lambda_seg
            )

            dt = time.time() - t0
            print(f"[Fold {fold_id}][Epoch {epoch:03d}] "
                  f"train: rec={rec_tr:.4f} seg={seg_tr:.4f} | "
                  f"val: rec={metrics['rec']:.4f} seg={metrics['seg']:.4f} "
                  f"psnr={metrics['psnr']:.2f} acc={metrics['acc']:.3f} | {dt:.1f}s")

            # append log (jsonl)
            with open(log_path, "a") as f:
                rec = dict(epoch=epoch, train_rec=rec_tr, train_seg=seg_tr, **metrics)
                f.write(json.dumps(rec) + "\n")

            # always save "last"
            last_ckpt = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "metrics": metrics,
                "mean": mean, "std": std,
            }
            if scaler is not None:
                last_ckpt["scaler"] = scaler.state_dict()
            torch.save(last_ckpt, fold_dir / "last.pt")

            # best checkpoint (lower is better: rec + λ * seg)
            val_score = metrics["rec"] + args.lambda_seg * metrics["seg"]
            if epoch == 1 or val_score < best_val - 1e-6:
                best_val = val_score
                torch.save(last_ckpt, fold_dir / "best.pt")
                torch.save(last_ckpt, fold_dir / f"best_epoch{epoch:03d}.pt")
                dump_samples(model, first_batch, fold_dir / "samples",
                             tag=f"best_e{epoch:03d}", mean=mean, std=std, device=DEVICE, grid_cols=5)

        # final sample
        dump_samples(model, first_batch, fold_dir/"samples", tag="final", mean=mean, std=std, device=DEVICE)

    print("\n[Done] All folds complete. See:", out_root)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
