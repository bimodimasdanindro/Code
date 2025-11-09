# S2_cae_unet_convlstm_v3.py
# ------------------------------------------------------------
# Cloud motion forecasting (U-Net + (Bi)ConvLSTM), no segmentation head.
# CUDA-stable hybrid loss (L1 + (1-SSIM) + grad + temporal smooth) for horizons.
# Windows-safe DataLoader: force num_workers=0 on Windows + auto-fallback.
# Resume: --resume-from per fold, atau pakai --resume-path.
# ------------------------------------------------------------
import os, io, json, math, gzip, pickle, argparse, shutil, time, random, platform, tempfile, sys, traceback
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

# =============== Utils ===============
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_image_grid(rgb_list, path, ncols=None):
    if not rgb_list: return
    h, w, _ = rgb_list[0].shape
    n = len(rgb_list); ncols = ncols or int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    canvas = np.zeros((nrows*h, ncols*w, 3), np.uint8)
    for i, img in enumerate(rgb_list):
        r, c = i // ncols, i % ncols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w, :] = img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)

def tensor_to_uint8(img_t: torch.Tensor, mean, std):
    x = img_t.detach().cpu().float()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
    x = x * std_t + mean_t
    x = x.clamp(0, 1) * 255.0
    return x.byte().permute(1,2,0).numpy()

def load_any_pickle(path):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Not found: {p}")
    with open(p, "rb") as fh: head = fh.read(2)
    def _pl(fobj):
        try: return pickle.load(fobj)
        except Exception:
            fobj.seek(0); return pickle.load(fobj, encoding="latin1")
    if head == b"\x1f\x8b":
        with gzip.open(p, "rb") as f: return _pl(f)
    else:
        with open(p, "rb") as f: return _pl(f)

def infer_minutes_per_step(ts_list):
    try:
        vals = []
        for i in range(1, len(ts_list)):
            t0, t1 = pd.to_datetime(ts_list[i-1], errors="coerce"), pd.to_datetime(ts_list[i], errors="coerce")
            if pd.isna(t0) or pd.isna(t1): continue
            dmin = (t1 - t0).total_seconds() / 60.0
            if dmin > 0: vals.append(dmin)
        return float(np.median(vals)) if vals else 1.0
    except Exception:
        return 1.0

def atomic_save(state, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt_", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, str(path))
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except OSError:
            pass

# =============== Dataset ===============
def _imread_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _ensure_np_img(img_like, fallback_path):
    if isinstance(img_like, np.ndarray): return img_like
    if fallback_path is not None: return _imread_rgb(fallback_path)
    return None

def _resize_img(img, size_hw):
    H, W = size_hw
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

def _resize_mask(mask, size_hw):
    H, W = size_hw
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

def _img_to_01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if img.dtype == np.uint8 or mx > 1.5: x /= 255.0
    return np.clip(x, 0.0, 1.0)

def _apply_sky_mask_to_img(img01: np.ndarray, sky_mask) -> np.ndarray:
    if not isinstance(sky_mask, np.ndarray): return img01
    H, W = img01.shape[:2]
    m = sky_mask
    if m.ndim == 3 and m.shape[2] == 1: m = m[..., 0]
    if m.shape[:2] != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = (m > 0)
    if m.all(): return img01
    if not m.any(): return np.zeros_like(img01, dtype=np.float32)
    return img01 * m[..., None].astype(np.float32)

def _to_tensor_img(img01): return torch.from_numpy(img01.transpose(2,0,1))  # [3,H,W]

def stack_window(row,
                 out_hw=(512, 512),
                 apply_sky_to_inputs=True,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5),
                 fill_missing_with=0.0,
                 input_colorspace=None,
                 **_):
    imgs     = row["images"]
    sky_list = row.get("sky_masks", [None] * len(imgs))
    fpaths   = row.get("filepaths", [None] * len(imgs))
    ts_raw   = row["timestamps"]

    Ht, Wt = out_hw
    X_list, SKY_list = [], []

    cs = (str(input_colorspace).upper() if input_colorspace is not None else None)

    for im_like, sky_like, fp in zip(imgs, sky_list, fpaths):
        im = _ensure_np_img(im_like, fp)
        if isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] == 3 and cs:
            if cs == "HSV": im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
            elif cs == "BGR": im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im is None:
            img01 = np.full((Ht, Wt, 3), float(fill_missing_with), np.float32); sky_resized = None
        else:
            im = _resize_img(im, (Ht, Wt)); img01 = _img_to_01(im)
            if isinstance(sky_like, np.ndarray):
                sky_resized = _resize_mask(sky_like, (Ht, Wt)); sky_resized = (sky_resized > 0).astype(np.uint8)
            else:
                sky_resized = None
            if apply_sky_to_inputs and (sky_resized is not None):
                img01 = _apply_sky_mask_to_img(img01, sky_resized)
        X_list.append(_to_tensor_img(img01))
        SKY_list.append(torch.from_numpy(sky_resized) if isinstance(sky_resized, np.ndarray) else None)

    X = torch.stack(X_list, 0)  # [T,3,H,W]
    if all(s is None for s in SKY_list):
        SKY = torch.ones((X.shape[0], Ht, Wt), dtype=torch.uint8)
    else:
        SKY = torch.stack([s if s is not None else torch.ones((Ht, Wt), dtype=torch.uint8) for s in SKY_list], 0)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).clamp_min(1e-6)
    X = (X - mean_t) / std_t
    ts = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in ts_raw]
    return X, SKY, ts

class ConvLSTMDataset(Dataset):
    def __init__(self, df, out_hw=(512,512),
                 apply_sky_to_inputs=True, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                 input_colorspace="auto"):
        self.df = df; self.out_hw = out_hw
        self.apply_sky = apply_sky_to_inputs
        self.mean, self.std = mean, std
        self.input_cs = input_colorspace
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        X, SKY, ts = stack_window(row, out_hw=self.out_hw,
                                  apply_sky_to_inputs=self.apply_sky,
                                  mean=self.mean, std=self.std,
                                  input_colorspace=self.input_cs)
        return {"x_in": X.clone(), "y_img": X, "sky": SKY, "timestamps": ts}

# =============== CV split & index ===============
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
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "start" in df.columns:
        idx = pd.to_datetime(df["start"], errors="coerce")
    elif "timestamps" in df.columns:
        starts = pd.to_datetime(df["timestamps"].apply(_first_ts), errors="coerce")
        df["start"] = starts; idx = starts
    else:
        raise ValueError("No 'start' or 'timestamps' to form DatetimeIndex.")
    if "end" not in df.columns:
        if "timestamps" in df.columns:
            df["end"] = pd.to_datetime(df["timestamps"].apply(_last_ts), errors="coerce")
        else:
            df["end"] = idx
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

def purged_time_split(seq_df, val_start_time, val_end_time, L=10, extra_embargo_min=0):
    assert isinstance(seq_df.index, pd.DatetimeIndex), "seq_df must be indexed by start-time"
    assert "end" in seq_df.columns, "seq_df must have an 'end' column"
    embargo = pd.Timedelta(minutes=max(L-1, 0) + int(extra_embargo_min))
    train_mask = seq_df["end"] <= (pd.Timestamp(val_start_time) - embargo)
    val_mask   = (seq_df.index >= (pd.Timestamp(val_start_time) + embargo)) & \
                 (seq_df.index <= (pd.Timestamp(val_end_time)   - embargo))
    return seq_df.loc[train_mask].copy(), seq_df.loc[val_mask].copy()

def ts_cv_splits_5(seq_df, n_splits=5, L=10, extra_embargo_min=0):
    df = seq_df.sort_index(); n = len(df)
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

# =============== Model ===============
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
        c = f * c + i * g; h = o * torch.tanh(c)
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
    def forward(self, x_seq):  # [B,T,C,H,W]
        B,T,C,H,W = x_seq.shape
        device = x_seq.device
        h_f = x_seq
        for cell in self.fwd:
            h, c = cell.init_state(B, H, W, device)
            outs = []
            for t in range(T):
                h, c = cell(h_f[:,t], (h,c))
                outs.append(h)
            h_f = torch.stack(outs, dim=1)
        if not self.bidirectional:
            return h_f
        h_b = x_seq.flip(1)
        for cell in self.bwd:
            h, c = cell.init_state(B, H, W, device)
            outs = []
            for t in range(T):
                h, c = cell(h_b[:,t], (h,c))
                outs.append(h)
            h_b = torch.stack(outs, dim=1)
        hs_b = h_b.flip(1)
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
    def __init__(self, in_ch=3, base=32, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.enc1 = DoubleConv(in_ch, base); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        in_lstm = base*4; hid = base*4
        self.temporal = ConvLSTM(in_ch=in_lstm, hid_ch=hid, k=3, n_layers=1, bidirectional=bidirectional)
        bott_out_ch = hid if not bidirectional else hid*2
        self.bott_reduce = nn.Conv2d(bott_out_ch, hid, 1)
        self.up2 = nn.ConvTranspose2d(hid, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base + base, base)
        self.recon_head = nn.Conv2d(base, 3, 1)
    def forward(self, x):  # x: [B,T,3,H,W]
        B,T,C,H,W = x.shape
        e1_list, e2_list, e3_list = [], [], []
        for t in range(T):
            f = x[:,t]
            e1 = self.enc1(f); p1 = self.pool1(e1)
            e2 = self.enc2(p1); p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            e1_list.append(e1); e2_list.append(e2); e3_list.append(e3)
        E3 = torch.stack(e3_list, dim=1)  # [B,T,4b,H/4,W/4]
        Hs = self.temporal(E3)            # [B,T,4b(x2 if bi),H/4,W/4]
        outs = []
        for t in range(T):
            h = self.bott_reduce(Hs[:,t])
            u2 = self.up2(h)
            d2 = self.dec2(torch.cat([u2, e2_list[t]], dim=1))
            u1 = self.up1(d2)
            d1 = self.dec1(torch.cat([u1, e1_list[t]], dim=1))
            outs.append(self.recon_head(d1))
        return torch.stack(outs, dim=1)  # [B,T,3,H,W]

# =============== SSIM util ===============
def _gauss_kernel_1d(size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    ax = torch.arange(size, dtype=dtype, device=device) - size//2
    k = torch.exp(-(ax**2)/(2*sigma**2)); k /= k.sum()
    return k

def _depthwise_conv2d_same(x: torch.Tensor, k2_11x11: torch.Tensor):
    B, C, H, W = x.shape
    k = k2_11x11.to(device=x.device, dtype=x.dtype)
    weight = k.expand(C, 1, k.size(-2), k.size(-1)).contiguous()
    try:
        return F.conv2d(x, weight, bias=None, stride=1, padding=k.size(-1)//2, groups=C)
    except Exception:
        outs = []
        pad = k.size(-1)//2
        for c in range(C):
            xc = x[:, c:c+1]
            yc = F.conv2d(xc, k, padding=pad)
            outs.append(yc)
        return torch.cat(outs, dim=1)

class _SSIMComputer:
    def __init__(self): self.cache = {}
    def get_kernel(self, device, dtype):
        key = (str(device), str(dtype))
        k = self.cache.get(key)
        if k is None:
            k1 = _gauss_kernel_1d(11, 1.5, device=device, dtype=dtype)
            k = torch.outer(k1, k1).reshape(1,1,11,11)
            self.cache[key] = k
        return k

_SSIM = _SSIMComputer()

def ssim_01(x, y, C1=0.01**2, C2=0.03**2):
    x = x.float().contiguous(); y = y.float().contiguous()
    k2 = _SSIM.get_kernel(x.device, x.dtype)
    mu_x = _depthwise_conv2d_same(x, k2);  mu_y = _depthwise_conv2d_same(y, k2)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sigma_x2 = _depthwise_conv2d_same(x*x, k2) - mu_x2
    sigma_y2 = _depthwise_conv2d_same(y*y, k2) - mu_y2
    sigma_xy = _depthwise_conv2d_same(x*y, k2) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-8)
    return ssim_map.mean()

# =============== Loss & metrics ===============
def denorm_to_01(x, mean, std):
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, -1, 1, 1).clamp_min(1e-6)
    x = x * s + m
    return x.clamp(0,1)

@torch.no_grad()
def psnr_forecast_shift_denorm(pred, target, horizon_steps, mean, std):
    vals = []
    B, T, C, H, W = pred.shape
    for h in horizon_steps:
        if h <= 0 or h > T:  # fixed
            continue
        Teff = T - h
        if Teff <= 0: 
            continue
        p = denorm_to_01(pred[:, :Teff], mean, std)
        g = denorm_to_01(target[:, h:h+Teff], mean, std)
        mse = F.mse_loss(p, g, reduction="mean").item()
        vals.append(99.0 if mse < 1e-10 else min(99.0, 10.0 * math.log10(1.0 / (mse + 1e-10))))
    return float(np.mean(vals)) if vals else 0.0

def gradient_loss(x, y):
    dx_x = x[..., 1:, :] - x[..., :-1, :]
    dy_x = x[..., :, 1:] - x[..., :, :-1]
    dx_y = y[..., 1:, :] - y[..., :-1, :]
    dy_y = y[..., :, 1:] - y[..., :, :-1]
    return (dx_x - dx_y).abs().mean() + (dy_x - dy_y).abs().mean()

def forecast_hybrid_loss(pred, target, horizon_steps, alpha=0.15, beta=0.25, gamma=0.05, eta=0.9, mean=None, std=None):
    pred = pred.float(); target = target.float()
    B,T,C,H,W = pred.shape
    if mean is not None and std is not None:
        p_den = denorm_to_01(pred.flatten(0,1), mean, std).view(B,T,C,H,W)
        g_den = denorm_to_01(target.flatten(0,1), mean, std).view(B,T,C,H,W)
    else:
        p_den = pred.clamp(0,1); g_den = target.clamp(0,1)
    tot = pred.new_zeros(())
    for h in horizon_steps:
        if h <= 0 or h > T: continue
        Teff = T - h
        if Teff <= 0: continue
        p01 = p_den[:, :Teff].flatten(0,1)
        g01 = g_den[:, h:h+Teff].flatten(0,1)
        l1 = (p01 - g01).abs().mean()
        s1 = 1.0 - ssim_01(p01, g01)
        gl = gradient_loss(p01, g01)
        w = (eta ** (h-1))
        tot = tot + w * (l1 + alpha * s1 + beta * gl)
    if T > 1:
        sm = (p_den[:,1:] - p_den[:,:-1]).abs().mean()
        tot = tot + gamma * sm
    return tot

# =============== Loaders (Windows-safe) ===============
def build_loaders(train_df, val_df, out_hw=(256,256), bs=2, num_workers=4, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                  device=None, prefetch_factor=4, persistent_workers=True):
    """Harden for Windows: force workers=0 on Windows to avoid spawn/pickle crashes."""
    is_windows = platform.system().lower().startswith("win")
    is_cuda = (isinstance(device, torch.device) and device.type=="cuda")

    if is_windows:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = 2
    else:
        # Linux/mac: optimize for CUDA
        if is_cuda:
            num_workers = max(4, int(num_workers))
            persistent_workers = True
            prefetch_factor = max(2, int(prefetch_factor))
        else:
            num_workers = 0
            persistent_workers = False
            prefetch_factor = 2

    pin = bool(is_cuda)

    tr_ds = ConvLSTMDataset(train_df, out_hw=out_hw, apply_sky_to_inputs=True, mean=mean, std=std)
    va_ds = ConvLSTMDataset(val_df,   out_hw=out_hw, apply_sky_to_inputs=True, mean=mean, std=std)

    tr_kwargs = dict(batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin, persistent_workers=persistent_workers)
    va_kwargs = dict(batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent_workers)
    if num_workers > 0:
        tr_kwargs["prefetch_factor"] = prefetch_factor
        va_kwargs["prefetch_factor"] = prefetch_factor

    tr = DataLoader(tr_ds, **tr_kwargs)
    va = DataLoader(va_ds, **va_kwargs)
    return tr, va

def build_loaders_with_fallback(train_df, val_df, *, out_hw, bs, num_workers, mean, std, device, prefetch_factor):
    """Try to build with requested num_workers; if the first batch raises a spawn/pickle error, rebuild with workers=0."""
    def _make(nw):
        return build_loaders(train_df, val_df, out_hw=out_hw, bs=bs,
                             num_workers=nw, mean=mean, std=std,
                             device=device, prefetch_factor=prefetch_factor,
                             persistent_workers=(nw>0 and not platform.system().lower().startswith("win")))
    tr, va = _make(num_workers)
    # Probe one batch safely
    try:
        _ = next(iter(tr))
        _ = next(iter(va))
        return tr, va
    except Exception as e:
        msg = str(e).lower()
        bad = ("pickle" in msg) or ("unpicklingerror" in msg) or ("invalid argument" in msg) or ("spawn" in msg)
        if platform.system().lower().startswith("win") or bad:
            print("[LOADERS][FALLBACK] Rebuilding DataLoaders with num_workers=0 (Windows-safe).")
            tr, va = _make(0)
            return tr, va
        raise

# =============== Device & AMP runtime ===============
def make_amp_runtime(args):
    req = (getattr(args, "device", "auto") or "auto").lower()

    if req in ("directml", "privateuseone"):
        import torch_directml as dml
        DEVICE = dml.device()
        amp_enabled = False; scaler = None
        def amp_ctx(): return nullcontext()
        print(f"[DEVICE] {DEVICE} | type=DirectML | cuda_available={torch.cuda.is_available()} | amp_enabled={amp_enabled}")
        return DEVICE, amp_enabled, scaler, amp_ctx

    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        else:
            try:
                import torch_directml as dml
                DEVICE = dml.device()
                print(f"[DEVICE] {DEVICE} | type=DirectML(auto) | cuda_available={torch.cuda.is_available()} | amp_enabled=False")
                return DEVICE, False, None, nullcontext
            except Exception:
                req = "cpu"

    if req == "cuda":
        DEVICE = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        amp_enabled = (getattr(args, "amp", "auto") != "off")
        scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
        def amp_ctx(): return torch.amp.autocast(device_type='cuda', enabled=amp_enabled)
        print(f"[DEVICE] {DEVICE} | type=cuda | cuda_available=True | amp_enabled={amp_enabled}")
        return DEVICE, amp_enabled, scaler, amp_ctx

    if req == "mps":
        DEVICE = torch.device("mps")
        print(f"[DEVICE] {DEVICE} | type=mps | cuda_available={torch.cuda.is_available()} | amp_enabled=False")
        return DEVICE, False, None, nullcontext

    DEVICE = torch.device("cpu")
    print(f"[DEVICE] {DEVICE} | type=cpu | cuda_available={torch.cuda.is_available()} | amp_enabled=False")
    return DEVICE, False, None, nullcontext

# =============== Train / Eval ===============
def train_one_epoch(model, loaders, optimizer, device,
                    scaler: torch.amp.GradScaler | None = None,
                    amp_enabled: bool | None = None,
                    horizons_steps: list[int] | None = None,
                    amp_ctx=nullcontext,
                    alpha=0.15, beta=0.25, gamma=0.05, eta=0.9,
                    mean=None, std=None,
                    max_train_batches: int = 0,
                    accum_steps: int = 1):
    model.train()
    tr_loader = loaders["train"]
    is_cuda = isinstance(device, torch.device) and (device.type == "cuda")
    use_amp = bool(amp_enabled and is_cuda)

    rec_meter, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    fwd_autocast = (torch.amp.autocast(device_type='cuda', enabled=True)
                    if (is_cuda and amp_enabled) else nullcontext())

    pbar = tqdm(tr_loader, total=len(tr_loader), desc="train", leave=False)
    for ib, batch in enumerate(pbar, 1):
        x_in   = batch["x_in"].to(device, non_blocking=is_cuda)
        y_img  = batch["y_img"].to(device, non_blocking=is_cuda)

        try:
            with fwd_autocast:
                recon_pred = model(x_in)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss = forecast_hybrid_loss(recon_pred, y_img, horizons_steps,
                                            alpha=alpha, beta=beta, gamma=gamma, eta=eta,
                                            mean=mean, std=std)
            loss_to_backprop = loss / max(1, accum_steps)
            if use_amp and (scaler is not None):
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (ib % max(1, accum_steps)) == 0:
                if use_amp and (scaler is not None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            rec_meter += float(loss.detach().item()); n += 1
            if is_cuda:
                pbar.set_postfix(rec=f"{rec_meter/n:.4f}", vram=f"{torch.cuda.memory_allocated()/1e9:.2f}GB")
            else:
                pbar.set_postfix(rec=f"{rec_meter/n:.4f}")

        except RuntimeError as e:
            emsg = str(e).lower()
            if (("out of memory" in emsg) or ("cudnn" in emsg)) and is_cuda:
                print(f"\n[WARN] CUDA error on batch {ib} — skipping")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            raise

        if max_train_batches and ib >= max_train_batches:
            break

    if n > 0 and (n % max(1, accum_steps)) != 0:
        if use_amp and (scaler is not None):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return rec_meter/max(1,n)

@torch.no_grad()
def evaluate(model, loaders, device,
             horizons_steps: list[int] | None = None,
             mean=None, std=None,
             alpha=0.15, beta=0.25, gamma=0.05, eta=0.9,
             max_val_batches: int = 0,
             amp_enabled: bool = False):
    assert horizons_steps and len(horizons_steps) > 0, "horizons_steps must be non-empty."
    model.eval()
    va_loader = loaders["val"]
    rec_losses, psnrs = [], []

    is_cuda = isinstance(device, torch.device) and (device.type == "cuda")
    fwd_autocast = (torch.amp.autocast(device_type='cuda', enabled=True)
                    if (is_cuda and amp_enabled) else nullcontext())

    for ib, batch in enumerate(va_loader, 1):
        x_in  = batch["x_in"].to(device, non_blocking=is_cuda)
        y_img = batch["y_img"].to(device, non_blocking=is_cuda)

        with fwd_autocast:
            recon_pred = model(x_in)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            l_rec = forecast_hybrid_loss(recon_pred, y_img, horizons_steps,
                                         alpha=alpha, beta=beta, gamma=gamma, eta=eta,
                                         mean=mean, std=std)
        rec_losses.append(float(l_rec.item()))
        psnrs.append(psnr_forecast_shift_denorm(recon_pred, y_img, horizons_steps, mean, std))
        if max_val_batches and ib >= max_val_batches:
            break
    return dict(rec=float(np.mean(rec_losses)) if rec_losses else 0.0,
                psnr=float(np.mean(psnrs)) if psnrs else 0.0)

def dump_samples(model, batch, outdir, tag, mean, std, device, grid_cols=5,
                 horizons_steps: list[int] | None = None):
    model.eval(); Path(outdir).mkdir(parents=True, exist_ok=True)
    x_in  = batch["x_in"][:1].to(device)
    y_img = batch["y_img"][:1].to(device)
    with torch.no_grad():
        recon_pred = model(x_in)
    xin = x_in[0]; yi = y_img[0]; rp = recon_pred[0]; T = rp.shape[0]
    def _to_u8(t): return tensor_to_uint8(t, mean, std)
    save_image_grid([_to_u8(xin[t]) for t in range(T)], Path(outdir)/f"{tag}_inputs.png", ncols=grid_cols)
    feasible = [h for h in (horizons_steps or [1]) if h < T]
    h0 = min(feasible) if feasible else 1
    Teff = max(0, T - h0)
    pred_list = [_to_u8(rp[t]) for t in range(Teff)]
    gt_list   = [_to_u8(yi[t+h0]) for t in range(Teff)]
    save_image_grid(pred_list, Path(outdir)/f"{tag}_pred_h{h0}steps.png", ncols=grid_cols)
    save_image_grid(gt_list,   Path(outdir)/f"{tag}_gt_h{h0}steps.png",   ncols=grid_cols)

# =============== Stats ===============
def estimate_mean_std(df, n=2048, out_hw=(512, 512), apply_sky=True, sky_only=True):
    if len(df) == 0:
        return np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)
    idxs = np.linspace(0, len(df) - 1, num=min(n, len(df)), dtype=int)
    sums  = np.zeros(3, dtype=np.float64); sums2 = np.zeros(3, dtype=np.float64); count = 0
    Ht, Wt = int(out_hw[0]), int(out_hw[1])
    for i in idxs:
        row = df.iloc[i]; imgs = row["images"]
        fpaths = row.get("filepaths", [None] * len(imgs))
        skylist = row.get("sky_masks", [None] * len(imgs))
        for img_like, fp, sky in zip(imgs, fpaths, skylist):
            im = _ensure_np_img(img_like, fp)
            if im is None: continue
            im = _resize_img(im, (Ht, Wt)); im = _img_to_01(im)
            sky_mask_resized = None
            if apply_sky and isinstance(sky, np.ndarray):
                sky_mask_resized = _resize_mask(sky, (Ht, Wt)) if sky.shape[:2] != (Ht, Wt) else sky
            if apply_sky and sky_mask_resized is not None:
                m = (sky_mask_resized > 0); nn = int(m.sum())
                if nn == 0: continue
                px = im[m]; sums  += px.sum(0); sums2 += (px ** 2).sum(0); count += nn
            else:
                px = im.reshape(-1, 3); sums  += px.sum(0); sums2 += (px ** 2).sum(0); count += px.shape[0]
    if count == 0:
        return np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)
    mean = (sums / count).astype(np.float32)
    var  = (sums2 / count) - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std  = np.maximum(std, 1e-3)
    return mean, std

# =============== Resume ===============
def try_resume_fold(fold_dir: Path, resume_from: str,
                    model: nn.Module, optim: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler | None, device):
    if resume_from is None:
        return 1, None
    ckpt_path = fold_dir / f"{resume_from}.pt"
    if not ckpt_path.exists():
        print(f"[RESUME] {ckpt_path} not found; starting fresh.")
        return 1, None
    print(f"[RESUME] Loading {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[RESUME][WARN] Failed to load {ckpt_path} ({e}); starting fresh.")
        return 1, None
    model.load_state_dict(ckpt["model"], strict=True)
    if "optim" in ckpt and ckpt["optim"] is not None:
        try: optim.load_state_dict(ckpt["optim"])
        except Exception as e: print(f"[RESUME][WARN] Optimizer not loaded ({e}); continue fresh.")
    if scaler is not None and ("scaler" in ckpt) and ckpt["scaler"] is not None:
        try: scaler.load_state_dict(ckpt["scaler"])
        except Exception as e: print(f"[RESUME][WARN] GradScaler not loaded ({e}); AMP scaler fresh.")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    metrics = ckpt.get("metrics", {})
    best_val = float(metrics.get("rec", float("inf"))) if metrics else float("inf")
    print(f"[RESUME] Continuing at epoch {start_epoch}")
    return start_epoch, best_val

def try_resume_from_path(resume_path: Path,
                         model: nn.Module, optim: torch.optim.Optimizer,
                         scaler: torch.amp.GradScaler | None, device):
    if resume_path is None: return 1, None, None
    rp = Path(resume_path)
    if not rp.exists():
        print(f"[RESUME] {rp} not found; starting fresh.")
        return 1, None, None
    print(f"[RESUME] Loading explicit checkpoint: {rp}")
    ckpt = torch.load(rp, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optim" in ckpt and ckpt["optim"] is not None:
        try: optim.load_state_dict(ckpt["optim"])
        except Exception as e: print(f"[RESUME][WARN] Optimizer not loaded ({e}); continue fresh.")
    if scaler is not None and ("scaler" in ckpt) and ckpt["scaler"] is not None:
        try: scaler.load_state_dict(ckpt["scaler"])
        except Exception as e: print(f"[RESUME][WARN] GradScaler not loaded ({e}); AMP scaler fresh.")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    metrics = ckpt.get("metrics", {})
    best_val = float(metrics.get("rec", float("inf"))) if metrics else float("inf")
    print(f"[RESUME] Continuing at epoch {start_epoch} (from path)")
    return start_epoch, best_val, ckpt

# =============== Main ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="Dataset/asi_seq_color.pkl.gz")
    ap.add_argument("--exp",  type=str, default="forecast_unet_convlstm")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--embargo-extra", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-h", type=int, default=512)
    ap.add_argument("--out-w", type=int, default=512)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps|directml")
    ap.add_argument("--resume-from", type=str, choices=["last", "best", None], default=None)
    ap.add_argument("--resume-path", type=str, default=None)
    ap.add_argument("--horizons", type=str, default="1,3,5,10")
    ap.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--alpha-ssim", type=float, default=0.15)
    ap.add_argument("--beta-edge", type=float, default=0.25)
    ap.add_argument("--gamma-smooth", type=float, default=0.05)
    ap.add_argument("--eta-horizon", type=float, default=0.9)
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--max-val-batches",   type=int, default=0)
    args = ap.parse_args()

    try:
        horizons_min = [int(x) for x in str(args.horizons).split(",") if str(x).strip() != ""]
        horizons_min = sorted(set(horizons_min))
    except Exception:
        horizons_min = [1,3,5,10]
    OUT_HW = (args.out_h, args.out_w)

    set_seed(args.seed)
    DEVICE, amp_enabled, scaler, amp_ctx = make_amp_runtime(args)

    out_root = Path("runs-s2") / args.exp
    if out_root.exists(): print(f"[INFO] Writing into existing {out_root} (will append).")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {args.data}")
    asi_seq = load_any_pickle(args.data)
    asi_seq = coerce_asi_seq_index(asi_seq)
    print("[DATA] columns:", list(asi_seq.columns))

    splits = ts_cv_splits_5(asi_seq, n_splits=args.fold, L=args.L, extra_embargo_min=args.embargo_extra) if hasattr(args, "fold") else ts_cv_splits_5(asi_seq, n_splits=args.folds, L=args.L, extra_embargo_min=args.embargo_extra)
    mean, std = estimate_mean_std(splits[0][0], out_hw=OUT_HW, apply_sky=False)
    print("[NORM] mean:", np.round(mean,4), "std:", np.round(std,4))

    # model/optim
    model = UNet_ConvLSTM(in_ch=3, base=32, bidirectional=args.bidirectional).to(DEVICE)
    if args.channels_last and isinstance(DEVICE, torch.device) and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # explicit resume (optional)
    global_start_epoch, global_best, _ = try_resume_from_path(
        Path(args.resume_path) if args.resume_path else None,
        model, optim, scaler, DEVICE
    )

    for (train_seq, val_seq, info) in splits:
        fold_id = info["fold"]; fold_dir = out_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "fold_info.json", "w") as f:
            json.dump({k: (str(v) if "time" in k else v) for k, v in info.items()}, f, indent=2)

        print(f"\n[Fold {fold_id}] val {info['val_start']} → {info['val_end']} | "
              f"train windows: {len(train_seq)} | val windows: {len(val_seq)}")

        # Build loaders with Windows-safe fallback
        tr_loader, va_loader = build_loaders_with_fallback(
            train_df=train_seq, val_df=val_seq,
            out_hw=OUT_HW, bs=args.bs, num_workers=args.num_workers,
            mean=mean, std=std, device=DEVICE, prefetch_factor=args.prefetch_factor
        )
        loaders = {"train": tr_loader, "val": va_loader}

        # If no explicit resume_path, use per-fold policy
        if args.resume_path is None:
            start_epoch, best_val = try_resume_fold(fold_dir, args.resume_from, model, optim, scaler, DEVICE)
        else:
            start_epoch, best_val = global_start_epoch, (global_best if global_best is not None else float("inf"))
            args.resume_path = None

        if best_val is None: best_val = float("inf")

        # probe minutes per step
        try:
            first_batch = next(iter(tr_loader))
        except StopIteration:
            print(f"[WARN] Fold {fold_id} empty train loader. Skipping.")
            try:
                vb = next(iter(va_loader))
                step_min = infer_minutes_per_step(vb["timestamps"][0]) if vb["timestamps"] else 1.0
                step_min = max(step_min, 1e-6)
                horizons_steps = sorted(set([max(1, int(round(m/step_min))) for m in horizons_min]))
                print(f"[FORECAST] minutes_per_step≈{step_min:.3f} | horizons_min={horizons_min} -> steps={horizons_steps}")
                dump_samples(model, vb, fold_dir/"samples", tag="init", mean=mean, std=std, device=DEVICE, grid_cols=5,
                             horizons_steps=horizons_steps)
            except StopIteration:
                pass
            continue

        step_min = infer_minutes_per_step(first_batch["timestamps"][0]) if first_batch["timestamps"] else 1.0
        step_min = max(step_min, 1e-6)
        horizons_steps = sorted(set([max(1, int(round(m/step_min))) for m in horizons_min]))
        print(f"[FORECAST] minutes_per_step≈{step_min:.3f} | horizons_min={horizons_min} -> steps={horizons_steps}")

        dump_samples(model, first_batch, fold_dir/"samples", tag="init",
                     mean=mean, std=std, device=DEVICE, grid_cols=5,
                     horizons_steps=horizons_steps)

        log_path = fold_dir / "log.jsonl"

        try:
            for epoch in range(start_epoch, args.epochs + 1):
                t0 = time.time()

                rec_tr = train_one_epoch(
                    model, loaders, optim, DEVICE,
                    scaler=scaler, amp_enabled=amp_enabled,
                    horizons_steps=horizons_steps, amp_ctx=amp_ctx,
                    alpha=args.alpha_ssim, beta=args.beta_edge, gamma=args.gamma_smooth, eta=args.eta_horizon,
                    mean=mean, std=std,
                    max_train_batches=args.max_train_batches,
                    accum_steps=max(1, int(args.accum_steps))
                )

                metrics = evaluate(
                    model, loaders, DEVICE,
                    horizons_steps=horizons_steps, mean=mean, std=std,
                    alpha=args.alpha_ssim, beta=args.beta_edge, gamma=args.gamma_smooth, eta=args.eta_horizon,
                    max_val_batches=args.max_val_batches,
                    amp_enabled=amp_enabled
                )

                dt = time.time() - t0
                print(f"[Fold {fold_id}][Epoch {epoch:03d}] train_rec={rec_tr:.4f} | "
                      f"val_rec={metrics['rec']:.4f} psnr={metrics['psnr']:.2f} | {dt:.1f}s")

                with open(log_path, "a") as f:
                    rec_json = dict(epoch=epoch, train_rec=rec_tr, **metrics)
                    f.write(json.dumps(rec_json) + "\n")

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
                atomic_save(last_ckpt, fold_dir / "last.pt")

                val_score = metrics["rec"]
                if epoch == 1 or val_score < best_val - 1e-6:
                    best_val = val_score
                    atomic_save(last_ckpt, fold_dir / "best.pt")
                    atomic_save(last_ckpt, fold_dir / f"best_epoch{epoch:03d}.pt")
                    dump_samples(model, first_batch, fold_dir / "samples",
                                 tag=f"best_e{epoch:03d}", mean=mean, std=std, device=DEVICE, grid_cols=5,
                                 horizons_steps=horizons_steps)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Caught Ctrl+C — saving interrupted checkpoint...")
            ckpt = {
                "model": model.state_dict(), "optim": optim.state_dict(),
                "epoch": epoch if 'epoch' in locals() else 0,
                "args": vars(args), "metrics": metrics if 'metrics' in locals() else {},
                "mean": mean, "std": std,
            }
            atomic_save(ckpt, fold_dir / "interrupted.pt")
            print(f"[INTERRUPT] Saved: {fold_dir/'interrupted.pt'}")
            return

        dump_samples(model, first_batch, fold_dir/"samples", tag="final",
                     mean=mean, std=std, device=DEVICE,
                     horizons_steps=horizons_steps)

    print("\n[Done] All folds complete. See:", out_root)

# =============== Guard ===============
if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        print(">>> [BOOT] starting main()")
        main()
        print(">>> [DONE] main() returned normally")
    except SystemExit as e:
        print(f">>> [SystemExit] code={e.code}")
        raise
    except Exception:
        print(">>> [EXCEPTION] Uncaught error in main():")
        traceback.print_exc()
        sys.exit(1)