# eval_s2_v3.py
# ------------------------------------------------------------
# Evaluate S2_cae_unet_convlstm_v3 checkpoints (single or all folds)
# on a month or explicit date range.
# ------------------------------------------------------------
import os, json, math, gzip, pickle, argparse, platform, sys, tempfile
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========= tiny utils =========
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_any_pickle(path):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    with open(p, "rb") as fh: head = fh.read(2)
    def _pl(fobj):
        try: return pickle.load(fobj)
        except Exception:
            fobj.seek(0); return pickle.load(fobj, encoding="latin1")
    if head == b"\x1f\x8b":
        import gzip as _gz
        with _gz.open(p, "rb") as f: return _pl(f)
    else:
        with open(p, "rb") as f: return _pl(f)

def save_image_grid(rgb_list, path, ncols=None):
    if not rgb_list: return
    h, w, _ = rgb_list[0].shape
    n = len(rgb_list); ncols = ncols or int(math.ceil(math.sqrt(n))); nrows = int(math.ceil(n / ncols))
    canvas = np.zeros((nrows*h, ncols*w, 3), np.uint8)
    for i, img in enumerate(rgb_list):
        r, c = divmod(i, ncols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)

def tensor_to_uint8(img_t, mean, std):
    x = img_t.detach().cpu().float()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
    x = (x * std_t + mean_t).clamp(0,1) * 255.0
    return x.byte().permute(1,2,0).numpy()

# ========= index & filtering =========
def _first_ts(x):
    if isinstance(x, (list, tuple)) and len(x)>0: return x[0]
    try: return x[0]
    except Exception: return pd.NaT

def _last_ts(x):
    if isinstance(x, (list, tuple)) and len(x)>0: return x[-1]
    try: return x[-1]
    except Exception: return pd.NaT

def coerce_asi_seq_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "start" in df.columns:
            idx = pd.to_datetime(df["start"], errors="coerce")
        elif "timestamps" in df.columns:
            idx = pd.to_datetime(df["timestamps"].apply(_first_ts), errors="coerce")
            df["start"] = idx
        else:
            raise ValueError("No 'start' or 'timestamps' to form DatetimeIndex.")
        df["_idx"] = idx
        df = df.dropna(subset=["_idx"]).set_index("_idx", drop=True).drop(columns=["_idx"])
    df.index.name = "start"
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    if "end" not in df.columns:
        if "timestamps" in df.columns:
            df["end"] = pd.to_datetime(df["timestamps"].apply(_last_ts), errors="coerce")
        else:
            df["end"] = df.index
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df.sort_index()

def filter_by_month(seq_df: pd.DataFrame, month_str: str) -> pd.DataFrame:
    y, m = map(int, month_str.split("-"))
    lb = pd.Timestamp(year=y, month=m, day=1)
    ub = (lb + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return seq_df.loc[(seq_df.index >= lb) & (seq_df.index <= ub)].copy()

def filter_by_dates(seq_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    lb, ub = pd.Timestamp(start), pd.Timestamp(end)
    return seq_df.loc[(seq_df.index >= lb) & (seq_df.index <= ub)].copy()

def infer_minutes_per_step(ts_list):
    try:
        vals = []
        for i in range(1, len(ts_list)):
            t0, t1 = pd.to_datetime(ts_list[i-1], errors="coerce"), pd.to_datetime(ts_list[i], errors="coerce")
            if pd.isna(t0) or pd.isna(t1): continue
            dmin = (t1 - t0).total_seconds()/60.0
            if dmin > 0: vals.append(dmin)
        return float(np.median(vals)) if vals else 1.0
    except Exception:
        return 1.0

# ========= dataset (eval) =========
def _imread_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _ensure_np_img(img_like, fallback_path):
    if isinstance(img_like, np.ndarray): return img_like
    if fallback_path is not None: return _imread_rgb(fallback_path)
    return None

def _resize_img(img, size_hw):
    H,W = size_hw; return cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)

def _resize_mask(mask, size_hw):
    H,W = size_hw; return cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

def _img_to_01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32); mx = float(np.nanmax(x)) if x.size else 0.0
    if img.dtype == np.uint8 or mx > 1.5: x /= 255.0
    return np.clip(x, 0.0, 1.0)

def _apply_sky_mask_to_img(img01: np.ndarray, sky_mask) -> np.ndarray:
    if not isinstance(sky_mask, np.ndarray): return img01
    H,W = img01.shape[:2]; m = sky_mask
    if m.ndim==3 and m.shape[2]==1: m = m[...,0]
    if m.shape[:2] != (H,W): m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
    m = (m>0)
    if m.all(): return img01
    if not m.any(): return np.zeros_like(img01, dtype=np.float32)
    return img01 * m[...,None].astype(np.float32)

def _to_tensor_img(img01): return torch.from_numpy(img01.transpose(2,0,1))

def stack_window(row, out_hw=(128,128), apply_sky_to_inputs=True, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), input_colorspace=None, **_):
    imgs = row["images"]; sky_list = row.get("sky_masks", [None]*len(imgs)); fpaths = row.get("filepaths", [None]*len(imgs))
    ts_raw = row["timestamps"]
    Ht,Wt = out_hw; X_list, SKY_list = [], []
    cs = (str(input_colorspace).upper() if input_colorspace is not None else None)
    for im_like, sky_like, fp in zip(imgs, sky_list, fpaths):
        im = _ensure_np_img(im_like, fp)
        if isinstance(im, np.ndarray) and im.ndim==3 and im.shape[2]==3 and cs:
            if cs=="HSV": im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
            elif cs=="BGR": im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im is None:
            img01 = np.zeros((Ht,Wt,3), np.float32); sky_resized=None
        else:
            im = _resize_img(im, (Ht,Wt)); img01 = _img_to_01(im)
            if isinstance(sky_like, np.ndarray):
                sky_resized = _resize_mask(sky_like, (Ht,Wt)); sky_resized = (sky_resized>0).astype(np.uint8)
            else:
                sky_resized=None
            if apply_sky_to_inputs and (sky_resized is not None):
                img01 = _apply_sky_mask_to_img(img01, sky_resized)
        X_list.append(_to_tensor_img(img01))
        SKY_list.append(torch.from_numpy(sky_resized) if isinstance(sky_resized, np.ndarray) else None)
    X = torch.stack(X_list, 0)  # [T,3,H,W]
    if all(s is None for s in SKY_list):
        SKY = torch.ones((X.shape[0], Ht, Wt), dtype=torch.uint8)
    else:
        SKY = torch.stack([s if s is not None else torch.ones((Ht,Wt), dtype=torch.uint8) for s in SKY_list], 0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1,3,1,1).clamp_min(1e-6)
    X = (X - mean_t)/std_t
    ts = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in ts_raw]
    return X, SKY, ts

class EvalDataset(Dataset):
    def __init__(self, df, out_hw=(128,128), mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.df=df; self.out_hw=out_hw; self.mean=mean; self.std=std
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        X, SKY, ts = stack_window(row, out_hw=self.out_hw, mean=self.mean, std=self.std)
        return {"x_in": X.clone(), "y_img": X, "sky": SKY, "timestamps": ts}

def build_loader(df, out_hw, bs, mean, std, device, num_workers=4, prefetch_factor=4):
    is_win = platform.system().lower().startswith("win")
    is_cuda = (isinstance(device, torch.device) and device.type=="cuda")
    if is_win or not is_cuda:
        num_workers=0; prefetch_factor=2
    pin = bool(is_cuda)
    ds = EvalDataset(df, out_hw=out_hw, mean=mean, std=std)
    kw = dict(batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)
    if num_workers>0: kw["prefetch_factor"]=prefetch_factor
    return DataLoader(ds, **kw)

# ========= model (exactly like train) =========
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
        c = f*c + i*g; h = o*torch.tanh(c)
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
        B,T,C,H,W = x_seq.shape; device = x_seq.device
        h_f = x_seq
        for cell in self.fwd:
            h,c = cell.init_state(B,H,W,device); outs=[]
            for t in range(T):
                h,c = cell(h_f[:,t], (h,c)); outs.append(h)
            h_f = torch.stack(outs,1)
        if not self.bidirectional: return h_f
        h_b = x_seq.flip(1)
        for cell in self.bwd:
            h,c = cell.init_state(B,H,W,device); outs=[]
            for t in range(T):
                h,c = cell(h_b[:,t], (h,c)); outs.append(h)
            h_b = torch.stack(outs,1)
        return torch.cat([h_f, h_b.flip(1)], dim=2)

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
    def forward(self, x):
        B,T,C,H,W = x.shape
        e1_list,e2_list,e3_list = [],[],[]
        for t in range(T):
            f = x[:,t]; e1 = self.enc1(f); p1 = self.pool1(e1)
            e2 = self.enc2(p1); p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            e1_list.append(e1); e2_list.append(e2); e3_list.append(e3)
        E3 = torch.stack(e3_list,1)
        Hs = self.temporal(E3)
        outs=[]
        for t in range(T):
            h = self.bott_reduce(Hs[:,t])
            u2 = self.up2(h)
            d2 = self.dec2(torch.cat([u2, e2_list[t]],1))
            u1 = self.up1(d2)
            d1 = self.dec1(torch.cat([u1, e1_list[t]],1))
            outs.append(self.recon_head(d1))
        return torch.stack(outs,1)

# ========= SSIM & metrics =========
class _SSIMComputer:
    def __init__(self): self.cache={}
    def get_kernel(self, device, dtype):
        key=(str(device),str(dtype)); k=self.cache.get(key)
        if k is None:
            ax = torch.arange(11, dtype=dtype, device=device) - 5
            k1 = torch.exp(-(ax**2)/(2*1.5**2)); k1 /= k1.sum()
            k = torch.outer(k1,k1).reshape(1,1,11,11); self.cache[key]=k
        return self.cache[key]
_SSIM = _SSIMComputer()

def _depthwise_conv2d_same(x, k2):
    B,C,H,W = x.shape
    k = k2.to(device=x.device, dtype=x.dtype)
    weight = k.expand(C,1,k.size(-2),k.size(-1)).contiguous()
    return F.conv2d(x, weight, stride=1, padding=k.size(-1)//2, groups=C)

def ssim_01(x, y, C1=0.01**2, C2=0.03**2):
    x = x.float().contiguous(); y = y.float().contiguous()
    k2 = _SSIM.get_kernel(x.device, x.dtype)
    mu_x = _depthwise_conv2d_same(x, k2); mu_y = _depthwise_conv2d_same(y, k2)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sigma_x2 = _depthwise_conv2d_same(x*x, k2) - mu_x2
    sigma_y2 = _depthwise_conv2d_same(y*y, k2) - mu_y2
    sigma_xy = _depthwise_conv2d_same(x*y, k2) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-8)
    return ssim_map.mean()

def denorm_to_01(x, mean, std):
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1,-1,1,1)
    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1,-1,1,1).clamp_min(1e-6)
    return (x*s + m).clamp(0,1)

@torch.no_grad()
def metrics_over_horizons(pred, target, horizon_steps, mean, std) -> Dict[str,float]:
    B,T,C,H,W = pred.shape
    out = {}
    ps, ss, ms, rs = [], [], [], []
    for h in horizon_steps:
        if h<=0 or h>T: continue
        Teff = T - h
        if Teff<=0: continue
        p = denorm_to_01(pred[:, :Teff], mean, std).flatten(0,1)
        g = denorm_to_01(target[:, h:h+Teff], mean, std).flatten(0,1)
        diff = p - g
        mse = diff.pow(2).mean().item()
        mae = diff.abs().mean().item()
        rmse = math.sqrt(max(mse,0.0))
        psnr = (99.0 if mse<1e-10 else min(99.0, 10.0*math.log10(1.0/(mse+1e-10))))
        ssim_v = float(ssim_01(p,g).item())
        out[f"psnr_h{h}"]=psnr; out[f"ssim_h{h}"]=ssim_v; out[f"mae_h{h}"]=mae; out[f"rmse_h{h}"]=rmse
        ps.append(psnr); ss.append(ssim_v); ms.append(mae); rs.append(rmse)
    out["psnr"]=float(np.mean(ps)) if ps else 0.0
    out["ssim"]=float(np.mean(ss)) if ss else 0.0
    out["mae"]=float(np.mean(ms)) if ms else 0.0
    out["rmse"]=float(np.mean(rs)) if rs else 0.0
    return out

def dump_samples(model, batch, outdir, tag, mean, std, device, grid_cols=5, horizons_steps=None):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    x_in = batch["x_in"][:1].to(device); y_img = batch["y_img"][:1].to(device)
    with torch.no_grad():
        pred = model(x_in)
    xin = x_in[0]; yi = y_img[0]; rp = pred[0]; T = rp.shape[0]
    _to_u8 = lambda t: tensor_to_uint8(t, mean, std)
    save_image_grid([_to_u8(xin[t]) for t in range(T)], Path(outdir)/f"{tag}_inputs.png", ncols=grid_cols)
    feasible = [h for h in (horizons_steps or [1]) if h < T]
    h0 = min(feasible) if feasible else 1
    Teff = max(0, T - h0)
    save_image_grid([_to_u8(rp[t]) for t in range(Teff)], Path(outdir)/f"{tag}_pred_h{h0}.png", ncols=grid_cols)
    save_image_grid([_to_u8(yi[t+h0]) for t in range(Teff)], Path(outdir)/f"{tag}_gt_h{h0}.png",   ncols=grid_cols)

# ========= device =========
def get_device(req="auto"):
    req = (req or "auto").lower()
    if req=="auto": req = "cuda" if torch.cuda.is_available() else "cpu"
    if req in ("cuda","cpu","mps"):
        return torch.device(req)
    return torch.device("cpu")

# ========= fold evaluator =========
def evaluate_ckpt(ckpt_path: Path, df_eval: pd.DataFrame, out_hw: Tuple[int,int],
                  device: torch.device, bs: int, max_batches: int=0) -> Dict[str,float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) or {}
    mean = np.asarray(ckpt.get("mean", [0.5,0.5,0.5]), dtype=np.float32)
    std  = np.asarray(ckpt.get("std",  [0.5,0.5,0.5]), dtype=np.float32)

    bidir = bool(args.get("bidirectional", False))
    base  = int(args.get("base", 32)) if "base" in args else 32
    out_h = int(args.get("out_h", out_hw[0])); out_w = int(args.get("out_w", out_hw[1]))
    out_hw_used = (out_h, out_w)

    model = UNet_ConvLSTM(in_ch=3, base=base, bidirectional=bidir).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    horizons_steps = ckpt.get("horizons_steps", None)
    if not horizons_steps:
        # reconstruct from minutes if needed
        try:
            horizons_min = [int(x) for x in str(args.get("horizons","1,3,5,10")).split(",") if str(x).strip()]
        except Exception:
            horizons_min = [1,3,5,10]
        ts0 = df_eval.iloc[0]["timestamps"]
        step_min = max(1e-6, infer_minutes_per_step(ts0))
        horizons_steps = sorted(set([max(1, int(round(m/step_min))) for m in horizons_min]))

    loader = build_loader(df_eval, out_hw=out_hw_used, bs=bs, mean=mean, std=std, device=device)

    agg = {"psnr":[], "ssim":[], "mae":[], "rmse":[]}
    per_h = {f: [] for f in [*(f"psnr_h{h}" for h in horizons_steps),
                             *(f"ssim_h{h}" for h in horizons_steps),
                             *(f"mae_h{h}"  for h in horizons_steps),
                             *(f"rmse_h{h}" for h in horizons_steps)]}
    n_batches=0
    for ib, batch in enumerate(tqdm(loader, desc=f"Eval {ckpt_path.parent.name}", leave=False), 1):
        x_in  = batch["x_in"].to(device, non_blocking=(device.type=="cuda"))
        y_img = batch["y_img"].to(device, non_blocking=(device.type=="cuda"))
        with torch.no_grad():
            pred = model(x_in)
        mets = metrics_over_horizons(pred, y_img, horizons_steps, mean, std)
        for k in ("psnr","ssim","mae","rmse"): agg[k].append(mets[k])
        for k,v in mets.items():
            if k in per_h: per_h[k].append(v)
        if ib==1:
            dump_samples(model, batch, ckpt_path.parent/"eval_samples", tag="eval", mean=mean, std=std,
                         device=device, grid_cols=5, horizons_steps=horizons_steps)
        n_batches += 1
        if max_batches and n_batches>=max_batches: break

    out = {k: float(np.mean(v)) if v else 0.0 for k,v in agg.items()}
    for k,vs in per_h.items(): out[k] = float(np.mean(vs)) if vs else 0.0
    out["batches"]=n_batches; out["frames_hw"]=f"{out_hw_used[0]}x{out_hw_used[1]}"; out["bidirectional"]=int(bidir)
    return out

# ========= driver =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-root", type=str, default=None, help="runs-s2/<exp> containing fold_*/best.pt")
    ap.add_argument("--ckpt-path", type=str, default=None, help="Single checkpoint: .../fold_k/best.pt")
    ap.add_argument("--month", type=str, default=None, help="YYYY-MM")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="eval-s2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out-h", type=int, default=None)
    ap.add_argument("--out-w", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.ckpt_root and not args.ckpt_path:
        sys.exit("Specify either --ckpt-root or --ckpt-path")

    if args.month:
        if args.start or args.end:
            sys.exit("Use either --month or --start/--end, not both.")
    else:
        if not (args.start and args.end):
            sys.exit("Specify --month YYYY-MM  OR  --start YYYY-MM-DD --end YYYY-MM-DD")

    set_seed(args.seed)
    device = get_device(args.device)

    df = load_any_pickle(args.data)
    df = coerce_asi_seq_index(df)
    eval_df = filter_by_month(df, args.month) if args.month else filter_by_dates(df, args.start, args.end)
    if len(eval_df)==0: sys.exit("No windows found for the requested range.")
    print(f"[EVAL] windows: {len(eval_df)} | time span: {eval_df.index.min()} → {eval_df['end'].max()}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Single checkpoint mode
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        out_hw = (args.out_h, args.out_w) if (args.out_h and args.out_w) else (128,128)
        res = evaluate_ckpt(ckpt_path, eval_df, out_hw, device=device, bs=args.bs, max_batches=args.max_batches)
        res["fold"] = ckpt_path.parent.name
        (out_dir / "single_metrics.json").write_text(json.dumps(res, indent=2))
        pd.DataFrame([res]).to_csv(out_dir / "single_metrics.csv", index=False)
        print("[SINGLE] PSNR={psnr:.2f}  SSIM={ssim:.4f}  MAE={mae:.5f}  RMSE={rmse:.5f}".format(**res))
        return

    # Root (all folds) mode
    root = Path(args.ckpt_root)
    folds = sorted([p for p in root.glob("fold_*") if p.is_dir()])
    if not folds: sys.exit(f"No fold_* under {root}")
    rows=[]
    for fd in folds:
        best = fd / "best.pt"
        if not best.exists():
            print(f"[WARN] {best} missing; skip"); continue
        out_hw = (args.out_h, args.out_w) if (args.out_h and args.out_w) else (128,128)
        m = evaluate_ckpt(best, eval_df, out_hw, device=device, bs=args.bs, max_batches=args.max_batches)
        m["fold"] = fd.name
        rows.append(m)
        (out_dir / f"{fd.name}_metrics.json").write_text(json.dumps(m, indent=2))
    if not rows: sys.exit("No checkpoints evaluated.")
    dfm = pd.DataFrame(rows).set_index("fold"); dfm.to_csv(out_dir/"fold_metrics.csv")
    mean_row = dfm.mean(numeric_only=True)
    mean_row.to_frame("mean_over_folds").to_csv(out_dir/"summary_mean.csv")
    (out_dir/"summary.json").write_text(json.dumps({"mean":mean_row.to_dict(),"per_fold":dfm.to_dict(orient="index")}, indent=2))
    print("\n[SUMMARY] Mean over folds:")
    print(mean_row[["psnr","ssim","mae","rmse"]].to_string())

if __name__ == "__main__":
    main()