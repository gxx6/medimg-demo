# app/inference.py
import os
import numpy as np
import torch

# === 模型定义 ===
from nets.simple3DUnet import simple_uet_model  # 确保该模块在 PYTHONPATH 中（放进项目或可安装包）

IN_CHANNELS = int(os.getenv("IN_CHANNELS", "3"))   # 你的模型输入通道，默认3
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "4"))   # 类别数（含背景）
USE_SLIDING = os.getenv("USE_SLIDING", "0") == "1" # 是否启用滑窗
TILE_D      = int(os.getenv("TILE_D", "96"))
OVERLAP     = float(os.getenv("OVERLAP", "0.5"))

_device = None
_model  = None

def _next_mul(x, k=16):
    return int((x + k - 1) // k) * k

def _pad_to_multiple(x_np, k=16):
    # x_np: (C,D,H,W) 或 (D,H,W)
    import numpy as np
    has_channel = (x_np.ndim == 4)
    if not has_channel:
        x_np = x_np[None, ...]  # -> (1,D,H,W)

    C, D, H, W = x_np.shape
    Dn, Hn, Wn = _next_mul(D, k), _next_mul(H, k), _next_mul(W, k)
    pad_d, pad_h, pad_w = Dn - D, Hn - H, Wn - W

    # 右侧补零（也可以用镜像填充）
    x_pad = np.pad(x_np,
                   pad_width=((0,0),(0,pad_d),(0,pad_h),(0,pad_w)),
                   mode="constant",
                   constant_values=0)
    meta = {"orig_shape": (D,H,W), "pad": (pad_d, pad_h, pad_w)}
    return x_pad, meta, has_channel

def _unpad_mask(mask, meta):
    # mask: (D,H,W)
    D,H,W = meta["orig_shape"]
    return mask[:D, :H, :W]

def _predict_depth_tiled(x, model, tile_d=96, overlap=0.5):
    """仅沿 D 维滑窗：x (1,C,D,H,W) -> logits (1,K,D,H,W)"""
    assert x.ndim == 5 and x.size(0) == 1
    _, _, D, H, W = x.shape
    step = max(1, int(tile_d * (1 - overlap)))

    logits_sum = None
    weight_1d  = torch.zeros((D,), device=x.device, dtype=torch.float32)

    for start in range(0, D, step):
        end = min(start + tile_d, D)
        out = model(x[:, :, start:end, :, :])  # (1,K,d,H,W)
        if logits_sum is None:
            K = out.size(1)
            logits_sum = torch.zeros((1, K, D, H, W), device=x.device, dtype=out.dtype)
        logits_sum[:, :, start:end, :, :] += out
        weight_1d[start:end] += 1.0
        if end == D:
            break

    logits_sum /= weight_1d.view(1, 1, D, 1, 1).clamp_min(1.0)
    return logits_sum

def load_model(model_path: str = None, device_str: str = None):
    global _device, _model
    model_path = model_path or os.getenv("MODEL_PATH", r"E:\DeepLearning\Deploy\weights\best_epoch_weights.pth")
    _device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = simple_uet_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    ckpt  = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.to(_device).eval()
    _model = model
    return _model


@torch.no_grad()
def predict_volume(volume: np.ndarray, meta: dict = None):
    # 统一 (C,D,H,W)
    if volume.ndim == 3:
        volume = volume[None, ...]  # -> (1,D,H,W)

    # 通道对齐到 3
    if volume.shape[0] == 1:
        volume = np.repeat(volume, 3, axis=0)   # -> (3,D,H,W)
    elif volume.shape[0] > 3:
        volume = volume[:3]

    # ---- 关键：pad 到 16 的倍数 ----
    vol_pad, pad_meta, _ = _pad_to_multiple(volume, k=16)  # vol_pad: (3, Dp, Hp, Wp)

    # ✅ 用 vol_pad 构造张量，而不是用 volume
    x = torch.from_numpy(vol_pad.astype(np.float32, copy=False))[None, ...].to(_device)  # (1,3,Dp,Hp,Wp)

    # 若滑窗，保证 TILE_D 合法
    if USE_SLIDING:
        Dp = x.shape[2]
        # 让 TILE_D 成为 16 的倍数，且不超过 Dp
        tile_d = min(((TILE_D + 15) // 16) * 16, Dp)
    else:
        tile_d = None

    with torch.cuda.amp.autocast(enabled=(_device.type == "cuda")):
        if USE_SLIDING:
            logits = _predict_depth_tiled(x, _model, tile_d=tile_d, overlap=OVERLAP)
        else:
            logits = _model(x)  # (1,K,Dp,Hp,Wp)
        probs = torch.softmax(logits, dim=1)
        pred  = torch.argmax(probs, dim=1)  # (1,Dp,Hp,Wp)

    mask_pad = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # (Dp,Hp,Wp)
    mask = _unpad_mask(mask_pad, pad_meta)  # -> (D,H,W)
    return mask
