# app/utils_io.py
import io
import os
import tempfile
import numpy as np
import nibabel as nib
from typing import Tuple, Dict, Any
from PIL import Image  # 读写 PNG/JPG

def _is_gz(data: bytes) -> bool:
    # gzip 头 0x1f 0x8b
    return len(data) >= 2 and data[:2] == b"\x1f\x8b"

def _is_zip(data: bytes) -> bool:
    # zip/npz 头 'PK'
    return len(data) >= 2 and data[:2] == b"PK"

def load_image_any(fobj: io.BytesIO, modality: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    统一读入多种输入：
      - nifti: 返回 (D,H,W) 或 (C,D,H,W)，meta['affine']
      - png/jpg/jpeg: 返回 (H,W,C) float32
      - npz: 期望有 key 'image'，形状 (C,H,W,D)；返回 (C,D,H,W)
    """
    modality = modality.lower().strip()

    # ---------- NIfTI ----------
    if modality == "nifti":
        data = fobj.read()
        try:
            # 直接在内存读，并禁用 mmap，避免后续访问已删除的临时文件
            img = nib.load(io.BytesIO(data), mmap=False)
            affine = img.affine
            arr = img.get_fdata(dtype=np.float32)
        except Exception:
            # 根据魔数决定写成 .nii 或 .nii.gz，再读（仍然 mmap=False）
            if _is_zip(data):
                raise ValueError("Uploaded file looks like ZIP/NPZ (starts with 'PK'), not a NIfTI (.nii/.nii.gz).")
            suffix = ".nii.gz" if _is_gz(data) else ".nii"
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, f"upload{suffix}")
                with open(tmp, "wb") as fp:
                    fp.write(data)
                img = nib.load(tmp, mmap=False)
                affine = img.affine
                arr = img.get_fdata(dtype=np.float32)

        # 统一 shape：3D -> (D,H,W)，4D -> (C,D,H,W)
        if arr.ndim == 3:      # (H,W,D) -> (D,H,W)
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 4:    # (C,H,W,D) -> (C,D,H,W)
            arr = np.transpose(arr, (0, 3, 1, 2))
        else:
            raise ValueError(f"NIfTI ndim must be 3/4, got {arr.ndim}")
        return arr, {"affine": affine}

    # ---------- PNG/JPG ----------
    if modality in ("png", "jpg", "jpeg"):
        img = Image.open(fobj).convert("RGB")
        arr = np.array(img).astype(np.float32)  # (H,W,3)
        return arr, {}

    # ---------- NPZ ----------
    if modality == "npz":
        data = np.load(fobj, allow_pickle=False)
        if "image" not in data:
            raise KeyError("NPZ missing key 'image'")
        arr = data["image"]  # 期望 (C,H,W,D)
        if arr.ndim != 4:
            raise ValueError(f"NPZ 'image' must be (C,H,W,D), got {arr.shape}")
        arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)  # -> (C,D,H,W)
        return arr, {}

    # ---------- 其他 ----------
    raise ValueError(f"Unsupported modality: {modality}")

def save_nifti_bytes(arr: np.ndarray, affine=None) -> bytes:
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    nii = nib.Nifti1Image(arr.astype(np.uint8), affine)

    # 用临时文件保存，再读回字节，避免 BytesIO 兼容性问题
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, "mask.nii.gz")  # 直接保存为 .nii.gz
        nib.save(nii, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    return data

def save_png_bytes(mask: np.ndarray) -> bytes:
    if mask.ndim != 2:
        raise ValueError("PNG mask must be 2D")
    img = Image.fromarray((mask * 255).astype(np.uint8))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def save_npz_bytes(**arrays) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    return bio.getvalue()
