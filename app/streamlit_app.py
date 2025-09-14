import streamlit as st
import numpy as np
import requests, io
from PIL import Image
import nibabel as nib
import tempfile,os

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="MedImg Seg Demo", layout="wide")
st.title("Medical Imaging Segmentation Demo")

tab1, tab2 = st.tabs(["NIfTI (MRI/CT)", "PNG/JPG",])
tab3 = st.tabs(["NPZ (image key)"])[0]

def _is_gz(data: bytes) -> bool:
    # gzip 魔数：0x1f 0x8b
    return len(data) >= 2 and data[:2] == b"\x1f\x8b"


def overlay_rgba(base_gray: np.ndarray, mask2d: np.ndarray, alpha: float=0.4):
    """将二值 mask 叠加到灰度图，返回 RGB 图像"""
    g = base_gray
    g = (255*(g - g.min())/(g.ptp()+1e-8)).astype(np.uint8)
    rgb = np.stack([g,g,g], axis=-1).astype(np.uint8)
    color = np.zeros_like(rgb)
    color[...,0] = 255  # 红色通道表示 mask
    mask3 = (mask2d>0)[...,None]
    out = (rgb*(1-alpha) + color*alpha*mask3).clip(0,255).astype(np.uint8)
    return out

with tab1:
    st.subheader("Upload NIfTI (.nii / .nii.gz)")
    nifti_file = st.file_uploader("Choose a NIfTI file", type=["nii","nii.gz"], key="nifti")
    if nifti_file is not None and st.button("Segment (NIfTI)"):
        # 后端预测
        src_bytes = nifti_file.getvalue()
        files = {"file": (nifti_file.name, src_bytes, "application/octet-stream")}
        resp = requests.post(API_URL, files=files, data={"modality":"nifti"})
        if resp.status_code != 200:
            st.error(resp.text)
        else:
            # 1) 读取后端返回的 mask（后端我们固定返回 .nii.gz，所以后缀就用 .nii.gz）
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_mask:
                tmp_mask.write(resp.content)
                tmp_mask.flush()
                tmp_mask_path = tmp_mask.name
            try:
                mask_img = nib.load(tmp_mask_path, mmap=False)
                mask = mask_img.get_fdata().astype(np.uint8)  # (D,H,W)
            finally:
                try: os.unlink(tmp_mask_path)
                except Exception: pass

            # 2) 读取上传的原图：根据内容判断后缀（.nii or .nii.gz）
            suffix = ".nii.gz" if _is_gz(src_bytes) else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_src:
                tmp_src.write(src_bytes)
                tmp_src.flush()
                tmp_src_path = tmp_src.name
            try:
                orig_img = nib.load(tmp_src_path, mmap=False)
                orig = orig_img.get_fdata().astype(np.float32)  # (H,W,D) 或 (C,H,W,D)
            finally:
                try: os.unlink(tmp_src_path)
                except Exception: pass

            # 统一到 (D,H,W)
            if orig.ndim == 4:
                orig = orig.mean(3)  # -> (H,W,D)

            # 将原图转换为 (D,H,W) —— 以 mask 的 depth 为基准
            if orig.ndim == 3:
                if orig.shape[2] == mask.shape[0]:  # (H,W,D)
                    orig = np.transpose(orig, (2, 0, 1))  # -> (D,H,W)
                elif orig.shape[0] == mask.shape[0]:  # 已经是 (D,H,W)
                    pass
                elif orig.shape[1] == mask.shape[0]:  # (H,D,W) 这种少见情况
                    orig = np.transpose(orig, (1, 0, 2))  # -> (D,H,W)
                else:
                    # 实在对不上就直接按 (H,W,D) 处理
                    orig = np.transpose(orig, (2, 0, 1))  # 假定最后一维是深度

            d = st.slider("Slice index", 0, int(mask.shape[0] - 1), int(mask.shape[0] // 2))
            alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.4, 0.05)

            base2d = orig[d].astype(np.float32)  # (H?,W?)
            mask2d = mask[d].astype(np.uint8)  # (H,W)

            # 如果 H/W 还不一致，按 mask 的大小把 base2d resize 一下
            if base2d.shape != mask2d.shape:
                from PIL import Image

                base2d = np.array(
                    Image.fromarray(base2d).resize(
                        (mask2d.shape[1], mask2d.shape[0]),  # (W,H)
                        resample=Image.BILINEAR
                    )
                ).astype(np.float32)

            vis = overlay_rgba(base2d, mask2d, alpha)
            st.image(vis, caption=f"Slice {d}", use_column_width=True)

            st.download_button("Download mask NIfTI",
                               data=resp.content,
                               file_name="mask.nii.gz",
                               mime="application/octet-stream")
with tab2:
    st.subheader("Upload PNG/JPG")
    img_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"], key="img")
    if img_file is not None and st.button("Segment (Image)"):
        files = {"file": (img_file.name, img_file.getvalue(), "image/png")}
        resp = requests.post(API_URL, files=files, data={"modality":"png"})
        if resp.status_code != 200:
            st.error(resp.text)
        else:
            mask_png = Image.open(io.BytesIO(resp.content)).convert("L")
            mask = np.array(mask_png)>127

            img = Image.open(io.BytesIO(img_file.getvalue())).convert("L")
            base = np.array(img).astype(np.float32)
            alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.4, 0.05, key="alpha_png")

            vis = overlay_rgba(base, mask.astype(np.uint8), alpha)
            st.image(vis, caption="Overlay", use_column_width=True)
            st.download_button("Download mask PNG", data=resp.content,
                               file_name="mask.png", mime="image/png")

with tab3:
    st.subheader("Upload NPZ (with key 'image' as (C,H,W,D))")
    npz_file = st.file_uploader("Choose a .npz file", type=["npz"])
    if npz_file is not None and st.button("Segment (NPZ)"):
        files = {"file": (npz_file.name, npz_file.getvalue(), "application/octet-stream")}
        resp = requests.post(API_URL, files=files, data={"modality":"npz", "return_format":"npz"})
        if resp.status_code != 200:
            st.error(resp.text)
        else:
            import numpy as np
            pred = np.load(io.BytesIO(resp.content))["pred"]  # (D,H,W)
            # 同时从上传文件里读取原图用于叠加
            img_npz = np.load(io.BytesIO(npz_file.getvalue()))
            img = img_npz["image"]  # (C,H,W,D)
            img = img.mean(0)       # -> (H,W,D)
            img = np.transpose(img, (2,0,1))  # -> (D,H,W)

            d = st.slider("Slice", 0, pred.shape[0]-1, pred.shape[0]//2, key="npz_slice")
            alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.4, 0.05, key="npz_alpha")
            vis = overlay_rgba(img[d].astype(np.float32), pred[d].astype(np.uint8), alpha)
            st.image(vis, use_column_width=True)
            st.download_button("Download pred.npz", data=resp.content,
                               file_name="pred.npz", mime="application/octet-stream")
