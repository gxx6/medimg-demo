# app/server.py
import io, os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import Literal

from app.utils_io import load_image_any, save_nifti_bytes, save_png_bytes, save_npz_bytes
from app.inference import load_model, predict_volume

app = FastAPI(title="MedImg Segmentation API")
AFFINE_FALLBACK = None  # 若无 affine，可用单位阵

@app.on_event("startup")
def _load():
    # 加载你的 .pth 模型；可通过环境变量 IN_CHANNELS/NUM_CLASSES/MODEL_PATH 配置
    load_model(os.getenv("MODEL_PATH", None))

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    modality: Literal["nifti","png","jpg","jpeg","npz"]="nifti",
    return_format: Literal["auto","nifti","png","npz"]="auto"
):
    try:
        data = await file.read()
        arr, meta = load_image_any(io.BytesIO(data), modality=modality)

        mask = predict_volume(arr, meta)  # (D,H,W) uint8

        if return_format == "auto":
            # NIfTI 输入 -> NIfTI；PNG/JPG 输入 -> PNG；NPZ 输入 -> NPZ
            if modality == "nifti":
                nii = save_nifti_bytes(mask, meta.get("affine"))
                return Response(nii, media_type="application/octet-stream",
                                headers={"Content-Disposition": 'attachment; filename="mask.nii.gz"'})
            elif modality in ("png","jpg","jpeg"):
                png = save_png_bytes(mask)
                return Response(png, media_type="image/png",
                                headers={"Content-Disposition": 'attachment; filename="mask.png"'})
            else:  # npz
                npz = save_npz_bytes(pred=mask.astype("uint8"))
                return Response(npz, media_type="application/octet-stream",
                                headers={"Content-Disposition": 'attachment; filename="pred.npz"'})
        elif return_format == "nifti":
            nii = save_nifti_bytes(mask, meta.get("affine"))
            return Response(nii, media_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="mask.nii.gz"'})
        elif return_format == "png":
            # 仅当 2D 才合适，这里简单处理第一张切片
            from numpy import squeeze
            png = save_png_bytes(mask[mask.shape[0]//2])
            return Response(png, media_type="image/png",
                            headers={"Content-Disposition": 'attachment; filename="mask.png"'})
        else:  # npz
            npz = save_npz_bytes(pred=mask.astype("uint8"))
            return Response(npz, media_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="pred.npz"'})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app",
                host="0.0.0.0",
                port=8000,
                reload=True)
