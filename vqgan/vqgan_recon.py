#!/usr/bin/env python3
"""
Reconstruct images with VQGAN (Esser et al., "Taming Transformers") and
save side-by-side comparisons (Original | Reconstruction).

Install:
  pip install torch torchvision pillow matplotlib omegaconf
  pip install git+https://github.com/CompVis/taming-transformers.git

Example:
  python vqgan_reconstruct.py \
    --config /path/to/vqgan_imagenet_f16_16384.yaml \
    --ckpt   /path/to/vqgan_imagenet_f16_16384.ckpt \
    --input  /path/to/your_image.jpg \
    --outdir ./recons --size 256

Or a folder:
  python vqgan_reconstruct.py --config ...yaml --ckpt ...ckpt --input ./imgs --outdir ./recons --size 256
"""

import os
import argparse
from glob import glob
from typing import List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def _get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    mod = importlib.import_module(module)
    if reload:
        importlib.reload(mod)
    return getattr(mod, cls)

def instantiate_from_config(config):
    """
    Works with OmegaConf or plain dict configs that look like:
    {'target': 'taming.models.vqgan.VQModel', 'params': {...}}
    """
    try:
        # OmegaConf -> dict
        from omegaconf import OmegaConf
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass

    target = config.get("target", None)
    params = config.get("params", {})
    if target is None:
        raise KeyError("Expected key 'target' in config.")
    return _get_obj_from_str(target)(**params)
# --- end fallback ---

# ----------------------------- I/O helpers -----------------------------

def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        files = []
        for ex in exts:
            files.extend(glob(os.path.join(path, ex)))
        files.sort()
        return files
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Input path not found: {path}")


def pil_to_tensor(img: Image.Image, size: int = 256, center_crop: bool = True) -> torch.Tensor:
    """PIL -> torch float tensor in [-1,1], shape (1,3,H,W)."""
    img = img.convert("RGB")
    # if center_crop:
    #     w, h = img.size
    #     s = min(w, h)
    #     left = (w - s) // 2
    #     top = (h - s) // 2
    #     img = img.crop((left, top, left + s, top + s))
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                    # CHW
    x = torch.from_numpy(arr)[None, ...]            # NCHW
    x = x * 2.0 - 1.0                               # [-1,1]
    return x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Tensor in [-1,1], NCHW or CHW -> PIL RGB."""
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    x = (x * 255.0).round().byte().numpy()  # CHW uint8
    x = np.transpose(x, (1, 2, 0))          # HWC
    return Image.fromarray(x, mode="RGB")


def save_comparison(orig: Image.Image, rec: Image.Image, out_path: str, title: str = ""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(orig); ax1.set_title("Original"); ax1.axis("off")
    ax2.imshow(rec);  ax2.set_title("VQGAN Reconstruction"); ax2.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Model loading -----------------------------

def load_vqgan(config_path: str, ckpt_path: str, device: torch.device):
    """
    Load VQGAN (Esser et al.) from a .yaml config and .ckpt weights.
    Works with standard 'taming-transformers' checkpoints (e.g., imagenet_f16_16384).
    """
    

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_vqgan] Missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
    if unexpected:
        print(f"[load_vqgan] Unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def reconstruct(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encode -> quantize -> decode with VQGAN. Input x in [-1,1], NCHW.
    For VQModel, model.encode(x) returns (z_q, emb_loss, info).
    """
    x = x.to(device, non_blocking=True)
    enc_out = model.encode(x)
    if isinstance(enc_out, (tuple, list)):
        z_q = enc_out[0]
    else:
        z_q = enc_out  # fallback
    x_rec = model.decode(z_q)
    return x_rec


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Reconstruct images with VQGAN and save comparisons.")
    ap.add_argument("--config", required=True, help="Path to VQGAN .yaml (e.g., vqgan_imagenet_f16_16384.yaml)")
    ap.add_argument("--ckpt",   required=True, help="Path to VQGAN .ckpt")
    ap.add_argument("--input",  required=True, help="Image file or folder")
    ap.add_argument("--outdir", default="./recons", help="Where to save outputs")
    ap.add_argument("--size",   type=int, default=256, help="Square resize (typical for f16 models)")
    ap.add_argument("--no_center_crop", action="store_true", help="Disable center-crop before resize")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda or cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_vqgan(args.config, args.ckpt, device)

    paths = list_images(args.input)
    print(f"[VQGAN] Reconstructing {len(paths)} image(s) on {device}...")

    os.makedirs(args.outdir, exist_ok=True)

    for p in paths:
        img = Image.open(p)
        x = pil_to_tensor(img, size=args.size, center_crop=not args.no_center_crop)
        x_rec = reconstruct(model, x, device)

        pil_rec  = tensor_to_pil(x_rec)
        pil_orig = img.convert("RGB").resize((args.size, args.size), Image.LANCZOS)

        base = os.path.splitext(os.path.basename(p))[0]
        rec_path = os.path.join(args.outdir, f"{base}_recon.png")
        cmp_path = os.path.join(args.outdir, f"{base}_compare.png")

        pil_rec.save(rec_path)
        save_comparison(pil_orig, pil_rec, cmp_path, title=os.path.basename(p))

        print(f"Saved:\n  {rec_path}\n  {cmp_path}")


if __name__ == "__main__":
    main()
