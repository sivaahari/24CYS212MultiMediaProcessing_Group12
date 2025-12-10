import numpy as np
from PIL import Image
import os

# Input images
low_img = np.array(Image.open("low_light.jpg").convert("L"))
brt_img = np.array(Image.open("bright_light.jpg").convert("L"))

out = "bitplane_results/"
os.makedirs(out, exist_ok=True)

# --- Function: extract 8 bitplanes (0–7) ---
def bitplanes(img):
    return [(img >> b) & 1 for b in range(8)]   # returns list of 8 planes (0/1)

# --- Function: reconstruct using lowest 3 bits ---
def reconstruct_lsb3(img):
    val = img & 7                # keep bits 0,1,2  (0–7)
    return (val * 255 // 7).astype(np.uint8)   # scale to 0–255

# --- Save bitplanes ---
def save_planes(img, prefix):
    planes = bitplanes(img)
    for i, p in enumerate(planes):
        Image.fromarray((p * 255).astype(np.uint8)).save(f"{out}/{prefix}_bit{i}.png")
    return planes

# ---------------- LOW LIGHT IMAGE ----------------
save_planes(low_img, "low")
low_rec = reconstruct_lsb3(low_img)
low_diff = np.abs(low_img - low_rec).astype(np.uint8)

Image.fromarray(low_rec).save(f"{out}/low_reconstructed.png")
Image.fromarray(low_diff).save(f"{out}/low_diff.png")

# ---------------- BRIGHT LIGHT IMAGE -------------
save_planes(brt_img, "bright")
brt_rec = reconstruct_lsb3(brt_img)
brt_diff = np.abs(brt_img - brt_rec).astype(np.uint8)

Image.fromarray(brt_rec).save(f"{out}/bright_reconstructed.png")
Image.fromarray(brt_diff).save(f"{out}/bright_diff.png")

print("All outputs saved in:", out)
