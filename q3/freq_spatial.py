from PIL import Image
import numpy as np
import os
path = input("Enter input image path: ")
img_gray = Image.open(path).convert("L")     # For frequency sampling
img_rgb  = Image.open(path).convert("RGB")   # For spatial sampling
print("Error: Cannot load image!")
gray = np.array(img_gray)
rgb  = np.array(img_rgb)
print("\nGrayscale Resolution :", gray.shape)     # (H, W)
print("RGB Resolution       :", rgb.shape)        # (H, W, 3)
os.makedirs("output_sampling", exist_ok=True)
# FREQUENCY SAMPLING  (uses 2D grayscale)
def freq_sample(im, f):
    F = np.fft.fftshift(np.fft.fft2(im))   # 2D FFT → center low freq
    H, W = F.shape
    # Low-frequency window size
    h2 = H // f
    w2 = W // f
    # Empty spectrum
    F_low = np.zeros_like(F)
    # Center crop coordinates
    hs = H//2 - h2//2
    he = H//2 + h2//2
    ws = W//2 - w2//2
    we = W//2 + w2//2
    # Keep only low frequencies
    F_low[hs:he, ws:we] = F[hs:he, ws:we]
    # Inverse FFT → blurred image
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(F_low)))
    # Normalize to 0–255
    out = (out - out.min()) / (out.max() - out.min())
    out = (out * 255).astype(np.uint8)
    return out
# SPATIAL SAMPLING  (uses RGB)
def spatial_sample(im, f):
    return im[::f, ::f]     # Downsample rows & columns
# Apply both methods for different sampling factors
factors = [2, 4, 8, 16]
for f in factors:
    # ------------ Frequency Sampling (Grayscale) ------------
    out_f = freq_sample(gray, f)
    save_f = f"output_sampling/freq_1_{f}.png"
    Image.fromarray(out_f).save(save_f)
    print(f"\nFrequency Sampling 1/{f}")
    print(" → Output Resolution:", out_f.shape)
    print("Saved:", save_f)
    # ------------ Spatial Sampling (RGB Color) ------------
    out_s = spatial_sample(rgb, f)
    save_s = f"output_sampling/spatial_1_{f}.png"
    Image.fromarray(out_s).save(save_s)
    print(f"Spatial Sampling 1/{f}")
    print(" → Output Resolution:", out_s.shape)
    print("Saved:", save_s)
print("\nAll images saved in: output_sampling/")
