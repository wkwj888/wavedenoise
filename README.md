# WaveDenoise: An Efficient Diffusion Framework for Image Restoration
## Abstract
Diffusion models have recently emerged as leading approaches for image restoration, delivering superior perceptual fidelity and robustness compared with CNN- and GAN-based baselines. However, their deployment is often limited by heavy computational cost and memory usage. We introduce WaveDenoise, an efficient diffusion framework that operates in the wavelet domain to enable high-quality restoration under resource constraints. The pipeline first applies a discrete wavelet transform (DWT) to decompose an input image into low- and high-frequency subbands. A computationally efficient reversible diffusion network reconstructs low-frequency structures, while a lightweight U-Net with pixel shuffle recovers high-frequency details in parallel. This decoupled design reduces the spatial burden on the diffusion component, markedly lowering the activation footprint and accelerating inference without sacrificing quality. Extensive experiments on raindrop removal, image dehazing, and real-world denoising demonstrate state-of-the-art performance in PSNR, SSIM, and FID, alongside substantial improvements in runtime and memory efficiency. Taken together, WaveDenoise offers a scalable and practical recipe for deploying diffusion-based restoration models on limited hardware.
## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Testing

```bash
python test.py
```
