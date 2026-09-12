# GAN and VAE Practical Demo

## Overview

This project implements two deep learning practicals using TensorFlow and Keras:

1. **Generative Adversarial Network (GAN)** using the Fashion-MNIST dataset.
2. **Autoencoder (AE) and Variational Autoencoder (VAE)** using the MNIST dataset.

The notebook is designed to run in Google Colab and can use a GPU for faster training.

## Requirements

- Python 3
- TensorFlow 2.x (latest stable version recommended)
- NumPy
- Matplotlib

Install TensorFlow if required:

```bash
pip install --upgrade tensorflow
```

Check the installed version:

```python
import tensorflow as tf
print(tf.__version__)
```

For faster execution in Google Colab, select:

**Runtime → Change runtime type → T4 GPU**

## Part 1 — GAN

### Dataset

Fashion-MNIST is loaded using TensorFlow/Keras. The images are 28×28 grayscale images.

The images are normalized from `[0, 255]` to `[-1, 1]` so they match the Generator's final `Tanh` activation.

### Generator

The Generator:

- Starts with a random noise vector of dimension 100.
- Uses a Dense layer followed by reshaping.
- Uses transposed convolution layers to increase image size.
- Uses Batch Normalization and LeakyReLU.
- Uses `Tanh` as the final activation.
- Produces 28×28 grayscale images.

Before training, generated images are expected to look like random noise.

### Discriminator

The Discriminator:

- Receives 28×28 grayscale images.
- Uses convolution layers.
- Classifies real images as `1`.
- Classifies generated/fake images as `0`.
- Uses a final Sigmoid activation.

### Training

The GAN is trained for **25 epochs**.

- Loss function: Binary Cross-Entropy.
- Optimizer: Adam.
- Generator tries to make fake images classified as real.
- Discriminator learns to distinguish real and fake images.
- A fixed noise vector is used to compare generated images across epochs.

Generated 4×4 image grids are saved after:

- Epoch 5
- Epoch 10
- Epoch 15
- Epoch 20
- Epoch 25

Images are saved in the `gan_generated/` folder.

### GAN Outputs

The notebook produces:

- Original Fashion-MNIST image.
- Generator output before training.
- Generated images at epochs 5, 10, 15, 20, and 25.
- Generated images from a random saved epoch.
- Generator loss graph.
- Discriminator loss graph.

### GAN Questions

**How do the images change from epoch 5 to epoch 25?**

At epoch 5, images are mostly noisy and unclear. As training progresses, the Generator learns clothing patterns. By epoch 25, the images generally have more recognizable clothing structures and details.

**Can you observe instability during training?**

Yes. GAN training can be unstable because the Generator and Discriminator compete with each other. Their losses may fluctuate rather than decrease smoothly, and image quality may temporarily improve or worsen.

**At which epoch do cloth-like shapes become recognizable?**

Typically around epoch 10–15, although the exact epoch can vary depending on the training run.

**Does the GAN keep improving throughout training?**

Not necessarily. Improvement can slow down or fluctuate after a certain point because the Generator and Discriminator continuously adapt to each other.

## Part 2 — Autoencoder

The normal Autoencoder uses MNIST images.

### Autoencoder Architecture

The Autoencoder contains:

- Encoder
- 2-dimensional latent representation
- Decoder
- Sigmoid output activation

The Autoencoder is trained for **15 epochs** using Mean Squared Error (MSE).

### AE Outputs

The notebook generates:

- MNIST images before training.
- Autoencoder training results.
- A 2D latent-space visualization.
- Different colors representing digit classes.

## Part 3 — Variational Autoencoder

The VAE uses the MNIST dataset and a 2-dimensional latent space.

### VAE Architecture

The encoder produces:

- Mean (`μ`)
- Log variance (`log σ²`)

The latent vector is generated using the reparameterization trick:

`z = μ + σ × ε`

The decoder reconstructs the input image from the sampled latent vector.

### VAE Loss

The total VAE loss is:

`Total Loss = Reconstruction Loss + KL Divergence`

The notebook records:

- Reconstruction loss
- KL divergence loss
- Total loss

The VAE is trained for **20 epochs**.

### VAE Outputs

The notebook produces:

- Reconstruction-loss graph.
- KL-divergence-loss graph.
- Total-loss graph.
- Original and reconstructed MNIST images.
- Compact 2D VAE latent-space visualization.
- Latent-space plot with digit-class colors.
- Digits generated across the latent space.
- Images generated from extreme latent points.
- An image generated from a point outside the learned latent space.

## Expected Observations

### VAE Reconstruction

The reconstructed digits should resemble the original MNIST digits, although they may appear slightly blurry.

### VAE Latent Space

The KL-divergence term encourages the latent space to become compact and approximately Gaussian. Different digit classes can form distinguishable regions.

### Latent-Space Generation

Nearby latent points generally produce related digit styles. Moving across the latent space demonstrates the continuous nature of the VAE representation.

### Extreme and Outside Points

Extreme latent points can produce unusual or less realistic digits. A point far outside the learned latent distribution may generate a blurry or poorly formed image.

## How to Run

1. Open `GAN_VAE_.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select a GPU such as **T4 GPU** if available.
4. Run the notebook cells from top to bottom.
5. Allow the GAN to complete 25 epochs.
6. Allow the VAE to complete 20 epochs.
7. Review the generated images, latent-space plots, and loss graphs.
8. The GAN generated images will be stored in `gan_generated/`.

## Project Structure

```text
GAN_VAE/
│
├── GAN_VAE_.ipynb
├── README.md
│
└── gan_generated/
    ├── epoch_05.png
    ├── epoch_10.png
    ├── epoch_15.png
    ├── epoch_20.png
    └── epoch_25.png
```

## Note

GAN training is stochastic. Therefore, the exact generated images, loss values, and the epoch at which recognizable clothing shapes appear can differ between runs.
