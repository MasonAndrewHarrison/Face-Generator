# Face Generator

A deep learning project that uses a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to generate realistic human face images.

The model is trained on a dataset of facial images and learns the underlying distribution of human faces, allowing it to create entirely new, synthetic faces that do not correspond to real individuals.

## Features

- WGAN-GP implementation for stable GAN training
- Automatic dataset download and preprocessing
- GPU (CUDA) and CPU support
- Generates realistic human face images
- Built with PyTorch


## Generated Face Examples

<table>
  <tr>
    <td><img src="images/img1.png" alt="Generated Face 1" width="150"/></td>
    <td><img src="images/img2.png" alt="Generated Face 2" width="150"/></td>
    <td><img src="images/img3.png" alt="Generated Face 3" width="150"/></td>
    <td><img src="images/img4.png" alt="Generated Face 4" width="150"/></td>
    <td><img src="images/img5.png" alt="Generated Face 5" width="150"/></td>
    <td><img src="images/img6.png" alt="Generated Face 6" width="150"/></td>
  </tr>
</table>

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MasonAndrewHarrison/Face-Generator.git
cd Face-Generator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Linux / macOS

```bash
source venv/bin/activate
```

#### Windows Command Prompt

```cmd
venv\Scripts\activate.bat
```

#### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install PyTorch

#### CUDA (NVIDIA GPU)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### CPU Only

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the Dataset

```bash
python create_dataset.py
```

### 7. Train the Model

```bash
python main.py
```


## How It Works

1. A random latent vector is sampled from a noise distribution.
2. The generator transforms this noise into a synthetic face image.
3. The critic (discriminator) evaluates how realistic the image appears.
4. Gradient Penalty (GP) is used to enforce the Lipschitz constraint, improving training stability.
5. Through adversarial training, the generator gradually learns to produce increasingly realistic faces.

