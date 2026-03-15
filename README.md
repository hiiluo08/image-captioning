# 🖼️ Image Captioning

Generate natural language descriptions for images using a **CNN + Transformer** architecture.

## 📋 Overview

This project implements an image captioning system that takes an image as input and generates a descriptive caption. It uses:

- **Encoder**: Pretrained CNN (ResNet-50) for extracting spatial image features
- **Decoder**: Transformer Decoder with multi-head cross-attention and causal masking
- **Dataset**: Flickr8k / Flickr30k

## 🏗️ Architecture

```
Image → CNN Encoder → Spatial Features [B, 49, d_model] → Transformer Decoder → Caption
                                                              ↑
                                          Caption Tokens + Positional Encoding
```

**Key components:**
- **CNN Encoder**: Extracts a grid of spatial features (e.g., 7×7 = 49 patches from ResNet-50)
- **Positional Encoding**: Sinusoidal encoding to inject sequence order information
- **Transformer Decoder**: Self-attention (causal) + Cross-attention over image features + Feed-forward
- **Causal Mask**: Prevents attending to future tokens during training & inference

## 📁 Project Structure

```
image-captioning/
├── configs/            # Hyperparameters & configuration
├── notebooks/          # EDA, Training, Evaluation notebooks
├── src/
│   ├── data/           # Dataset, Vocabulary, DataLoader
│   ├── models/         # Encoder, Transformer Decoder, Positional Encoding
│   ├── training/       # Training loop & checkpointing
│   ├── evaluation/     # Metrics (BLEU, METEOR, etc.)
│   └── inference/      # Prediction & beam search
├── data/               # Dataset files (gitignored)
├── checkpoints/        # Saved models (gitignored)
├── logs/               # Training logs (gitignored)
└── outputs/            # Inference results (gitignored)
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/image-captioning.git
cd image-captioning
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Download Dataset

Download the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it in the `data/` directory.

### 4. Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Train the Model

```bash
jupyter notebook notebooks/02_training.ipynb
# or
python -m src.training.trainer
```

### 6. Evaluate

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

### 7. Inference

```bash
python -m src.inference.predict --image path/to/image.jpg
```

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| BLEU-1/2/3/4 | N-gram precision |
| METEOR | Alignment-based metric |
| CIDEr | Consensus-based evaluation |

## 📚 References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)
- [PyTorch Image Captioning Tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

## 📄 License

This project is for educational purposes.
