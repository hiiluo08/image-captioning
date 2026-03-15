# 🖼️ Image Captioning

Generate natural language descriptions for images using a deep learning **Encoder-Decoder** architecture.

## 📋 Overview

This project implements an image captioning system that takes an image as input and generates a descriptive caption. It uses:

- **Encoder**: Pretrained CNN (ResNet-50 / EfficientNet) for feature extraction
- **Decoder**: LSTM/GRU with optional Attention mechanism for caption generation
- **Dataset**: Flickr8k / Flickr30k

## 🏗️ Architecture

```
Image → CNN Encoder → Feature Vector → RNN Decoder → Caption
```

## 📁 Project Structure

```
image-captioning/
├── configs/            # Hyperparameters & configuration
├── notebooks/          # EDA, Training, Evaluation notebooks
├── src/
│   ├── data/           # Dataset, Vocabulary, DataLoader
│   ├── models/         # Encoder, Decoder, Attention
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

- [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)
- [Show and Tell (Vinyals et al., 2015)](https://arxiv.org/abs/1411.4555)
- [PyTorch Image Captioning Tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

## 📄 License

This project is for educational purposes.
