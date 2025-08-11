# Transformer-based Image Captioning Model

![Demo GIF or Image](<blockquote class="imgur-embed-pub" lang="en" data-id="a/Rk8h1Bj" data-context="false" ><a href="//imgur.com/a/Rk8h1Bj"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>)
*This project uses a pretrained EfficientNet CNN encoder and a Transformer decoder to generate descriptive captions for images.*

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Model Architecture](#model-architecture)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project is an implementation of an image captioning model that combines a Convolutional Neural Network (CNN) with a Transformer-based decoder. The goal is to take an image as input and generate a human-readable, descriptive caption. The CNN acts as an encoder to "understand" the contents of the image, and the Transformer decoder acts as a language model to generate the caption word by word.

This project was developed to explore the synergy between computer vision and natural language processing and to build a deeper understanding of attention mechanisms and sequence-to-sequence models.

---

## Features
- **CNN Encoder:** Uses a pretrained `EfficientNet-B0` to extract powerful image features.
- **Transformer Decoder:** Leverages a multi-head attention mechanism to generate context-aware captions.
- **Flickr8k Dataset:** Trained on the popular Flickr8k dataset.
- **Command-Line Interface:** Scripts for both training the model and running inference on new images.

---

## Installation

Follow these steps to set up the project environment. It is recommended to use a virtual environment.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/image-captioning-transformer.git](https://github.com/your-username/image-captioning-transformer.git)
cd image-captioning-transformer

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Please download the Flickr8k dataset and place the `Images` folder
# and `captions.txt` file inside a `data/` directory at the root of the project.
# You can find the dataset on Kaggle.
```

---

## Usage

### Training
To train the model from scratch, run the `train.py` script from the root directory.

```bash
python src/train.py --data_dir ./data --epochs 10 --batch_size 32
```
Model weights will be saved to `weights.pt` by default.

### Prediction
To generate a caption for a new image, use the `predict.py` script. You will need the vocabulary file (`vocab.pkl`) and the trained model weights.

```bash
python src/predict.py --image_path /path/to/your/image.jpg --weights_path weights.pt --vocab_path vocab.pkl
```

---

## Model Architecture

The model consists of two main components:
1.  **Encoder:** An `EfficientNet-B0` model, pretrained on ImageNet, processes the input image and outputs a single feature vector. This vector is then projected into the Transformer's expected dimension (`model_dim`).
2.  **Decoder:** A standard `nn.TransformerDecoder` takes the encoded image feature vector as its `memory` and the sequence of previously generated words as its `target` to predict the next word in the caption.

---

## Acknowledgements
- This project was built using PyTorch, timm, and Albumentations.
- The architecture is inspired by the original "Attention Is All You Need" paper.
