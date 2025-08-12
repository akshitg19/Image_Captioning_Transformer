# Transformer-based Image Captioning Model

![Project Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange)

This repository contains my implementation of an image captioning model that combines a pretrained CNN with a Transformer decoder. I built this project to get hands-on experience with classic multi-modal architectures and to explore the synergy between computer vision and natural language processing.

---

## My Goal for This Project

My main objective was to build a complete, end-to-end pipeline for a complex task. This involved everything from data processing and building a custom PyTorch Dataset to implementing the model architecture and training loop. It was a great way to solidify my understanding of attention mechanisms and how to effectively combine different types of neural networks.

---

## My Implementation Plan & Progress

Here's the roadmap I followed for this project. I'm currently working on refining the training and adding evaluation metrics.

- [x] **1. CNN Encoder:** Successfully integrated a pretrained `EfficientNet-B0` to extract powerful image features.
- [x] **2. Transformer Decoder:** Built the decoder architecture from scratch using PyTorch's `nn.TransformerDecoderLayer`.
- [x] **3. Data Pipeline:** Created a custom `Dataset` and `DataLoader` to efficiently process the Flickr8k dataset.
- [x] **4. Training Loop:** The initial training script is functional. My next step is to train the model to convergence and fine-tune hyperparameters.
- [x] **5. Inference Script:** Build a clean, easy-to-use script for generating captions on new images.
- [ ] **6. Evaluation:** Implement BLEU scores to quantitatively measure the quality of the generated captions.

---

## Installation

Follow these steps to set up the project environment. I recommend using a virtual environment.

### Step 1: Clone the Repository
```bash
# Clone this project to your local machine
git clone [https://github.com/your-username/image-captioning-transformer.git](https://github.com/your-username/image-captioning-transformer.git)
cd image-captioning-transformer
```

### Step 2: Set Up a Python Environment
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required dependencies
pip install -r requirements.txt
```

### Step 3: Download and Place the Dataset
This project requires the **Flickr8k dataset**.

1.  **Download the dataset from Kaggle:**
    * [**Link to Flickr8k Dataset on Kaggle**](https://www.kaggle.com/datasets/adityajn105/flickr8k)

2.  **Create a `data` folder** at the root of your project directory.

3.  **Unzip and organize the files** so that your directory structure looks like this:

    ```
    image-captioning-transformer/
    ├── data/
    │   ├── Images/
    │   │   └── ... (all .jpg files)
    │   └── captions.txt
    ├── src/
    │   └── ...
    └── README.md
    ```

---

## Usage

### Training
To train the model, run the `train.py` script from the root directory.

```bash
python src/train.py --data_dir ./data --epochs 10 --batch_size 32
```

### Prediction
Once the inference script is complete, you'll be able to run it like this:

```bash
# Note: This is the target usage once predict.py is implemented
python src/predict.py --image_path /path/to/your/image.jpg --weights_path weights.pt --vocab_path vocab.pkl
```

---

## Model Architecture

The model consists of two main components:
1.  **Encoder:** An `EfficientNet-B0` model, pretrained on ImageNet, processes the input image and outputs a single feature vector.
2.  **Decoder:** A standard `nn.TransformerDecoder` takes the encoded image feature vector as its `memory` and the sequence of previously generated words to predict the next word in the caption.

---

## Acknowledgements
- This project was built using PyTorch, timm, and Albumentations.
- The architecture is inspired by the original "Attention Is All You Need" paper.
