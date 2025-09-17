# SEENTENCE

*Transforming Images into Captions with AI*

![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-blue)

---

### Built with the tools and technologies:

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=nvidia&logoColor=white)
![Matplotl ib](https://img.shields.io/badge/-Matplotlib-11557C)
![Pandas](https://img.shields.io/badge/-Pandas-150458)
![NumPy](https://img.shields.io/badge/-NumPy-013243)
![tqdm](https://img.shields.io/badge/-tqdm-yellow)

![Pillow](https://img.shields.io/badge/-Pillow-blue)

---

## 📚Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Pre-requisites](#pre-requisites)
  - [Installation](#installation)

- [Usage](#usage)
  - [Create captions](#create-captions)
  - [Improve the model](#improve-the-model)

- [Contribute](#contribute)
  - [License](#license)

---

## 🧠 Overview

**Seentence** is a transformer-based neural network created to extract features from images and, through a context behind it, generate a coherent caption based on the vocabulary, combining advanced deep learning techniques.

### Why Sentence?

- 🦾 **Powerful Architecture**: Combines Transformers to Vision, Encoder and Decoder for best results
- 🎯 **End-to-End Workflow**: Supports training, inference, and model management with configurable hyperparameters.
- 📊 **Progress Visualization**: Tracks training metrics with integrated plotting tools for performance monitoring.
- 🛠️ **Custom Data Pipelines**: Facilitates preprocessing, tokenization, and batching tailored for image captioning tasks, allowing you to create your own vocabulary.

---

## 🚀 Getting Started

### 📦 Pre-requisites

- Python 3.8+
- PyTorch
- torchvision
- CUDA Toolkit (optional for GPU acceleration)
- numpy
- pandas
- tqdm
- matplotlib
- nltk
- Pillow

### 🔧 Installation

If you don't know what is it, use first option!

Without virtual environment:

```bash
git clone https://github.com/ErickRad/Seentence.git
cd Seentence
pip install -r requirements.txt

```
With virtual environment:

```bash
git clone https://github.com/ErickRad/Seentence.git
python -m venv Sentence
cd Seentence
source bin/activate
pip install -r requirements.txt

```

---

## 🧪 Usage 

### 🔠 Create captions for images

- Paste your image in the Seetence directory

```bash
python captionize.py --image-path <your_image.jpg> --num-caption 5 (Default: 1) 

```

### ⚡️ Improve the model

```bash
python train.py

```

---

## Contribute 

### 📃 License

- This software was built above BSD 3-Clause License, feel free to contribute and improve this project.

--- 