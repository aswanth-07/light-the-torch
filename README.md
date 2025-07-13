# 🚀 My PyTorch Learning Journey

Welcome to my PyTorch learning repository! This is my personal notebook vault as I work through the fantastic **[Zero to Mastery: Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)** course.

The main goal?  
To organize everything I learn — from basic tensor operations to building neural networks — in one place. It’s my go-to hub for revisiting concepts, tracking progress, and sharing what I’ve built.

---

## 📁 Repository Structure

Here's how this repo is structured:

- `datasets/` – Datasets used throughout the course (or links to where to get them).
- `models/` – PyTorch model files or saved model checkpoints.
- `notebooks/` – The core learning notebooks (Jupyter format).
- `requirements.txt` – All the Python packages needed to run the notebooks.
- `README.md` – You’re reading it right now!

---

## 📓 Notebooks Overview

| Notebook File Name              | Topic / Module Covered                        | Key Concepts Demonstrated                                                                                   | Status       |
| ------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------ |
| [`00 - PyTorch Fundamentals.ipynb`](notebooks/00%20-%20PyTorch%20Fundamentals.ipynb) | Intro to PyTorch Tensors                   | Tensor creation & manipulation (indexing, reshaping, viewing, stacking, squeezing, unsqueezing, permuting), common operations, datatypes, NumPy bridge, GPU acceleration, reproducibility                 | ✅ Completed |
| [`01 - PyTorch Workflow.ipynb`](notebooks/01%20-%20PyTorch%20Workflow.ipynb)        | End-to-End Linear Regression               | Custom `nn.Module`, device-agnostic code, training loop, loss/optimizer, predictions, save/load model       | ✅ Completed |
| [`01.1 - Practice PyTorch Workflow.ipynb`](notebooks/01.1%20-%20Practice%20PyTorch%20Workflow.ipynb) | Practice: End-to-End Linear Regression | Data loading (CSV) & preprocessing (StandardScaler), custom `nn.Module` for regression, train/test split, full training/evaluation loop (L1Loss, SGD), model persistence (save/load), R2 score for evaluation | ✅ Completed |
| [`02 - PyTorch Classification.ipynb`](notebooks/02%20-%20PyTorch%20Classification.ipynb) | Binary & Multi-class Classification      | `make_circles` dataset, custom `nn.Module` for binary classification, `nn.ReLU`, `nn.BCEWithLogitsLoss`, `SGD`, decision boundary plotting, multi-class classification with custom dataset, data preprocessing (missing values), `nn.CrossEntropyLoss`, `Adam`, `torch.argmax` for multi-class predictions | ✅ Completed |
| [`03 - PyTorch CNN and CV.ipynb`](notebooks/03%20-%20PyTorch%20CNN%20and%20CV.ipynb) | Computer Vision & CNNs | `torchvision.datasets` (FashionMNIST), `DataLoader`, custom CNN (`nn.Module`, `nn.Conv2d`, `nn.MaxPool2d`), training/evaluation loops, model comparison (MLP vs. CNN), confusion matrix, saving models. | ✅ Completed |

> ✨ *I'll keep this table updated as I work through new notebooks and modules!*

---

## 🛠️ Setup & Usage

Want to try these notebooks locally? Here's how:

1. **Clone this repo:**
   ```bash
   git clone https://github.com/aswanth-07/pytorch-learning-journey.git
   cd pytorch-learning-journey
   ```

2. **(Optional but recommended) Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   > First, make sure you've renamed `requirements.txt.txt` to `requirements.txt` if needed.
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook or Lab:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

5. Head into the `notebooks/` folder and open the notebook you want!

---

## 🎓 Course Details

- **Course Name:** Zero to Mastery: Learn PyTorch for Deep Learning
- **Instructor:** Daniel Bourke
- **Platform:** [learnpytorch.io](https://www.learnpytorch.io/)
- **Goal:** Master deep learning fundamentals with PyTorch and build strong foundations for CV and generative models.

---

## ✍️ Notes & Reflections

Here’s where I’ll occasionally jot down thoughts or things that stood out:

- PyTorch’s dynamic nature makes experimentation feel intuitive and clean.
- The `.to(device)` concept (CPU/GPU flexibility) was a game-changer for writing portable code.
- Loved the visual explanations — they really helped me internalize tensor broadcasting and model flow.
- So far, the course structure has been solid — practical and beginner-friendly, with just the right touch of theory.
- The `torch.inference_mode()` context manager is a clean way to ensure no gradients are calculated during evaluation or testing.
- Grasping the standard training loop (forward pass, loss calculation, zero gradients, backward pass, optimizer step) is fundamental for building and training models.
- PyTorch's `nn.Module` and `nn.Parameter` provide a powerful and flexible way to define custom models and their learnable parameters.
- Saving and loading model states with `torch.save()` and `load_state_dict()` is straightforward for model persistence and sharing.
- Understanding the differences between binary and multi-class classification is key:
    - Output layer units (1 for binary, N classes for multi-class).
    - Loss functions (`nn.BCEWithLogitsLoss` for binary, `nn.CrossEntropyLoss` for multi-class).
    - Activation functions (e.g., `nn.Sigmoid` often used with binary, `nn.Softmax` implicitly with `CrossEntropyLoss` for multi-class).
- Adding non-linear activation functions like `nn.ReLU` is crucial for models to learn complex patterns.
- Visualizing decision boundaries is a great way to understand how a classification model is learning.
- Real-world datasets often require preprocessing, such as handling missing values, before model training.
- For multi-class predictions, `torch.argmax` is used to find the class with the highest probability.
- `DataLoader` is essential for batching data, which improves computational efficiency and provides more frequent gradient updates.
- `torchvision` is the go-to library for computer vision in PyTorch, offering datasets, models, and transforms.
- The output dimensions of a `Conv2d` layer can be calculated with the formula: `output = floor((input_size - kernel_size + 2 * padding) / stride) + 1`. This is crucial for designing CNN architectures.
- Building custom CNNs involves stacking `nn.Conv2d` and `nn.MaxPool2d` layers to learn spatial hierarchies from image data.
- A confusion matrix is a powerful tool for evaluating classification models, offering a more detailed breakdown of performance than accuracy alone.

---

Happy learning and model-building! 🧠⚡  
If you're reading this and also learning PyTorch, feel free to fork this repo or get inspired by the structure.

— **Aswanth**