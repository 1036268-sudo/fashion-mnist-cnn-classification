# Fashion-MNIST Image Classification with a CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1036268-sudo/fashion-mnist-cnn-classification/blob/main/fashion_mnist_cnn.ipynb)

This project implements and evaluates a convolutional neural network (CNN) for classifying Fashion-MNIST images into 10 clothing categories. It was prepared for **Milestone 2: AI for Non-tabular Data | Deep Learning**.

## Project objectives

- Source a public image dataset.
- Prepare normalized training, validation, and test sets.
- Develop a CNN with TensorFlow/Keras.
- Monitor training and validation performance.
- Evaluate the final model with accuracy, a confusion matrix, a classification report, and example predictions.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images, each 28 × 28 pixels, across 10 balanced classes. The official 60,000-image training set is split into 50,000 training images and 10,000 validation images. The official 10,000-image test set is kept untouched for final evaluation.

| Split | Images | Purpose |
|---|---:|---|
| Training | 50,000 | Learn model parameters |
| Validation | 10,000 | Monitor generalisation and control training |
| Test | 10,000 | Final unbiased evaluation |

Pixel values are converted from integers in `[0, 255]` to `float32` values in `[0, 1]`, and a channel dimension is added to produce tensors with shape `(28, 28, 1)`.

## CNN architecture

The network uses two convolutional blocks with batch normalisation, max pooling, and dropout, followed by a dense classification head:

1. Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
2. Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
3. Flatten → Dense(128) → Dropout → Dense(10, softmax)

The model is trained with Adam and sparse categorical cross-entropy. Early stopping restores the best validation weights, and ReduceLROnPlateau lowers the learning rate when validation loss stalls.

## Run in Google Colab

Open `fashion_mnist_cnn.ipynb` in Google Colab, select **Runtime → Run all**, and wait for training and evaluation to finish. A GPU is optional because Fashion-MNIST is small.

The notebook produces:

- class and split inspection;
- sample images;
- model summary;
- training/validation loss and accuracy curves;
- test accuracy and loss;
- confusion matrix;
- precision, recall, F1-score, and support for every class;
- example predictions; and
- a saved `fashion_mnist_cnn.keras` model.

## Expected result

With the fixed random seed and default settings, the CNN should usually achieve about **90%–93% test accuracy**. Exact results can vary slightly across hardware and TensorFlow versions. The notebook prints the actual result from the current run; this README intentionally does not claim an unexecuted score.

## Local execution

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Generated plots, reports, and the saved model are written to `artifacts/`.

## Repository structure

```text
fashion-mnist-cnn/
├── fashion_mnist_cnn.ipynb  # Colab-ready documented implementation
├── train.py                 # Reproducible command-line implementation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore
```

## Limitations and possible improvements

- Fashion-MNIST is low-resolution and grayscale, so results do not directly represent performance on complex real-world photographs.
- Shirts, T-shirts/tops, coats, and pullovers are visually similar and are commonly confused.
- Future work could include data augmentation, hyperparameter tuning, cross-validation, transfer learning on a colour-image dataset, and explainability methods such as Grad-CAM.

## Reproducibility and responsible use

Random seeds are fixed where practical. Small numerical differences may still occur between devices. Fashion-MNIST contains product images rather than people, reducing privacy risk, but the model should still be tested on data representative of its intended use before deployment.

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
- TensorFlow. *Convolutional Neural Network (CNN)* and Keras API documentation.
- scikit-learn. *Classification metrics* documentation.
