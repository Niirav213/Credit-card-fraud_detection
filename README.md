# Credit Card Fraud Detection

An educational credit-card fraud detection project built from scratch with a
small automatic-differentiation engine and multilayer perceptron. The notebooks
train an autoencoder on normal transactions and use reconstruction error,
optionally combined with proximity scores, to identify potential fraud.

## Project structure

- [`NeuralNet.py`](NeuralNet.py) - scalar automatic differentiation primitives
  (`Value`) and a simple `Neuron`/`Layer`/`MLP` implementation.
- [`test file.ipynb`](test%20file.ipynb) - autoencoder training and
  reconstruction-error-based detection.
- [`test file (1).ipynb`](test%20file%20(1).ipynb) - extended workflow that
  also evaluates proximity-based scores.
- [`NeuralNet.ipynb`](NeuralNet.ipynb) - notebook version of the neural-network
  implementation.

## Dataset

The dataset is not included because of its size. Download
[`creditcard.csv`](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle
and place it in the repository root.

## Setup

Python 3.10+ is recommended. Install the notebook dependencies:

```bash
python -m pip install numpy pandas scikit-learn jupyter
```

## Run

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open `test file (1).ipynb` and run the cells in order. The notebook expects
`creditcard.csv` in the same directory and reports classification metrics after
calculating reconstruction and proximity scores.

## Notes

This project is intended for learning and experimentation. The model and
thresholds have not been validated for production fraud detection.
