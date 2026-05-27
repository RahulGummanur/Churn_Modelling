# Churn Modelling with Artificial Neural Network

Predicts whether a bank customer will leave (churn) based on their profile, using a deep Artificial Neural Network built with TensorFlow/Keras.

## Problem Statement

Banks lose revenue when customers close their accounts. This model analyses customer attributes — such as geography, credit score, age, and account balance — to predict the probability of churn, enabling proactive retention strategies.

## Dataset

**File:** `Churn_Modelling.csv`

The dataset contains 10,000 bank customer records with the following key features:

| Feature | Description |
|---|---|
| Geography | Country of the customer (France, Germany, Spain) |
| Gender | Male / Female |
| CreditScore | Customer's credit score |
| Age | Age of the customer |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card (0/1) |
| IsActiveMember | Whether the customer is an active member (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | **Target** — 1 if the customer left, 0 if they stayed |

## Methodology

### 1. Data Preprocessing
- **Label Encoding** applied to the `Gender` column (Male → 1, Female → 0)
- **One-Hot Encoding** applied to the `Geography` column via `ColumnTransformer`
- **Train/Test Split** — 80% training, 20% testing (`random_state=1`)
- **Feature Scaling** using `StandardScaler` (fit on training set, applied to both)

### 2. ANN Architecture

```
Input Layer  →  12 features (after encoding)
Hidden Layer 1  →  6 neurons, ReLU activation
Hidden Layer 2  →  6 neurons, ReLU activation
Output Layer    →  1 neuron, Sigmoid activation (binary probability)
```

### 3. Training
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metric:** Accuracy
- **Batch Size:** 32
- **Epochs:** 100

### 4. Evaluation
- Predictions are thresholded at 0.5
- Performance evaluated via **Confusion Matrix** and **Accuracy Score**

## Results

The model achieves approximately **86% accuracy** on the test set.

A sample prediction is also included — for a 40-year-old male customer from France with a credit score of 600, balance of $60,000, and estimated salary of $50,000 — to demonstrate single-record inference.

## Requirements

```
tensorflow
scikit-learn
numpy
pandas
```

Install with:
```bash
pip install tensorflow scikit-learn numpy pandas
```

## Usage

```bash
python Churn_Modelling.py
```

Ensure `Churn_Modelling.csv` is in the same directory as the script.
