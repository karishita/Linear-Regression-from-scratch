# Linear Regression from Scratch using OLS and Gradient Descent

This project implements Linear Regression using two approaches:

- **Ordinary Least Squares (OLS)** using the Normal Equation (with pseudoinverse for non-invertible matrices)
- **Batch Gradient Descent (GD)** with manually computed gradients

The  results are compared with Scikit-learn’s  LinearRegression class on the **California Housing Dataset**.

---

##  Features

- No use of `sklearn.linear_model`
- Uses only `NumPy`, `Pandas`, and `Matplotlib`
- Implements loss tracking and visualization

---

## 📊 Dataset

- **California Housing Dataset**

---

## 🔢 Methods

### 1. Ordinary Least Squares (OLS)

OLS is implemented using the **Moore-Penrose Pseudoinverse** to handle cases where $( X^T X \)$ may be non-invertible:


$$\beta = (X^T X)^{-1} X^T y$$


### 2. Batch Gradient Descent

Weights and bias are updated using the following equations:

$$
\theta = \theta - \eta \cdot \nabla_\theta J(\theta)
$$

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 📈 Results

- R² Score (OLS & Scikit-learn): **0.619**
- R² Score (Gradient Descent): **0.614**
- Loss vs Epoch graph clearly shows convergence with 2000 epochs

---




## Future Work

- Add regularization (Ridge, Lasso)
- Extend to polynomial regression
- Mini-batch or stochastic gradient descent

---


