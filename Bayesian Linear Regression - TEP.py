# Bayesian Linear Regression

# ----------------- Environment Setup -----------------
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pandas as pd

#from sklearn.model_selection import train_test_split
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Custom train_test_split function"""
    np.random.seed(random_state)
    n = len(X)
    n_test = int(n * test_size)
    indices = np.random.permutation(n)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

import warnings
warnings.filterwarnings('ignore')


# ----------------- Data Preparation -----------------
# Model the relationship in the TEP dataset

# Load the dataset
TEP = pd.read_csv(r'F:\UW2\Research\Python\d00_te.dat', sep=r'\s+', engine='python') # 959*52

# Select a feature and the target variable
X = TEP.iloc[:, 0]  # Feature: XEMAS 1 - A feed (stream 1)
y = TEP.iloc[:, 8]  # Target variable: XEMAS 9 - Reactor temperature

# Standardize the feature (mean=0, std=1) for better sampling performance
def standardize_data(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std

X_scaled = standardize_data(X)
y_scaled = standardize_data(y)

# Split into train/test sets (Optional) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.7, label='Test Data', color='orange')
plt.xlabel('A feed (stream 1)')   # XEMAS 1
plt.ylabel('Reactor temperature') # XEMAS 9
plt.title('Tennessee Eastman Process Data')
plt.legend()
plt.show()

# ----------------- Bayesian Linear Regression Model -----------------
# Define the Bayesian linear regression model using PyMC3
# Define probabilistic distributions for the model parameters (priors) and the likelihood of the observed data
with pm.Model() as bayesian_linear_model:
    
    # 1. Define Priors: Initial beliefs about model parameters
    # Intercept
    alpha = pm.Normal('alpha', mu=y_train.mean(), sigma=10)
    # Slope coefficient for 'RM'
    beta = pm.Normal('beta', mu=0, sigma=10)
    # Standard deviation of the noise (must be positive)
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    # 2. Define Linear Model: The deterministic relationship
    mu = alpha + beta * X_train
    
    # 3. Define Likelihood: Connects the model to the observed data
    # This describes how the data is distributed around the line (mu)
    likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
    
    # 4. Perform Inference: Sample from the posterior distribution
    # Use the NUTS sampler (efficient MCMC algorithm)
    trace = pm.sample(
        draws=2000,                # Number of samples to draw from the posterior
        tune=1000,                 # Number of tuning steps for the sampler
        chains=2,                  # Number of independent chains to run
        cores=2,                   # Number of CPU cores to use
        target_accept=0.95,        # Helps with convergence
        return_inferencedata=True  # Use modern ArviZ data structure
    )

# ----------------- Model Diagnostics -----------------
# Check if the sampling process was successful and the model converged to the true posterior distribution
# 1. Plot the traces for each parameter
az.plot_trace(trace)
plt.tight_layout()
plt.show()

# 2. View posterior summary statistics
summary = az.summary(trace, round_to=2)
print(summary)
# Check R-hat (~1.0 = good)
print("\nR-hat values:")
print(summary['r_hat'])
# Check effective sample size (high = good)
print("\nEffective sample sizes:")
print(summary['ess_bulk'])

# 3. Plot the energy landscape (good for identifying convergence issues)
az.plot_energy(trace)
plt.show()


# ----------------- Posterior Analysis and Visualization -----------------
# Analyze and interpret the results of the inference

# 1. Extract posterior samples for parameters
alpha_post = trace.posterior['alpha'].values.flatten()
beta_post = trace.posterior['beta'].values.flatten()

# 2. Plot the posterior distributions of parameters
az.plot_posterior(trace, var_names=['alpha', 'beta', 'sigma'], kind='kde')
plt.show()

# 3. Visualize the uncertainty in the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Observed Data')

# Plot a subset of regression lines from the posterior
for i in range(100): # Plot every 100th sample
    idx = np.random.randint(len(alpha_post))
    plt.plot(X_train, alpha_post[idx] + beta_post[idx] * X_train, 'r-', alpha=0.05)

# Plot the mean regression line
plt.plot(X_train, alpha_post.mean() + beta_post.mean() * X_train, 'k-', linewidth=2, label='Mean Posterior Regression Line')

plt.xlabel('A feed (stream 1)')   # XEMAS 1
plt.ylabel('Reactor temperature') # XEMAS 9
plt.title('Bayesian Linear Regression with Posterior Uncertainty')
plt.legend()
plt.show()

#------------------ Posterior Predictive Checking -----------------
# Check how well the model replicates the observed data
# Generate posterior predictive samples
with bayesian_linear_model:
    # Generate predictions on the training data
    post_pred = pm.sample_posterior_predictive(trace, predictions=True)

# Visualize the posterior predictive distribution
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(trace, data_pairs={"y_obs": "y_obs"}, ax=ax)
plt.show()

# Calculate and display prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y, alpha=0.7, label='Observed Data')

# Calculate and plot predictions for the test set
# We need to manually generate predictions for new data
X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 100)
y_pred = np.zeros((len(X_plot), 1000)) # Matrix to store predictions

for i, x in enumerate(X_plot):
    # For each x value, calculate y = alpha + beta*x for 1000 posterior samples
    y_pred[i] = alpha_post[:1000] + beta_post[:1000] * x

# Calculate mean and 94% Highest Density Interval (HDI)
mean_pred = y_pred.mean(axis=1)
hdi_data = az.hdi(np.array([y_pred])) # Calculate HDI

# Plot the final predictive uncertainty
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data', color='blue')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data', color='orange')
plt.plot(X_plot, mean_pred, 'r-', label='Mean Prediction')
plt.fill_between(X_plot, hdi_data[0, :, 0], hdi_data[0, :, 1], alpha=0.3, color='red', label='94% HDI')
plt.xlabel('A feed (stream 1)')   # XEMAS 1
plt.ylabel('Reactor temperature') # XEMAS 9
plt.title('Posterior Predictive Distribution with 94% HDI')
plt.legend()
plt.show()