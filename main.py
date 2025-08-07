import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt

# Load the dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Data shape: {X.shape}")

# ===================================================================
#                      EDA SECTION (Add this to your code)
# ===================================================================
import pandas as pd
import seaborn as sns # A great library for beautiful visualizations

# --- Step 0: Create a Pandas DataFrame for easier analysis ---
# We use cancer.feature_names to give our columns meaningful names
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y # Add the target column (0: Malignant, 1: Benign)

print("\n--- Starting Exploratory Data Analysis (EDA) ---")


# --- Step 1: Get Basic Information ---
print("\n[EDA] 1. Basic DataFrame Info:")
df.info()

# --- Interpretation of .info() ---
# This tells us:
# - There are 569 entries (samples).
# - There are 31 columns (30 features + 1 target).
# - CRITICAL: All 30 feature columns have 569 non-null values. This is great news! It means there are NO MISSING VALUES.
# - All feature columns are numeric (float64), which is what our model needs.


# --- Step 2: Get Descriptive Statistics ---
print("\n[EDA] 2. Descriptive Statistics:")
print(df.describe())

# --- Interpretation of .describe() ---
# Look at the 'mean', 'std' (standard deviation), 'min', and 'max' rows.
# You'll immediately notice that the features have VASTLY different scales.
# For example:
# - 'mean radius' has a mean of ~14.
# - 'mean area' has a mean of ~654.
# - 'area error' has a mean of ~40.
# CONCLUSION: This is the most important finding. It confirms that using StandardScaler is not just a good idea, it's ESSENTIAL. Without it, features with larger values (like 'mean area') would dominate the learning process.


# --- Step 3: Check the Target Variable Balance ---
print("\n[EDA] 3. Target Variable Distribution:")
print(df['target'].value_counts())

# --- Interpretation of .value_counts() ---
# This shows how many samples belong to each class.
# - 1: 357 (Benign)
# - 0: 212 (Malignant)
# CONCLUSION: The dataset is not perfectly 50/50, but it is reasonably balanced. We don't have a situation where one class has 99% of the samples. This means we can proceed without needing special techniques to handle class imbalance.


# --- Step 4: Visualize Feature Distributions with Histograms ---
print("\n[EDA] 4. Plotting feature distributions (histograms)...")
plt.figure(figsize=(20, 15))
df.hist(bins=20, figsize=(20, 15), layout=(6, 5))
plt.suptitle("Histograms of All Features")
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
plt.show()

# --- Interpretation of Histograms ---
# This gives you a feel for the shape of each feature's data.
# - Some features like 'mean radius' look somewhat "normal" (a bell-curve shape).
# - Others like 'area error' are heavily skewed to the right.
# This visual confirmation is a standard part of any good analysis.


# --- Step 5: Visualize Correlations with a Heatmap ---
# This is a bit more advanced but incredibly useful.
print("\n[EDA] 5. Plotting a correlation heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm') # annot=True is too crowded
plt.title('Correlation Matrix of Features')
plt.show()

# --- Interpretation of the Heatmap ---
# This map shows how features relate to each other.
# - Bright Red: Strong positive correlation (when one feature goes up, the other tends to go up).
# - Bright Blue: Strong negative correlation.
# - White/Pale colors: Low correlation.
# OBSERVATION: You will see large red squares. For example, 'mean radius', 'mean perimeter', and 'mean area' are all very highly correlated. This makes sense! A larger radius leads to a larger perimeter and area. This indicates "multicollinearity", where features provide redundant information. For this assignment, we don't need to act on it, but in a real project, you might use it for feature selection.


print("\n--- EDA Complete ---")
# ===================================================================
#                  END OF EDA SECTION
# ===================================================================# ===================================================================
#                      EDA SECTION (Add this to your code)
# ===================================================================
import pandas as pd
import seaborn as sns # A great library for beautiful visualizations

# --- Step 0: Create a Pandas DataFrame for easier analysis ---
# We use cancer.feature_names to give our columns meaningful names
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y # Add the target column (0: Malignant, 1: Benign)

print("\n--- Starting Exploratory Data Analysis (EDA) ---")


# --- Step 1: Get Basic Information ---
print("\n[EDA] 1. Basic DataFrame Info:")
df.info()

# --- Interpretation of .info() ---
# This tells us:
# - There are 569 entries (samples).
# - There are 31 columns (30 features + 1 target).
# - CRITICAL: All 30 feature columns have 569 non-null values. This is great news! It means there are NO MISSING VALUES.
# - All feature columns are numeric (float64), which is what our model needs.


# --- Step 2: Get Descriptive Statistics ---
print("\n[EDA] 2. Descriptive Statistics:")
print(df.describe())

# --- Interpretation of .describe() ---
# Look at the 'mean', 'std' (standard deviation), 'min', and 'max' rows.
# You'll immediately notice that the features have VASTLY different scales.
# For example:
# - 'mean radius' has a mean of ~14.
# - 'mean area' has a mean of ~654.
# - 'area error' has a mean of ~40.
# CONCLUSION: This is the most important finding. It confirms that using StandardScaler is not just a good idea, it's ESSENTIAL. Without it, features with larger values (like 'mean area') would dominate the learning process.


# --- Step 3: Check the Target Variable Balance ---
print("\n[EDA] 3. Target Variable Distribution:")
print(df['target'].value_counts())

# --- Interpretation of .value_counts() ---
# This shows how many samples belong to each class.
# - 1: 357 (Benign)
# - 0: 212 (Malignant)
# CONCLUSION: The dataset is not perfectly 50/50, but it is reasonably balanced. We don't have a situation where one class has 99% of the samples. This means we can proceed without needing special techniques to handle class imbalance.


# --- Step 4: Visualize Feature Distributions with Histograms ---
print("\n[EDA] 4. Plotting feature distributions (histograms)...")
plt.figure(figsize=(20, 15))
df.hist(bins=20, figsize=(20, 15), layout=(6, 5))
plt.suptitle("Histograms of All Features")
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
plt.show()

# --- Interpretation of Histograms ---
# This gives you a feel for the shape of each feature's data.
# - Some features like 'mean radius' look somewhat "normal" (a bell-curve shape).
# - Others like 'area error' are heavily skewed to the right.
# This visual confirmation is a standard part of any good analysis.


# --- Step 5: Visualize Correlations with a Heatmap ---
# This is a bit more advanced but incredibly useful.
print("\n[EDA] 5. Plotting a correlation heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm') # annot=True is too crowded
plt.title('Correlation Matrix of Features')
plt.show()

# --- Interpretation of the Heatmap ---
# This map shows how features relate to each other.
# - Bright Red: Strong positive correlation (when one feature goes up, the other tends to go up).
# - Bright Blue: Strong negative correlation.
# - White/Pale colors: Low correlation.
# OBSERVATION: You will see large red squares. For example, 'mean radius', 'mean perimeter', and 'mean area' are all very highly correlated. This makes sense! A larger radius leads to a larger perimeter and area. This indicates "multicollinearity", where features provide redundant information. For this assignment, we don't need to act on it, but in a real project, you might use it for feature selection.


print("\n--- EDA Complete ---")
# ===================================================================
#                  END OF EDA SECTION
# ===================================================================

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
# IMPORTANT: Fit the scaler ONLY on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Task 1: Baseline Model ---
print("\n--- Building and Evaluating Baseline Model ---")

# Build the model architecture
baseline_model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)), # Input layer
    keras.layers.Dense(64, activation='relu'),            # Single hidden layer
    keras.layers.Dense(1, activation='sigmoid')            # Output layer for binary classification
])

# Compile the model
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Train the model
history = baseline_model.fit(
    X_train_scaled,
    y_train,
    epochs=20, # We'll use 20 epochs for quick training
    validation_split=0.2, # Use part of the training data for validation
    verbose=0 # Set to 1 if you want to see epoch-by-epoch progress
)

# Evaluate on the test set
loss, baseline_accuracy = baseline_model.evaluate(X_test_scaled, y_test)
print(f"Baseline Model Test Accuracy: {baseline_accuracy:.4f}")

# Store results for final comparison
results = {
    'Baseline': {'accuracy': baseline_accuracy, 'time': None}
}

# --- Task 2: Grid Search ---
print("\n--- Performing Grid Search ---")

# Function to create a Keras model.
# Arguments (neurons, learning_rate) will be passed by GridSearchCV.
def create_model(neurons=64, learning_rate=0.01):
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(units=neurons, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model so it can be used by scikit-learn
keras_clf = KerasClassifier(build_fn=create_model, epochs=20, verbose=0)

# Define the grid of hyperparameters to search
param_grid = {
    'batch_size': [32, 64],
    'model__neurons': [32, 64, 128],
    'model__learning_rate': [0.001, 0.01]
}

# Create the Grid Search object
grid_search = GridSearchCV(estimator=keras_clf, param_grid=param_grid, cv=3, n_jobs=-1) # n_jobs=-1 uses all available cores

# Record start time
start_time = time.time()

# Run the search
grid_search.fit(X_train_scaled, y_train)

# Calculate and record total time
grid_search_time = time.time() - start_time

print(f"Grid Search completed in {grid_search_time:.2f} seconds")
print(f"Best Score (Accuracy): {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Store results for final comparison
results['Grid Search'] = {
    'accuracy': grid_search.best_score_,
    'time': grid_search_time
}

# --- Task 3: Random Search ---
print("\n--- Performing Random Search ---")

# Define the parameter space for Random Search
# Note the wider, more continuous ranges
param_dist = {
    'batch_size': [16, 32, 64, 128],
    'model__neurons': np.arange(16, 257, 16), # Sample from 16, 32, ..., 256
    'model__learning_rate': np.logspace(-4, -1, 100) # Uniformly sample from 1e-4 to 1e-1 on a log scale
}

# Create the Random Search object
# n_iter=10 means it will try 10 random combinations
random_search = RandomizedSearchCV(
    estimator=keras_clf,
    param_distributions=param_dist,
    n_iter=10, # Number of random combinations to try
    cv=3,
    n_jobs=-1,
    random_state=42
)

# Record start time
start_time = time.time()

# Run the search
random_search.fit(X_train_scaled, y_train)

# Calculate and record total time
random_search_time = time.time() - start_time

print(f"Random Search completed in {random_search_time:.2f} seconds")
print(f"Best Score (Accuracy): {random_search.best_score_:.4f}")
print(f"Best Parameters: {random_search.best_params_}")

# Store results for final comparison
results['Random Search'] = {
    'accuracy': random_search.best_score_,
    'time': random_search_time
}

# --- Task 4: Compare and Conclude ---
print("\n--- Final Comparison ---")

# Prepare data for plotting
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
times = [results[name]['time'] for name in names if name != 'Baseline'] # Time not applicable for baseline

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Accuracy Comparison
ax1.bar(names, accuracies, color=['gray', 'blue', 'green'])
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylabel('Best Cross-Validation Accuracy')
ax1.set_ylim([min(accuracies) - 0.02, 1.0]) # Adjust y-axis for better visibility

# Plot Time Comparison
ax2.bar(['Grid Search', 'Random Search'], times, color=['blue', 'green'])
ax2.set_title('Tuning Time Comparison')
ax2.set_ylabel('Time (seconds)')

plt.suptitle('Grid Search vs. Random Search Performance')
plt.show()