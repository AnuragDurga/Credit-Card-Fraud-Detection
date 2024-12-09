# Credit-Card-Fraud-Detection
Credit Card Fraud Detection
Project Description:
The primary goal of this project is to develop a robust machine learning model that can accurately identify fraudulent transactions in credit card data. Fraud detection is a critical issue for financial institutions, as fraudulent transactions can lead to significant financial losses and damage to customer trust. The dataset used in this project contains transactions made by credit cards in September 2013 by European cardholders, with a notable imbalance between the number of legitimate and fraudulent transactions.

Dataset Overview:
Source: The dataset is publicly available and contains 284,807 transactions, with only 492 (0.172%) labeled as fraudulent.
Features: The dataset includes 30 features, which are anonymized and scaled, along with a 'Class' label indicating whether a transaction is fraudulent (1) or legitimate (0).
Imbalance: The dataset is highly imbalanced, which poses challenges for training machine learning models, as they may become biased towards the majority class (legitimate transactions).
Steps Involved:
Data Loading and Exploration:

The project begins by importing necessary libraries such as pandas, numpy, seaborn, and matplotlib.
The dataset is loaded using pd.read_csv(), and initial exploration is performed using methods like head(), tail(), and info() to understand the structure and contents of the data.
The shape of the dataset is checked to confirm the number of rows and columns, and the presence of any missing values is assessed.
Data Preprocessing:

The 'Time' column is dropped as it does not contribute to the predictive power of the model.
Duplicate entries are checked using data.duplicated().any() and removed to ensure data integrity.
The class distribution is examined using value_counts() to understand the imbalance between normal and fraudulent transactions.
Data Visualization:

A count plot is created using seaborn to visualize the distribution of the 'Class' variable, highlighting the imbalance in the dataset. This visualization helps in understanding the extent of the problem and the need for techniques to handle class imbalance.
Feature and Target Variable Separation:

The features (X) are separated from the target variable (y), which indicates whether a transaction is fraudulent (1) or not (0). This separation is crucial for training the machine learning models.
Train-Test Split:

The dataset is split into training and testing sets using train_test_split() from sklearn, with 80% of the data used for training and 20% for testing. This ensures that the model can be evaluated on unseen data.
Model Training:

Two classifiers, Logistic Regression and Decision Tree Classifier, are implemented to train the model on the training data.
The models are evaluated using metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into the model's performance, especially in terms of its ability to correctly identify fraudulent transactions.
Handling Class Imbalance:

Undersampling: A balanced dataset is created by undersampling the majority class (normal transactions) to match the number of fraudulent transactions. This is done using the sample() method to randomly select a subset of normal transactions.
The models are retrained and evaluated on this new balanced dataset, and the performance metrics are compared to those obtained from the original imbalanced dataset.
Oversampling: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to create synthetic samples of the minority class (fraudulent transactions) to balance the dataset. This technique generates new instances of the minority class by interpolating between existing instances.
The models are again retrained and evaluated on the oversampled dataset, and the performance metrics are compared to those obtained from both the original and undersampled datasets.
Model Persistence:

The trained Decision Tree Classifier model is saved using joblib for future use. This allows the model to be reused without needing to retrain it, which is particularly useful in production environments.
Prediction:

The saved model is loaded, and predictions are made on new transaction data to classify them as either normal or fraudulent. The model takes in a feature vector representing a transaction and outputs a prediction based on the learned patterns.
Outcomes:
The project successfully demonstrates the ability to detect fraudulent transactions using machine learning techniques.
The evaluation metrics (accuracy, precision, recall, F1 score) provide insights into the model's performance, especially in handling imbalanced datasets. For instance, while accuracy may be high due to the majority class, precision and recall are more informative for assessing the model's effectiveness in identifying fraud.
The use of both undersampling and oversampling techniques shows improvements in model performance, particularly in recall, which is crucial for fraud detection.
