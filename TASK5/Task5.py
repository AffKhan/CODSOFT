import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Load the credit card data
credit_data = pd.read_csv("TASK5/creditcard.csv")

# Display the first few rows of the dataset
print(credit_data.head())

# Display information about the dataset
print(credit_data.info())

# Display summary statistics of the dataset
print(credit_data.describe())

# Display the last few rows of the dataset
print(credit_data.tail())

# Check for missing values
print(credit_data.isnull().sum())

# Display the distribution of the 'Class' column
class_distribution = credit_data['Class'].value_counts()
print(class_distribution)
print("`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````")
# Visualize the class distribution
class_distribution_percentage = (class_distribution / len(credit_data)) * 100
print(class_distribution_percentage)
class_distribution_percentage.plot.pie()

# Pairplot for selected features
#selected_features = ['V1', 'V2', 'V3', 'V4', 'V5']
#sns.pairplot(credit_data, vars=selected_features, hue='Class')
#plt.show()

# Boxplots for Amount and Time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x='Class', y='Amount', data=credit_data)
plt.title('Amount Distribution by Class')
plt.subplot(1, 2, 2)
sns.boxplot(x='Class', y='Time', data=credit_data)
plt.title('Time Distribution by Class')
plt.show()

# Violin plots for selected features
selected_features = ['V4', 'V12', 'V14']
plt.figure(figsize=(12, 6))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(1, 3, i)
    sns.violinplot(x='Class', y=feature, data=credit_data)
    plt.title(f'Violin Plot for {feature}')
plt.show()

# Calculate and print the percentage of normal and fraud transactions
normal_percentage = round(class_distribution_percentage[0], 2)
fraud_percentage = round(class_distribution_percentage[1], 2)
print("Percentage of normal transactions:", normal_percentage)
print("Percentage of fraud transactions:", fraud_percentage)

# Check the correlation between features
correlation_matrix = credit_data.corr()
print(correlation_matrix)

# Plot a heatmap for the correlation
plt.figure(figsize=(25, 18))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.show()

# Separate legitimate and fraud transactions
legitimate_transactions = credit_data[credit_data['Class'] == 0]
fraudulent_transactions = credit_data[credit_data['Class'] == 1]

# Display statistics for the 'Amount' column in both classes
print("Statistics for legitimate transactions:")
print(legitimate_transactions['Amount'].describe())
print("Statistics for fraudulent transactions:")
print(fraudulent_transactions['Amount'].describe())

# Display mean values for each feature in both classes
print("Mean values for each feature in legitimate transactions:")
print(legitimate_transactions.mean())
print("Mean values for each feature in fraudulent transactions:")
print(fraudulent_transactions.mean())

# Split the data into features (x) and target (y)
x = credit_data.drop('Class', axis=1)
y = credit_data['Class']

# Display the shape of x and y
print("Shape of features (x):", x.shape)
print("Shape of target (y):", y.shape)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3, stratify=y)

# Plot histograms for each feature in both classes
columns = list(x.columns)
plt.figure(figsize=(15, 30))
for i, column in enumerate(columns):
    plt.subplot(10, 3, i + 1)
    sns.histplot(legitimate_transactions[column], color='blue', kde=True, stat='density', label='Legitimate')
    sns.histplot(fraudulent_transactions[column], color='red', kde=True, stat='density', label='Fraudulent')
    plt.title(column, fontsize=12)
plt.legend()
plt.show()

# Create a Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict on the training data
y_pred_train = model.predict(x_train)

# Predict on the testing data
y_pred_test = model.predict(x_test)

# Evaluate the model using various metrics
train_accuracy = round(accuracy_score(y_pred_train, y_train) * 100, 2)
print('Accuracy on training data:', train_accuracy)

test_accuracy = round(accuracy_score(y_pred_test, y_test) * 100, 2)
print('Accuracy on testing data:', test_accuracy)

classification_rep = classification_report(y_pred_test, y_test)
print('Classification report:\n', classification_rep)

# Visualize the classification report as a heatmap
print(" Visualize the classification report as a heatmap:- ")
class_rep = classification_report(y_test, y_pred_test, output_dict=True)
class_rep_df = pd.DataFrame(class_rep).transpose()

plt.figure(figsize=(8, 4))
sns.heatmap(class_rep_df, annot=True, cmap='Blues', fmt=".2f")
plt.title('Classification Report Heatmap')
plt.show()