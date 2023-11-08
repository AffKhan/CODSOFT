import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

#loading the iris dataset and displaying the first 10 rows
iris_data = pd.read_csv('TASK3\IRIS.csv')
print("First 10 rows of the Iris dataset:")
print(iris_data.head(10))

# Separating feats(X) and labels(y) 
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Test and train split (3 different random state scenarios are considered to check accuracy)
random_states = [42, 100, 2023]
for seed in random_states:
    print("Random_state: ", seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Create and train a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Making predictions on testing dataset
    y_pred = rf_classifier.predict(X_test)

    # Calculating and displaying accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Display a classification report and confusion matrix
    class_report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    print("Classification report: \n", class_report)
    print("Confusion Matrix: ", confusion)


    # Visualize feature importance
    feature_names = iris_data.columns[:-1]
    feature_importance = rf_classifier.feature_importances_
    sorted_indices = np.argsort(feature_importance)[::-1]

    # Data Visualization
    # Pairplot with hue
    sns.pairplot(iris_data, hue='species', markers=["o", "s", "D"])
    plt.suptitle("Pair Plot with Hue")
    plt.show()

    # Histograms for each feature
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(iris_data.columns[:-1]):
        plt.subplot(2, 2, i + 1)
        sns.histplot(iris_data, x=feature, hue='species', kde=True, element="step", common_norm=False)
        plt.title(f"Histogram of {feature}")
    plt.tight_layout()
    plt.show()

    #Creating a bar plot to visualize features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance[sorted_indices], y=feature_names[sorted_indices])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.xticks(rotation=45)
    plt.show()


    #Taking new data for further prediction
    new_data = np.array([
        [6.1, 3.1, 5.1, 1.4],  
        [5.7, 2.8, 4.1, 1.3],  
        [7.3, 2.9, 6.3, 1.8]   
    ])

    species_predictions = rf_classifier.predict(new_data)
    print("Predictions for New Data")
    for i, prediction in enumerate(species_predictions):
        print(f"Data {i+1}: {prediction}")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")