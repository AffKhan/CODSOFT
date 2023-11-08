import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings


# Load the Titanic dataset 
data = pd.read_csv('TASK1/tested.csv')

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target variable
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier in a pipeline with imputation
rf_classifier = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Ignore the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', classification_rep)

# Sample passenger data for prediction 
sample_passenger_data = [
    [3, 25, 0, 0, 7.25, 1, 0, 1],  # Feature values for passenger 1
    [1, 35, 1, 0, 53.1, 0, 0, 0],  # Feature values for passenger 2
    [2, 22.5, 2, 1, 61.45, 0, 0, 0]  # Feature values for passenger 3
]

# Predict whether each passenger survived (1 for survived, 0 for not survived)
predictions = rf_classifier.predict(sample_passenger_data)

print("\nPredictions for New Passengers:")
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Passenger {i + 1}: Survived")
    else:
        print(f"Passenger {i + 1}: Did not survive")

