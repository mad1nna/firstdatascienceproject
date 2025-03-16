# Titanic Survival Prediction

## Overview
This project analyzes the Titanic dataset to predict passenger survival using machine learning. The dataset includes information such as passenger class, sex, and the number of relatives on board. A Random Forest Classifier is used for prediction.

## Dataset
- Training Data: train.csv (Includes passenger details and survival status)
- Test Data: test.csv (Similar details but without survival status)

## Libraries Used

import numpy as np  # Numerical operations and array handling
import pandas as pd  # Data handling and analysis
import os  # Interacting with the operating system


## Loading the Data

# Listing dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Reading Titanic dataset
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


## Data Analysis
### Survival Rate by Gender

# Calculating survival rate for women
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
print("% of women who survived:", rate_women)

# Calculating survival rate for men
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% of men who survived:", rate_men)


## Machine Learning Model
### Preparing Data for Training

from sklearn.ensemble import RandomForestClassifier

# Selecting relevant features
features = ["Pclass", "Sex", "SibSp", "Parch"]
y = train_data["Survived"]
X = pd.get_dummies(train_data[features])  # Convert categorical variables
X_test = pd.get_dummies(test_data[features])


### Training the Model

# Initializing and training the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)


### Making Predictions

# Predicting survival on the test set
predictions = model.predict(X_test)

# Saving predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


## Key Steps in the Project
1. Data Loading: Read and explore the dataset.
2. Feature Engineering: Select important features and convert categorical data.
3. Model Training: Train a Random Forest model on the training data.
4. Prediction & Submission: Generate predictions and save them to a CSV file.

## Dependencies
Ensure you have the following installed:

pip install numpy pandas scikit-learn


## Conclusion
This project demonstrates how to use Random Forest Classifier for survival prediction on the Titanic dataset. By analyzing gender, passenger class, and family relationships, we can make informed predictions about survival rates.
