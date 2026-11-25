# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required Python libraries.
2.Load the student placement dataset and separate the input and output variables.
3.Split the dataset into training and testing data, then train the Logistic Regression model.
4.Predict the placement status and evaluate the model accuracy.
## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'CGPA': [6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
             6.0, 7.2, 8.2, 9.2],
    'IQ': [90, 100, 110, 120, 130, 140,
           85, 105, 125, 145],
    'Placement': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['CGPA', 'IQ']]
y = df['Placement']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict placement
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predicted Placement Status:", y_pred)
print("Actual Placement Status:", y_test.values)
print("Model Accuracy:", accuracy * 100, "%")

# Predict for a new student
new_student = pd.DataFrame([[8.5, 130]], columns=['CGPA', 'IQ'])

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")
```

## Output:
<img width="720" height="112" alt="image" src="https://github.com/user-attachments/assets/6738809f-339c-4d2b-9a05-b1e98a9684ec" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
