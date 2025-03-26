import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train_df[features]
y = train_df['Survived']

X_test_data = test_df[features]

print(f"Training data NaN values:\n{X.isna().sum()}")
print(f"Test data NaN values:\n{X_test_data.isna().sum()}")

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

predictions = model.predict(X_test_data)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created")
