import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = {
    "Income": [60,75,85.5,52.8,64.8,64.8,61.5,43.2,87,84,110.1,49.2,108,59.2,82.8,66,69,47.4,93,33,51,51,81,63],
    "Lawn_Size": [18.4,19.6,16.8,20.8,21.6,17.2,20.8,20.4,23.6,17.6,19.2,17.6,17.6,16,22.4,18.4,20,16.4,20.8,18.8,22,14,20,14.8],
    "Owner": ["Owner","Nonowner","Owner","Nonowner","Owner","Nonowner","Owner","Nonowner","Owner","Nonowner",
               "Owner","Nonowner","Owner","Nonowner","Owner","Nonowner","Owner","Nonowner","Owner","Nonowner",
               "Owner","Nonowner","Owner","Nonowner"]
}
df = pd.DataFrame(data)

le = LabelEncoder()
df["Owner"] = le.fit_transform(df["Owner"])

X = df[["Income","Lawn_Size"]]
y = df["Owner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
