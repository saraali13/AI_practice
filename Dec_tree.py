import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("Iris.csv")

x = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction=model.predict([[5.9,3.6,2.5,0.5]])
predicted_species = le.inverse_transform(prediction)
print("Predicted Species: ", predicted_species[0])

dt_scores = cross_val_score(model, x, y)
print("Decision Tree Average Accuracy: ", dt_scores.mean())

y_pred_test = model.predict(X_test)
print("Testing Accuracy: ", accuracy_score(y_test, y_pred_test))

score=model.score(X_train,y_train)
print("Training Accuracy: ",score)
