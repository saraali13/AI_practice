import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("Iris.csv")

x = df.iloc[:, 1:5].values #features
y = df.iloc[:, -1].values  #target

# encode the target val to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Applying regression 
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

prediction=model.predict([[5.9,3.6,2.5,0.5]])
#decode 
predicted_species = le.inverse_transform(prediction)

print("Predicted Species:", predicted_species[0])
y_pred_test = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_test))
