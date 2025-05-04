import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score

df = pd.read_csv("Iris.csv")

x = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values
x=pd.DataFrame(x)
y=pd.Series(y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)
kfold=KFold(n_splits=7,shuffle=True,random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
scores=[]
for i,j in kfold.split(x):
    X_train, X_test = x.iloc[i], x.iloc[j]
    y_train, y_test = y.iloc[i], y.iloc[j]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print("K-Fold scores: ",scores)
print("K-Fold CV Average Accuracy:", np.mean(scores))
