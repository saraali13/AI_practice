import pandas as pd
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score

df = pd.read_csv("Iris.csv")

x = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)
model = SVC(probability=True) #kernel = linear,rbf,poly ect based of data visuals 
model.fit(X_train, y_train)

prediction=model.predict([[5.9,3.6,2.5,0.5]])
predicted_species = le.inverse_transform(prediction)
y_pred_test = model.predict(X_test)
print("Prediction: ", predicted_species)
print("SVM Accuracy score:", SVM_score.mean())# mean Score else score is a list
#print("SVM Accuracy:", accuracy_score(y_test, y_pred)) can do it like this as well
print("Confusion Matrix",confusion_matrix(y_test, y_pred_test))
# ROC curve (for 2 classes )
#fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
