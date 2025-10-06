from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#knn = KNeighborsClassifier(n_neighbors=3)
#knn = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
#knn = KNeighborsClassifier(n_neighbors=3,algorithm="ball_tree")                        
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

train_acc=knn.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred_knn)

print("KNN Accuracy:", knn.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

#diff ks
acc=[]
for i in range(1,11):
  knn2=KNeighborsClassifier(n_neighbors=i)
  knn2.fit(X_train,y_train)
  pred_y=knn.predict(X_test)
  acc_=accuracy_score(y_test,pred_y)
  acc.append(acc_)

TP=cn_mat[1,1]# true positive
TN=cn_mat[0,0]# true negative
FP=cn_mat[0,1]# false positive
FN=cn_mat[1,0]# false negative

precision= TP/(TP+FP)
recall=TP/(TP+FN)
training_accuracy=knn.score(x_train,y_train)
testing_accuracy=knn.score(x_test,y_test)

print("Accuracy: ",acc)
print("Classification Report: ",cl_rep)
print("Confusion Matrix: ",cn_mat)

print(f"True Positive: {TP}\nTrue Negative: {TN}\nFalse Positive: {FP}\nFalse Negative: {FN}")
print("Precision: ",precision)
print("Recall: ",recall)
print("Training Accuracy: ",training_accuracy)
print("Testing Accuracy: ",testing_accuracy)
