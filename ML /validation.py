from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

#80% 20%
#thwn validation that 80% -> 70% 30%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train2,x_test_val,y_train2,y_test_val=train_test_split(x_train,y_train,test_size=0.3,random_state=0)


scores = cross_val_score(dt, X, y, cv=5)
print("K-Fold CV Scores:", scores)
print("Average Accuracy:", np.mean(scores))

loo = LeaveOneOut()
scores = cross_val_score(knn, X, y, cv=loo)
print("Leave-One-Out CV Accuracy:", np.mean(scores))
