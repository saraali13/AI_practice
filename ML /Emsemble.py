from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", rf.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#Boosting algos
#ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada = AdaBoostClassifier(algorithm='SAMME', random_state=42) #or SAMMER
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost Accuracy:", ada.score(X_test, y_test))

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("AdaBoost Accuracy:", axgb.score(X_test, y_test))

dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)

#final preds
#1) AVD
final_pred=(y_pred_rf + y_pred_ada + y_pred_dt)/3
print(final_pred)

#2) Voting Classifiers
#hard voting
hvoting = VotingClassifier(estimators=[('knn', knn), ('dt', dt_entropy), ('rf', rf)], voting='hard')
hvoting.fit(X_train, y_train)
print("Voting Classifier Accuracy:", hvoting.score(X_test, y_test))
#soft voting
svoting = VotingClassifier(estimators=[('knn', knn), ('dt', dt_entropy), ('rf', rf)], voting='hard')
svoting.fit(X_train, y_train)
print("Voting Classifier Accuracy:", svoting.score(X_test, y_test))

#Bagging
bagging_model=BaggingClassifier(DecisionTreeClassifier(random_state=42))
bagging_model.fit(X_train, y_train)
bagging_model.score(X_test, y_test)
