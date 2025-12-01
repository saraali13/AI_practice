from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


log = LogisticRegression()#not necessary max_iter

log = LogisticRegression(max_iter=500,solver="newton-cg")#not necessary max_iter, solvers-> liblinear,lbfs and newton-cg (for high dim dataset), sag or saga (large dataset)
model=log.fit(X_train, y_train)
pred = model.predict(X_test)
acc=accuracy_score(y_test,pred)

slope=model.coef_
intercept=model.intercept_

#multinomial logistic Regression
MLR=LogisticRegression(multi_class="multinomial")
model=log.fit(X_train, y_train)
pred = model.predict(X_test)

log = LogisticRegression(solver="liblinear",penalty="l2")#penalty=l1,l2, elastic net regularization

