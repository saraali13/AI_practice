from sklearn.svm import SVC

svc=SVC()
model=svc.fit(xtrain,ytrain)
pred=model.predict(xtest)
acc=accuracy_score(ytest,pred)

svc=SVC(kernel="linear")
svc=SVC(kernel="rbf")
svc=SVC(kernel="poly",degree=4)

svm = SVC(
    kernel='rbf', 
    C=1,#c parameter
    gamma='scale', #not big val
)
svm.fit(X_train, y_train)

kernels=["linear","poly","rbf","sigmoid"]
for k in kernels:
    model=SVC(kernel=k,random_state=40)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    acc=accuracy_score(y_test,pred)
    print(f"SVM model with kernel {k} has accuracy {acc*100}")

