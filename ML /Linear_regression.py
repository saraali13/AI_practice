from sklearn.linear_model import LinearRegression
from sklearn.metrices import r2_score
from sklearn.preprocessing import PolynomialFeatures

#single-> 1 x 
#multi-> n xs
lr = LinearRegression()
model=lr.fit(X_train, y_train)

pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

slope=model.coef_
intercept=model.intercept_
sns.regplot(x=y_test,y=pred,ci=None,color="red")

#polynomial regression
poly=PolynomialFeatures(degree=3,include_bias=True)
x_train_trans=poly.fit_transform(X_train)
x_test_trans=poly.transform(X_test)
#now model training
lr = LinearRegression()
model=lr.fit(x_train_trans, y_train)
pred = model.predict(x_test_trans)
print(r2_score(y_test, pred))

