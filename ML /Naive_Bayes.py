from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB, BernoulliNB, CpmplementNB
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
X = cv.fit_transform(df['text'])

X_train, X_test, y_train, y_test = train_test_split(X, df['label'])

Gnb = GaussianNB()
Mnb = MultinomialNB()
Mnb.fit(X_train, y_train)

pred = Mnb.predict(X_test)
acc=accuracy_score(y_test,pred)

bin_x_train=(xtrain > xtrain.mean(axis=0)).astype(int)
bin_x_test=(xtest > xtest.mean(axis=0)).astype(int)
Bnb=BernoulliNB()
Bnb.fit(bin_x_train, y_train)
pred = Bnb.predict(bin_x_test)

Cnb=CpmplementNB()

crp=classification_report(y_test,pred)
cm=comfusion_matrix(y_test,pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])



