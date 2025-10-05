from sklearn.tree import DecisionTreeClassifier, plot_tree

#dt = DecisionTreeClassifier()
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_dt = dt_entropy.predict(X_test)

print("Decision Tree (Entropy) Accuracy:", dt_entropy.score(X_test, y_test))#or accuracy scire
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

plt.figure(figsize=(10,6))
plot_tree(dt_entropy, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

#dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_pruned = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.05, random_state=42)
dt_pruned.fit(X_train, y_train)
print("Decision Tree (Pruned) Accuracy:", dt_pruned.score(X_test, y_test))

dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
print("Decision Tree (Gini) Accuracy:", dt_gini.score(X_test, y_test))
dt_ginip = DecisionTreeClassifier(criterion='gini',ccp_alpha=0.05)
dt_ginip.fit(X_train, y_train)
print("Decision Tree (Gini pruned) Accuracy:", dt_ginip.score(X_test, y_test))

dt_df=tree.export_graphviz(dt,feature_name=x,filled=True,rounded=True,special_charater=True)
graph=graphviz.Score(dt_data)
print(graph)
