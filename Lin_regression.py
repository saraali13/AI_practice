import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# Load data
df = pd.read_csv("Iris.csv")
x = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for a sample input
prediction = model.predict([[5.9, 3.6, 2.5, 0.5]])
rounded_prediction = int(round(prediction[0])) #for continuous data in linear regression 
predicted_species = le.inverse_transform([rounded_prediction])
print("Predicted Species:", predicted_species[0])

# Predict on test set
y_pred_test = model.predict(X_test)

# Convert float predictions to class labels
y_pred_labels = [int(round(p)) for p in y_pred_test]
print("Accuracy:", accuracy_score(y_test, y_pred_labels))

# Calculate MSE (Regression metric)
mse = mean_squared_error(y_test, y_pred_test)
print("MSE:", mse)
