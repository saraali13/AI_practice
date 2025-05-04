import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Iris.csv")

# Extract features and encode target
X = df.iloc[:, 1:5].values  # SepalLength, SepalWidth, PetalLength, PetalWidth
y_ = df.iloc[:, -1]      # Species column

# Encode 'Setosa' as 1, others as 0 (binary classification for ROC)
y = (y_== "Iris-setosa").astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with probability enabled
model = SVC(probability=True)
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class '1' (Setosa)

# ROC computation
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='SVM (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--') #(0,0) to (1,1) K=black -- = dotted line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Iris-Setosa vs Others')
plt.legend(loc='lower right')
plt.grid()
plt.show()
