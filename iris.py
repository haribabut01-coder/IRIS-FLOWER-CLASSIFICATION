# ==========================================================
# 🌸 IRIS FLOWER CLASSIFICATION PROJECT
# ==========================================================

# Step 1️⃣: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================================
# Step 2️⃣: Load and explore the dataset
# ==========================================================
iris = load_iris()

# Convert dataset to pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("🔹 First 5 rows of dataset:")
print(df.head())

print("\n🔹 Dataset Information:")
print(df.info())

print("\n🔹 Statistical Summary:")
print(df.describe())

# ==========================================================
# Step 3️⃣: Data Visualization
# ==========================================================

# Pairplot for visualizing relationships
sns.pairplot(df, hue="species", palette="Set2")
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Heatmap for correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================================================
# Step 4️⃣: Data Preprocessing
# ==========================================================
X = df.iloc[:, :-1]  # Independent features
y = df.iloc[:, -1]   # Target variable

# Split dataset into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Step 5️⃣: Model Training with Multiple Algorithms
# ==========================================================
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42)
}

accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    
    print(f"\n🔹 Model: {name}")
    print("Accuracy:", round(acc*100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ==========================================================
# Step 6️⃣: Model Comparison Visualization
# ==========================================================
plt.figure(figsize=(8, 4))
plt.barh(list(accuracy_results.keys()), list(accuracy_results.values()), color='skyblue')
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.xlim(0.8, 1.0)
plt.show()

# ==========================================================
# Step 7️⃣: Selecting the Best Model
# ==========================================================
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]
print(f"\n✅ Best Performing Model: {best_model_name}")
print(f"🎯 Accuracy: {accuracy_results[best_model_name]*100:.2f}%")

# ==========================================================
# Step 8️⃣: Predicting New Data
# ==========================================================
# Example Input: [sepal length, sepal width, petal length, petal width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)

print(f"\n🌸 Predicted Species for input {sample.tolist()} is ➡️ {prediction[0]}")

# ==========================================================
# Step 9️⃣: Save the Model (Optional)
# ==========================================================
import joblib
joblib.dump(best_model, "iris_model.pkl")
print("\n💾 Model saved successfully as iris_model.pkl")
