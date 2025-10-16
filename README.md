# IRIS-FLOWER-CLASSIFICATION📘 Overview

The Iris Flower Classification project is a classic example of a machine learning classification problem.
It aims to classify Iris flowers into one of three species — Setosa, Versicolor, or Virginica — based on their sepal and petal measurements.

Using Python and popular ML libraries like scikit-learn, pandas, and seaborn, this project demonstrates the full workflow of a supervised learning model — including data analysis, visualization, model training, and evaluation.

🧠 Objectives

Load and analyze the Iris dataset.

Visualize the relationships between flower features.

Train multiple classification models (SVM, KNN, Decision Tree, Logistic Regression).

Evaluate performance and identify the best algorithm.

Predict the flower species based on user inputs.

📊 Dataset Details

Dataset Name: Iris Dataset

Source: UCI Machine Learning Repository
 / sklearn.datasets

Samples: 150

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Classes:

Iris-setosa

Iris-versicolor

Iris-virginica

⚙️ Installation and Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification

2️⃣ Create a Virtual Environment (optional)
python -m venv venv
source venv/bin/activate     # For macOS/Linux
venv\Scripts\activate        # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Project
python iris_classification.py


Or, if using Jupyter Notebook:

jupyter notebook Iris_Flower_Classification.ipynb

📦 Requirements

All dependencies are listed in requirements.txt:

numpy
pandas
matplotlib
seaborn
scikit-learn


(Optional for development)

jupyter
flask
streamlit

📈 Project Workflow

Import Libraries → pandas, numpy, seaborn, scikit-learn

Load Dataset → sklearn.datasets.load_iris()

Data Exploration → Display summary statistics and correlations

Data Visualization → Pair plots, heatmaps, and accuracy charts

Model Training → Train multiple models

Model Evaluation → Compare accuracy and confusion matrix

Prediction → Predict species from user input

🧾 Example Code Snippet
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

🏆 Results
Model	Accuracy
Logistic Regression	97%
K-Nearest Neighbors	96%
Decision Tree	95%
Support Vector Machine	98–99% (Best)

✅ Best Performing Model: Support Vector Machine (SVM)

🌼 Sample Prediction

Input: [5.1, 3.5, 1.4, 0.2]
Output: Iris-setosa
