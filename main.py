from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


data_path = "./inputs/policy_data.csv"
df = pd.read_csv(data_path)

# 分離 features and label
X = df.drop("falses", axis=1)
y = df["falses"]

# 拆分 training_data, test_data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 創建模型實例
decision_tree_clf = DecisionTreeClassifier(random_state=42)
knn_clf = KNeighborsClassifier()
logistic_regression_clf = LogisticRegression(random_state=42, max_iter=1000)
gaussian_nb_clf = GaussianNB()

# 模型字典
models = {
    "Decision Tree": decision_tree_clf,
    "K-Nearest Neighbors": knn_clf,
    "Logistic Regression": logistic_regression_clf,
    "Gaussian Naive Bayes": gaussian_nb_clf,
}

for name, model in models.items():
    # Training
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # 評估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 輸出结果與性能
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("-" * 80)

# Draw decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree_clf,
    filled=True,
    feature_names=X.columns,
    class_names=["No Falses", "Falses"],
    rounded=True,
    proportion=False,
    precision=2,
)
plt.show()
