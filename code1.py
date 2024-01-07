import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

class MLTrainer:
    def __init__(self, master):
        self.master = master
        self.master.title("ML Trainer")

        self.file_label = tk.Label(self.master, text="Drag and drop training files here:")
        self.file_label.pack(pady=10)

        self.drop_area = tk.Label(self.master, text="Drop Area", relief="groove", width=40, height=10, bg="lightgray")
        self.drop_area.pack(pady=10)

        self.train_button = tk.Button(self.master, text="Train Models", command=self.train_models)
        self.train_button.pack(pady=10)

    def train_models(self):
        file_paths = filedialog.askopenfilenames(title="Select training files", filetypes=[("CSV files", "*.csv")])

        if not file_paths:
            return

        X, y = self.load_and_preprocess_data(file_paths)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = [
            ("Random Forest", self.build_random_forest_model()),
            ("XGBoost", self.build_xgboost_model()),
            ("LightGBM", self.build_lightgbm_model()),
            ("SVM", SVC()),
            ("K-Nearest Neighbors", KNeighborsClassifier()),
            ("Logistic Regression", LogisticRegression()),
            ("Naive Bayes", GaussianNB())
        ]

        results = {}

        for name, model in models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results[name] = accuracy
            print(f"{name} Model Accuracy: {accuracy}")

        self.plot_results(results)

    def load_and_preprocess_data(self, file_paths):
        iris = load_iris()
        X, y = iris.data, iris.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        return X, y

    def build_random_forest_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def build_xgboost_model(self):
        return xgb.XGBClassifier(n_estimators=100, random_state=42)

    def build_lightgbm_model(self):
        return lgb.LGBMClassifier(n_estimators=100, random_state=42)

    def plot_results(self, results):
        names, accuracies = zip(*results.items())
        plt.figure(figsize=(10, 6))
        sns.barplot(x=accuracies, y=names, palette="viridis")
        plt.title("Model Accuracies")
        plt.xlabel("Accuracy")
        plt.show()

def main():
    root = tk.Tk()
    ml_trainer = MLTrainer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
