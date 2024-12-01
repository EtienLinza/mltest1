import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
import pandas as pd
import threading


class MLTrainer:
    def __init__(self, master):
        self.master = master
        self.master.title("ML Trainer")
        self.master.geometry("600x500")

        self.file_paths = []  # To store file paths

        # GUI Elements
        self.file_label = tk.Label(self.master, text="Drag and drop files or click below to select:", font=("Arial", 12))
        self.file_label.pack(pady=10)

        self.drop_area = tk.Label(self.master, text="Drop files here", relief="groove", width=40, height=5, bg="lightgray")
        self.drop_area.pack(pady=10)
        self.drop_area.bind("<Button-1>", self.simulate_drop)  # Simulates drag-and-drop

        self.select_button = tk.Button(self.master, text="Select Files", command=self.select_files)
        self.select_button.pack(pady=10)

        self.clear_button = tk.Button(self.master, text="Clear Selected Files", command=self.clear_files)
        self.clear_button.pack(pady=5)

        self.train_button = tk.Button(self.master, text="Train Models", command=self.train_models)
        self.train_button.pack(pady=10)

        self.status_label = tk.Label(self.master, text="Status: Waiting for input", fg="blue", font=("Arial", 10))
        self.status_label.pack(pady=5)

    def simulate_drop(self, event):
        """Simulates file drop with a file dialog."""
        files = filedialog.askopenfilenames(title="Select Files", filetypes=[("CSV files", "*.csv")])
        if files:
            self.file_paths.extend(files)
            self.drop_area.config(text=f"{len(self.file_paths)} file(s) added")
            self.status_label.config(text=f"Status: {len(self.file_paths)} file(s) selected", fg="green")

    def select_files(self):
        """Allows file selection via dialog."""
        files = filedialog.askopenfilenames(title="Select Files", filetypes=[("CSV files", "*.csv")])
        if files:
            self.file_paths.extend(files)
            self.status_label.config(text=f"Status: {len(self.file_paths)} file(s) selected", fg="green")
            messagebox.showinfo("Files Added", f"{len(files)} file(s) selected successfully!")

    def clear_files(self):
        """Clears the file selection."""
        self.file_paths.clear()
        self.drop_area.config(text="Drop files here")
        self.status_label.config(text="Status: Waiting for input", fg="blue")
        messagebox.showinfo("Info", "All selected files cleared!")

    def train_models(self):
        """Trains models on selected files."""
        if not self.file_paths:
            messagebox.showerror("Error", "No files selected.")
            self.status_label.config(text="Status: No files selected", fg="red")
            return

        threading.Thread(target=self._train_models).start()  # Avoid freezing the GUI

    def _train_models(self):
        try:
            self.status_label.config(text="Status: Training in progress...", fg="orange")

            # Load and preprocess data
            X, y = self.load_and_preprocess_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=42)),
                ("LightGBM", lgb.LGBMClassifier(n_estimators=100, random_state=42)),
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

            self.status_label.config(text="Status: Training complete", fg="green")
            self.plot_results(results)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Status: Error occurred", fg="red")

    def load_and_preprocess_data(self):
        """Loads and preprocesses data from CSV files."""
        data_frames = [pd.read_csv(file_path) for file_path in self.file_paths]
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Assuming the last column is the target and the rest are features
        X = combined_data.iloc[:, :-1].values
        y = combined_data.iloc[:, -1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        return X, y

    def plot_results(self, results):
        """Plots results."""
        names, accuracies = zip(*results.items())
        plt.figure(figsize=(10, 6))
        sns.barplot(x=accuracies, y=names, palette="viridis")
        plt.title("Model Accuracies")
        plt.xlabel("Accuracy")
        plt.show()


def main():
    root = tk.Tk()
    app = MLTrainer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
