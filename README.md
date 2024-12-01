# MLTrainer: A GUI for Machine Learning Model Training

**MLTrainer** is a Python-based graphical user interface (GUI) that allows users to load datasets, preprocess the data, and train various machine learning models. This application provides an easy-to-use interface for training different classifiers, such as Random Forest, XGBoost, LightGBM, SVM, KNN, Logistic Regression, and Naive Bayes. Additionally, it visualizes the model performance in the form of a bar chart.

## Features

- **File Upload**: Drag-and-drop or select CSV files for training data.
- **Model Training**: Train multiple machine learning models on the provided dataset.
- **Model Accuracy**: Visualize the accuracy of the trained models using a bar chart.
- **Preprocessing**: Automatically handles data scaling (StandardScaler) and dimensionality reduction (PCA).
- **Multi-threaded**: Trains models in a separate thread to avoid freezing the GUI during training.
- **Clear Selection**: Clear the file selections and status with the click of a button.

## Installation

### Prerequisites

- **Python 3.7+**: Make sure you have Python installed. If not, download and install it from [here](https://www.python.org/downloads/).
- **Required Libraries**: The following libraries are required for running this project:
    - `tkinter`: For the graphical user interface.
    - `scikit-learn`: For machine learning model training and data preprocessing.
    - `xgboost`: For training XGBoost models.
    - `lightgbm`: For training LightGBM models.
    - `matplotlib`: For visualizing the results.
    - `seaborn`: For enhanced data visualization.
    - `pandas`: For handling CSV file data and data manipulation.

### Install Dependencies

Clone the repository:

```bash
git clone https://github.com/yourusername/mltrainer.git
cd mltrainer
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

If `tkinter` is not installed by default, you may need to install it using:

```bash
sudo apt-get install python3-tk
```

Or, on Windows, it should already be installed with the standard Python installation.

### Requirements File

Create a `requirements.txt` file in your project directory with the following content:

```txt
tkinter
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
```

You can generate this file with:

```bash
pip freeze > requirements.txt
```

## Usage

1. **Running the Application**:
   - To start the application, run the following command in the project directory:
   ```bash
   python MLTrainer.py
   ```

2. **Using the GUI**:
   - **Select Files**: Click the "Select Files" button to choose one or more CSV files containing your dataset.
   - **Drag-and-Drop**: You can also drag and drop your files directly into the drop area.
   - **Clear Files**: Click the "Clear Selected Files" button to remove selected files.
   - **Train Models**: After selecting your files, click the "Train Models" button to start training the models. The program will display the training progress, and once completed, it will show the accuracy of each model in a bar chart.

3. **Expected CSV Format**:
   The application assumes that the last column in your CSV files is the target (label) and the rest are features. Here’s an example of a simple dataset:

   | Feature1 | Feature2 | Feature3 | Target |
   |----------|----------|----------|--------|
   | 1.2      | 3.4      | 5.6      | 0      |
   | 7.8      | 9.0      | 1.2      | 1      |

4. **Model Training Process**:
   The following models are trained:
   - Random Forest
   - XGBoost
   - LightGBM
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Naive Bayes

   After training, the accuracy for each model is shown in a bar chart.

5. **Visualization**:
   After the models are trained, a bar chart is displayed showing the accuracy of each model. This visualization helps compare the performance of the models on the provided dataset.

## Code Explanation

- **MLTrainer Class**: This is the core class that defines the GUI and handles the logic for training the models.
- **File Handling**: Users can either select files using a file dialog or drag-and-drop them into the GUI. Files are loaded into pandas DataFrames.
- **Data Preprocessing**: Before training, the data is scaled using `StandardScaler`, and dimensionality reduction is applied using `PCA` to reduce the feature space to two components.
- **Model Training**: Various models (e.g., Random Forest, XGBoost, LightGBM, etc.) are trained using the preprocessed data, and their accuracies are computed.
- **Results Visualization**: A bar chart is plotted using `matplotlib` and `seaborn` to show the model accuracies.

### Key Functions:

- `simulate_drop(self, event)`: Simulates a drag-and-drop file selection using a file dialog.
- `select_files(self)`: Opens a file dialog to allow the user to select CSV files.
- `clear_files(self)`: Clears the selected files and resets the GUI.
- `train_models(self)`: Starts training the models in a separate thread to keep the GUI responsive.
- `load_and_preprocess_data(self)`: Loads the CSV files, combines them into a single DataFrame, and preprocesses the data (scaling and PCA).
- `plot_results(self, results)`: Plots a bar chart of the accuracy results of each model.

## Contributions

If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

### To Do

- Add more model options such as neural networks, decision trees, etc.
- Improve error handling for various types of data inconsistencies.
- Provide more visualizations for model evaluation (confusion matrix, ROC curves, etc.).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
