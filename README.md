Machine Learning Model Trainer with GUI
This repository provides a graphical user interface (GUI) application for training multiple machine learning models using CSV files. It allows users to load datasets, preprocess the data, train multiple models, and visualize the accuracy of each model. The application supports popular machine learning algorithms such as Random Forest, XGBoost, LightGBM, SVM, K-Nearest Neighbors, Logistic Regression, and Naive Bayes.

Features
Drag-and-Drop File Selection: Users can drag and drop CSV files into the GUI for easy file selection.
Model Training: The application supports training multiple models on the selected dataset.
Preprocessing: The dataset is preprocessed with features scaled using StandardScaler and reduced to 2 dimensions using PCA.
Visualization: After training, the accuracies of all models are visualized in a bar chart.
Multithreading: Model training is done in a separate thread to ensure the GUI remains responsive during the process.
Requirements
To run this project, you need the following Python libraries:

tkinter: For the graphical user interface.
pandas: For data manipulation and reading CSV files.
xgboost: For the XGBoost model.
lightgbm: For the LightGBM model.
sklearn: For machine learning models, preprocessing, and evaluation.
matplotlib: For plotting results.
seaborn: For enhanced data visualization.
Install these dependencies using pip:

bash
pip install pandas xgboost lightgbm scikit-learn matplotlib seaborn
How It Works
User Interface:

The main window has drag-and-drop functionality or a button for file selection. Users can add CSV files for model training.
Once files are selected, users can train models by clicking the "Train Models" button.
After training, the results of each model's accuracy are plotted on a bar chart for easy comparison.
Preprocessing:

The program assumes that the last column in each CSV file is the target variable, and all other columns are feature variables.
The data is standardized using StandardScaler and reduced to 2 dimensions using Principal Component Analysis (PCA).
Training Models:

The application supports the following machine learning algorithms:
Random Forest
XGBoost
LightGBM
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Logistic Regression
Naive Bayes
Results:

The accuracy of each model is calculated using accuracy_score and displayed on a bar chart.
Example of Use
Launch the Application:

Run the Python script to launch the GUI.
bash
python app.py
Load Data:

Drag and drop CSV files into the "Drop files here" area or click the "Select Files" button to choose CSV files.
Train Models:

Click the "Train Models" button to start training. The models will be trained on the selected files, and their accuracy will be displayed in a bar chart.
View Results:

After training is complete, a bar chart showing the accuracies of all models will appear.
Code Explanation
Here’s an overview of the main components of the code:

MLTrainer Class:

This class defines the application and handles the logic for loading data, preprocessing, training models, and displaying results.

simulate_drop(): Simulates a file drop using the file dialog.

select_files(): Allows the user to select CSV files.

clear_files(): Clears the selected files.

train_models(): Starts the training process.

_train_models(): Handles model training and evaluation on a separate thread to keep the GUI responsive.

load_and_preprocess_data(): Loads the CSV data, scales features, and applies PCA.

plot_results(): Plots the model accuracies.

Main Function:

The main() function initializes the Tkinter window and starts the application.

Future Improvements
Support for More Algorithms: Add more machine learning algorithms like Neural Networks, Gradient Boosting, etc.
Cross-validation: Implement cross-validation to get a better estimate of model performance.
Hyperparameter Tuning: Allow users to fine-tune model hyperparameters.
License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
