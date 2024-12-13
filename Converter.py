import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog
import os

# Function to load and clean the dataset
def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    df_cleaned = df.drop(columns=['Customer Id', 'First Name', 'Last Name', 'Company', 'Phone 1', 'Phone 2', 'Email', 'Subscription Date', 'Website'])

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to categorical columns
    categorical_columns = ['City', 'Country']
    for col in categorical_columns:
        df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])

    # Get the desktop path
    desktop_path = os.path.expanduser("~/Desktop")
    cleaned_file_path = os.path.join(desktop_path, 'cleaned_customers.csv')

    # Save the cleaned dataset
    df_cleaned.to_csv(cleaned_file_path, index=False)

    return cleaned_file_path

# Function to open file dialog and select the CSV file
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        cleaned_file_path = clean_data(file_path)
        result_label.config(text=f"Cleaned file saved at: {cleaned_file_path}")
    else:
        result_label.config(text="No file selected.")

# Set up the GUI window
root = tk.Tk()
root.title("Data Cleaning UI")

# Set window size
root.geometry("400x200")

# Create and pack the UI elements
instruction_label = tk.Label(root, text="Select a CSV file to clean:")
instruction_label.pack(pady=10)

open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
