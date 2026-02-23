# ======================
# SVM Spam Email Classifier with GUI & Confusion Matrix
# ======================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox, ttk

# ======================
# Step 1: Load Dataset
# ======================

df = pd.read_csv("emails.csv", encoding='latin-1')

# Encode target variable: ham=0, spam=1
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Step 2: Train SVM
# ======================

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# ======================
# Step 3: Evaluate Model
# ======================

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# Step 4: Display Confusion Matrix
# ======================

def show_confusion_matrix():
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

show_confusion_matrix()

# ======================
# Step 5: GUI for Sample Prediction
# ======================

def predict_email():
    try:
        email_text = email_entry.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Input Error", "Please enter some email text.")
            return

        email_vector = vectorizer.transform([email_text])
        pred = svm_model.predict(email_vector)[0]

        if pred == 1:
            result_label.config(text="Prediction: SPAM ❌", foreground="red")
        else:
            result_label.config(text="Prediction: NOT SPAM ✅", foreground="green")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# ======================
# GUI Layout
# ======================

root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("600x400")
root.configure(bg="#f0f0f5")

title = ttk.Label(root, text="📧 Email Spam Detector using SVM", font=("Helvetica", 18, "bold"))
title.pack(pady=15)

email_label = ttk.Label(root, text="Enter Email Text:", font=("Arial", 12))
email_label.pack(pady=5)

email_entry = tk.Text(root, height=8, width=70, font=("Arial", 11))
email_entry.pack(pady=5)

predict_btn = ttk.Button(root, text="Predict", command=predict_email)
predict_btn.pack(pady=10)

result_label = ttk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=15)

footer = ttk.Label(root, text="SVM | Desktop ML Application", font=("Arial", 10))
footer.pack(pady=10)

root.mainloop()