import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ======================
# LOAD DATASET
# ======================

df = pd.read_csv("Loan_Prediction_Problem_Dataset.csv")

# Drop Loan_ID
df = df.drop("Loan_ID", axis=1)

# ======================
# HANDLE MISSING VALUES
# ======================

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].median(), inplace=True)

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["Dependents"] = df["Dependents"].replace("3+", "3")

# Convert target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Encode categorical data
df = pd.get_dummies(df, drop_first=True)

# ======================
# SPLIT DATA
# ======================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODEL
# ======================

model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# DISPLAY DECISION TREE
# ======================

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["Rejected", "Approved"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ======================
# GUI FUNCTION
# ======================

def predict_loan():
    try:
        applicant_income = float(app_income_entry.get())
        loan_amount = float(loan_amount_entry.get())
        credit_history = float(credit_history_entry.get())

        input_data = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "LoanAmount": loan_amount,
            "Credit_History": credit_history
        }])

        # Add missing columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result_label.config(text="Loan Approved ✅", foreground="green")
        else:
            result_label.config(text="Loan Rejected ❌", foreground="red")

    except:
        messagebox.showerror("Error", "Enter valid numeric values")

# ======================
# CREATE GUI
# ======================

root = tk.Tk()
root.title("Loan Approval Prediction - Decision Tree")
root.geometry("500x400")

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill="both", expand=True)

title = ttk.Label(main_frame, text="🏦 Loan Approval Prediction", font=("Arial", 18, "bold"))
title.grid(row=0, column=0, columnspan=2, pady=20)

ttk.Label(main_frame, text="Applicant Income").grid(row=1, column=0, pady=10)
app_income_entry = ttk.Entry(main_frame)
app_income_entry.grid(row=1, column=1)

ttk.Label(main_frame, text="Loan Amount").grid(row=2, column=0, pady=10)
loan_amount_entry = ttk.Entry(main_frame)
loan_amount_entry.grid(row=2, column=1)

ttk.Label(main_frame, text="Credit History (1/0)").grid(row=3, column=0, pady=10)
credit_history_entry = ttk.Entry(main_frame)
credit_history_entry.grid(row=3, column=1)

predict_btn = ttk.Button(main_frame, text="Predict Loan Status", command=predict_loan)
predict_btn.grid(row=4, column=0, columnspan=2, pady=20)

result_label = ttk.Label(main_frame, text="", font=("Arial", 14, "bold"))
result_label.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()