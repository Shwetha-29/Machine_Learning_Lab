import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv")

df = df.drop("User_ID", axis=1)
df = pd.get_dummies(df, drop_first=True)

df["Stress_Category"] = df["Stress_Level"].apply(lambda x: 1 if x >= 5 else 0)

X = df.drop(["Stress_Level", "Stress_Category"], axis=1)
y = df["Stress_Category"]

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODEL
# ======================

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# DISPLAY ONE DECISION TREE
# ======================

plt.figure(figsize=(20, 10))

tree = rf.estimators_[0]  # First tree from Random Forest

plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["Low", "High"],
    filled=True,
    max_depth=3  # Limit depth for clarity
)

plt.title("Random Forest - Sample Decision Tree")
plt.show()

# ======================
# GUI FUNCTION
# ======================

def predict_stress():
    try:
        age = int(age_entry.get())
        phone = float(phone_entry.get())
        social = float(social_entry.get())
        productivity = int(prod_entry.get())
        sleep = float(sleep_entry.get())
        app_usage = int(app_entry.get())
        caffeine = int(caffeine_entry.get())
        weekend = float(weekend_entry.get())

        input_data = pd.DataFrame([{
            "Age": age,
            "Daily_Phone_Hours": phone,
            "Social_Media_Hours": social,
            "Work_Productivity_Score": productivity,
            "Sleep_Hours": sleep,
            "App_Usage_Count": app_usage,
            "Caffeine_Intake_Cups": caffeine,
            "Weekend_Screen_Time_Hours": weekend
        }])

        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]
        prediction = rf.predict(input_data)[0]

        if prediction == 1:
            result_label.config(
                text="Predicted Stress Level: HIGH ⚠",
                foreground="red"
            )
        else:
            result_label.config(
                text="Predicted Stress Level: LOW 😊",
                foreground="green"
            )

    except:
        messagebox.showerror("Input Error", "Please enter valid numeric values")


# ======================
# CREATE UI
# ======================

root = tk.Tk()
root.title("Stress Level Predictor")
root.geometry("520x650")
root.configure(bg="#eef2f7")

style = ttk.Style()
style.theme_use("clam")

main_frame = ttk.Frame(root, padding=25)
main_frame.pack(fill="both", expand=True)

title = ttk.Label(
    main_frame,
    text="📱 Stress Level Prediction System",
    font=("Helvetica", 20, "bold")
)
title.grid(row=0, column=0, columnspan=2, pady=25)

labels = [
    "Age",
    "Daily Phone Hours",
    "Social Media Hours",
    "Productivity Score (1-10)",
    "Sleep Hours",
    "App Usage Count",
    "Caffeine Intake (cups)",
    "Weekend Screen Time Hours"
]

entries = []

for i, text in enumerate(labels):
    lbl = ttk.Label(main_frame, text=text, font=("Arial", 12))
    lbl.grid(row=i+1, column=0, sticky="w", pady=10)

    entry = ttk.Entry(main_frame, width=28, font=("Arial", 11))
    entry.grid(row=i+1, column=1, pady=10)
    entries.append(entry)

age_entry, phone_entry, social_entry, prod_entry, sleep_entry, \
app_entry, caffeine_entry, weekend_entry = entries

predict_btn = ttk.Button(
    main_frame,
    text="Predict Stress Level",
    command=predict_stress
)
predict_btn.grid(row=10, column=0, columnspan=2, pady=25)

result_label = ttk.Label(
    main_frame,
    text="",
    font=("Arial", 14, "bold")
)
result_label.grid(row=11, column=0, columnspan=2, pady=15)

footer = ttk.Label(
    main_frame,
    text="Random Forest Model | Desktop ML Application",
    font=("Arial", 9)
)
footer.grid(row=12, column=0, columnspan=2, pady=10)

root.mainloop()