import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

X = data['sqft_living'].values[:200]
y = data['price'].values[:200]

X_mat = np.c_[np.ones(len(X)), X]

tau = 500

# LWLR Function
def lwlr(query_point, X, y, tau):

    m = X.shape[0]
    W = np.eye(m)

    for i in range(m):
        diff = query_point[1] - X[i][1]
        W[i,i] = np.exp(-(diff**2)/(2*tau**2))

    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

    return query_point @ theta


# Predictions for dataset
def compute_predictions():

    y_pred = []

    for val in X:
        qp = np.array([1,val])
        y_pred.append(lwlr(qp,X_mat,y,tau))

    return np.array(y_pred)


# Calculate evaluation metrics automatically
y_pred = compute_predictions()

mse = mean_squared_error(y,y_pred)
mae = mean_absolute_error(y,y_pred)
r2 = r2_score(y,y_pred)


# Prediction function
def predict_price():

    try:
        sqft = float(entry_sqft.get())

        qp = np.array([1,sqft])
        price = lwlr(qp,X_mat,y,tau)

        result_label.config(text=f"Predicted Price: ${price:,.2f}")

    except:
        messagebox.showerror("Error","Enter valid number")


# Graph
def show_graph():

    plt.figure()

    plt.scatter(X,y,label="Actual Data")
    plt.scatter(X,y_pred,label="Predicted Data")

    plt.xlabel("sqft_living")
    plt.ylabel("House Price")
    plt.title("Locally Weighted Linear Regression")

    plt.legend()
    plt.show()


# Tkinter Window
root = tk.Tk()
root.title("House Price Prediction using LWLR")
root.geometry("420x420")


title = tk.Label(root,text="LWLR House Price Predictor",font=("Arial",16))
title.pack(pady=10)


# Evaluation metrics displayed automatically
metrics_label = tk.Label(root,
text=f"""Evaluation Metrics

MSE : {mse:.2f}
MAE : {mae:.2f}
R2 Score : {r2:.3f}
""",
font=("Arial",10),
justify="left")

metrics_label.pack(pady=10)


# Input section
tk.Label(root,text="Enter House Size (sqft_living)").pack()

entry_sqft = tk.Entry(root)
entry_sqft.pack(pady=5)


predict_btn = tk.Button(root,text="Predict Price",command=predict_price)
predict_btn.pack(pady=10)


result_label = tk.Label(root,text="")
result_label.pack(pady=10)


graph_btn = tk.Button(root,text="Show Regression Graph",command=show_graph)
graph_btn.pack(pady=10)


root.mainloop()