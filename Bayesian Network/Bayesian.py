import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

st.title("Heart Disease Prediction using Bayesian Network")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("heart.csv")

# Drop unnecessary columns
data = data.drop(columns=["id","dataset","ca","thal"])

# Convert TRUE/FALSE
data["fbs"] = data["fbs"].replace({"TRUE":1,"FALSE":0})
data["exang"] = data["exang"].replace({"TRUE":1,"FALSE":0})

# Encode categorical columns
data["sex"] = data["sex"].replace({"Male":1,"Female":0})

data["cp"] = data["cp"].replace({
"typical angina":0,
"atypical angina":1,
"non-anginal":2,
"asymptomatic":3
})

data["restecg"] = data["restecg"].replace({
"normal":0,
"lv hypertrophy":1,
"st-t abnormality":2
})

data["slope"] = data["slope"].replace({
"upsloping":0,
"flat":1,
"downsloping":2
})

# Binary target
data["num"] = data["num"].apply(lambda x:0 if x==0 else 1)

# Fill missing values
for col in ["trestbps","chol","thalch","oldpeak"]:
    data[col] = data[col].fillna(data[col].median())

for col in ["sex","cp","fbs","restecg","exang","slope"]:
    data[col] = data[col].fillna(data[col].mode()[0])

# Discretize numeric features
data["age"] = pd.qcut(data["age"],3,labels=[0,1,2])
data["trestbps"] = pd.qcut(data["trestbps"],3,labels=[0,1,2])
data["chol"] = pd.qcut(data["chol"],3,labels=[0,1,2])
data["thalch"] = pd.qcut(data["thalch"],3,labels=[0,1,2])
data["oldpeak"] = pd.qcut(data["oldpeak"],3,labels=[0,1,2])

# -----------------------------
# Train/Test Split
# -----------------------------
train_data, test_data = train_test_split(data,test_size=0.2,random_state=42)

# -----------------------------
# Learn Bayesian Structure
# -----------------------------
hc = HillClimbSearch(train_data)
dag = hc.estimate(scoring_method=BIC(train_data))

model = DiscreteBayesianNetwork(dag.edges())

# Train model
model.fit(train_data, estimator=BayesianEstimator)

# Inference engine
infer = VariableElimination(model)

# -----------------------------
# Evaluation Metrics
# -----------------------------
y_true = test_data["num"]
y_pred = []

for _,row in test_data.iterrows():

    evidence = row.drop("num").to_dict()

    q = infer.query(variables=["num"], evidence=evidence)

    pred = q.values.argmax()

    y_pred.append(pred)

accuracy = accuracy_score(y_true,y_pred)
precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)

st.subheader("Model Evaluation Metrics")

st.write("Accuracy :", round(accuracy,3))
st.write("Precision :", round(precision,3))
st.write("Recall :", round(recall,3))
st.write("F1 Score :", round(f1,3))

# -----------------------------
# Graph Display Button
# -----------------------------
if st.button("Show Bayesian Network Graph"):

    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    fig = plt.figure(figsize=(10,7))

    pos = nx.spring_layout(G,k=1)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2500,
        font_size=11,
        font_weight="bold",
        arrowsize=20
    )

    plt.title("Bayesian Network Structure")

    st.pyplot(fig)

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Patient Details")

age = st.selectbox(
"Age Group",
[0,1,2],
format_func=lambda x:["Young","Middle","Old"][x]
)

cp = st.selectbox(
"Chest Pain Type",
[0,1,2,3],
format_func=lambda x:["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x]
)

trestbps = st.selectbox(
"Resting Blood Pressure",
[0,1,2],
format_func=lambda x:["Low","Medium","High"][x]
)

chol = st.selectbox(
"Cholesterol Level",
[0,1,2],
format_func=lambda x:["Low","Medium","High"][x]
)

thalch = st.selectbox(
"Maximum Heart Rate",
[0,1,2],
format_func=lambda x:["Low","Medium","High"][x]
)

oldpeak = st.selectbox(
"ST Depression",
[0,1,2],
format_func=lambda x:["Low","Medium","High"][x]
)

exang = st.selectbox(
"Exercise Induced Angina",
[0,1],
format_func=lambda x:["No","Yes"][x]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    evidence = {
    "age":age,
    "cp":cp,
    "trestbps":trestbps,
    "chol":chol,
    "thalch":thalch,
    "oldpeak":oldpeak,
    "exang":exang
    }

    q = infer.query(variables=["num"], evidence=evidence)

    prob = q.values[1]

    st.write("Probability of Heart Disease :", round(prob,3))

    if prob > 0.5:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")