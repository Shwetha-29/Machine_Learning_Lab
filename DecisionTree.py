# =========================================
# Decision Tree using Entropy + Info Gain
# Loan Approval Dataset
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -----------------------------------------
# Step 1: Create Dataset
# -----------------------------------------

data = {
    'Income':      ['High','High','High','Medium','Medium','Low','Low','Low'],
    'Credit':      ['Good','Good','Bad','Good','Bad','Good','Bad','Bad'],
    'Employment':  ['Yes','Yes','Yes','Yes','No','No','No','Yes'],
    'Loan':        ['Approve','Approve','Approve','Approve','Reject','Reject','Reject','Reject']
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# -----------------------------------------
# Step 2: Encode categorical → numbers
# -----------------------------------------

le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

print("\nEncoded Dataset:\n")
print(df)

# -----------------------------------------
# Step 3: Split features & target
# -----------------------------------------

X = df[['Income','Credit','Employment']]
y = df['Loan']

# -----------------------------------------
# Step 4: Train Decision Tree (Entropy)
# -----------------------------------------

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)
print("\nModel trained successfully!")

# -----------------------------------------
# Step 5: Prediction Example
# -----------------------------------------

# use DataFrame to avoid warning
sample = pd.DataFrame([[0,1,1]], columns=['Income','Credit','Employment'])

prediction = model.predict(sample)

print("\nNew Person Details:")
print("Income = High, Credit = Good, Employment = Yes")

if prediction[0] == 0:
    print("Prediction = Approve")
else:
    print("Prediction = Reject")

# -----------------------------------------
# Step 6: Clean Tree Visualization
# -----------------------------------------

plt.figure(figsize=(10,5))

plot_tree(
    model,
    feature_names=['Income','Credit','Employment'],
    class_names=['Approve','Reject'],
    filled=False,       # clean look
    impurity=False,     # hide entropy values
    rounded=True,
    fontsize=11
)

plt.tight_layout()
plt.show()