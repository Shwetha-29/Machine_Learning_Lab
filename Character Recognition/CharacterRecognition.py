# Handwritten Digit & Alphabet Recognition GUI with Correct Label Mapping

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

# -------------------------
# 1. Load MNIST digits
# -------------------------
(X_train_digits, y_train_digits), (X_test_digits, y_test_digits) = mnist.load_data()
X_train_digits = X_train_digits / 255.0
X_test_digits = X_test_digits / 255.0
y_train_digits_36 = np.zeros((y_train_digits.shape[0],36))
y_train_digits_36[np.arange(y_train_digits.shape[0]), y_train_digits] = 1
y_test_digits_36 = np.zeros((y_test_digits.shape[0],36))
y_test_digits_36[np.arange(y_test_digits.shape[0]), y_test_digits] = 1

# -------------------------
# 2. Load EMNIST letters
# -------------------------
ds_train = tfds.load('emnist/letters', split='train', as_supervised=True)
ds_test = tfds.load('emnist/letters', split='test', as_supervised=True)

def prepare_emnist(ds):
    images = []
    labels = []
    for img, lbl in tfds.as_numpy(ds):
        img = img.astype(np.float32)/255.0
        images.append(img)
        labels.append(lbl)
    images = np.array(images)
    labels = np.array(labels)
    labels = labels - 1  # 0-25 for letters A-Z
    return images, labels

X_train_letters, y_train_letters = prepare_emnist(ds_train)
X_test_letters, y_test_letters = prepare_emnist(ds_test)

# Fix dimensions
X_train_letters = X_train_letters.squeeze()
X_test_letters = X_test_letters.squeeze()

# Rotate and flip letters to match GUI orientation
X_train_letters = np.array([np.fliplr(np.rot90(img, k=1)) for img in X_train_letters])
X_test_letters = np.array([np.fliplr(np.rot90(img, k=1)) for img in X_test_letters])

# Convert labels to one-hot 36-dim (letters occupy neurons 10-35)
y_train_letters_36 = np.zeros((y_train_letters.shape[0],36))
y_train_letters_36[np.arange(y_train_letters.shape[0]), 10 + y_train_letters] = 1
y_test_letters_36 = np.zeros((y_test_letters.shape[0],36))
y_test_letters_36[np.arange(y_test_letters.shape[0]), 10 + y_test_letters] = 1

# -------------------------
# 3. Combine datasets
# -------------------------
X_train = np.concatenate([X_train_digits, X_train_letters])
y_train = np.concatenate([y_train_digits_36, y_train_letters_36])
X_test = np.concatenate([X_test_digits, X_test_letters])
y_test = np.concatenate([y_test_digits_36, y_test_letters_36])

print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# -------------------------
# 4. Build MLP model
# -------------------------
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 5. Train model
# -------------------------
print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# -------------------------
# 6. Evaluate model
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy on Test Set: {accuracy*100:.2f}%\n")

# -------------------------
# 7. GUI for drawing
# -------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit & Alphabet Recognition")
        self.geometry("400x500")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict_char)
        self.predict_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=10)

        self.result_label = tk.Label(self, text="Prediction: ", font=("Arial", 16))
        self.result_label.pack(pady=10)

        self.image = Image.new("L", (280,280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x-8), (event.y-8)
        x2, y2 = (event.x+8), (event.y+8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,280,280], fill='white')
        self.result_label.config(text="Prediction: ")

    def predict_char(self):
        img = self.image.resize((28,28))
        img = ImageOps.invert(img)
        img = np.array(img)/255.0
        img = img.reshape(1,28,28)
        pred = model.predict(img)
        idx = np.argmax(pred)
        if idx < 10:
            char_type = "Digit"
            char = str(idx)
        else:
            char_type = "Alphabet"
            char = chr(ord('A') + idx - 10)
        self.result_label.config(text=f"Type: {char_type} | Recognized: {char}")

# Run the app
app = App()
app.mainloop()