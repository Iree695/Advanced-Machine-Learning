# Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 10 classes (official Fashion-MNIST order)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load CSV dataset (you only have the test CSV)
data = pd.read_csv("fashion-mnist_test.csv")

# Separate labels and pixels (rows and colums)
labels = data.values[:, 0]
pixels = data.values[:, 1:]

# Reshape pixels to images (28x28x1) and normalize
x = pixels.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y = labels.astype("int64")


# Split into Train / Validation / Test from the same CSV
#  80% train,10% val, 10% test
X_temp, X_test, y_temp, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111111, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# CNN model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation = "relu", input_shape=(28, 28, 1)), # Faster and less gradient problems
    layers.MaxPooling2D(2), # 3*3 -> 2*2

    layers.Conv2D(64, 3, activation = "relu"), 
    layers.MaxPooling2D(2), # 3*3 -> 2*2

    layers.Flatten(), # 2D a 1D
    layers.Dense(128, activation = "relu"), # Faster and less gradient problems
    layers.Dropout(0.5), # off cells
    layers.Dense(10, activation="softmax", name="probs") # Exits in probabilities
])
model.summary()

# Compile
model.compile(
    optimizer="adam", # Fast and stable converges
    loss="sparse_categorical_crossentropy", # poorly prediction model : high->poor / low->good
    metrics=["accuracy"]
)
# Train
history = model.fit(
    X_train, y_train,
    epochs = 15,        # Dataset views to learn patterns
    batch_size = 64,    # less memory, stable train, better generalization
    validation_data=(X_val, y_val),
    verbose = 1         # info in screen
)
# Final evaluation on test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save model 
model.save("Fashion_mnist_cnn.keras") # Keras 
model.export("Fashion_mnist_savedmodel") # for ONNX export

# Accuracy + Loss plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("resultados_cnn.png", dpi=150)
plt.show()

# Confusion matrix (on the test split)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()