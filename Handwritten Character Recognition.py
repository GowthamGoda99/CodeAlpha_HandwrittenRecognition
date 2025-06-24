# Step 1: Upload dataset manually
from google.colab import files
uploaded = files.upload()  # Upload 'train.csv' or 'train.xlsx'

import pandas as pd

# Step 2: Read dataset (CSV or Excel)
file_name = next(iter(uploaded))
if file_name.endswith('.csv'):
    df = pd.read_csv(file_name)
elif file_name.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(file_name)
else:
    raise ValueError("Unsupported file format!")

print(f"✅ Dataset loaded: {file_name} — Shape:", df.shape)
df.head()

# Step 3: Data Preparation
import numpy as np

y = df.iloc[:, 0].values                     # First column = labels
X = df.iloc[:, 1:].values                    # Rest = pixel values

X = X / 255.0                                # Normalize 0-1
X = X.reshape(-1, 28, 28, 1)                 # Reshape for CNN

from tensorflow.keras.utils import to_categorical
y_cat = to_categorical(y, num_classes=10)    # One-hot encode

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Step 4: Model Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Model Training
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=2
)

# Step 6: Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")
print(f"🧪 Test Loss    : {test_loss:.4f}")

# Step 7: Plot Accuracy and Loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Step 8: Classification Report & Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("📄 Classification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Step 9: Visualize Some Wrong Predictions
errors = np.where(y_true != y_pred)[0][:9]

plt.figure(figsize=(6,6))
for i, idx in enumerate(errors):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"T: {y_true[idx]}, P: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
