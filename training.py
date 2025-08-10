#@title 🏗 Image model: build, train, evaluate
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

num_classes = len(IMAGE_CLASS_NAMES)
print("Num image classes:", num_classes)

def build_cnn(input_shape=(128,128,3), num_classes=num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

cnn = build_cnn()
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

EPOCHS_IMG = 6  # change to 10 for better results if you have time
history = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_IMG)

# Save
cnn.save('/content/crisisvision_cnn.h5')
print("Saved image model at /content/crisisvision_cnn.h5")

# Plot training curves
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Image model accuracy'); plt.show()

# Small confusion matrix sample
y_true = []; y_pred = []
for batch in val_ds.take(20):
    imgs, labels = batch
    preds = np.argmax(cnn.predict(imgs), axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.tolist())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=False); plt.title('Confusion matrix (val sample)'); plt.show() 
