#@title 🖼 Prepare Image Dataset (satellite proxy: EuroSAT fallback)
import tensorflow as tf
import tensorflow_datasets as tfds
IMG_SIZE = (128,128)
BATCH = 32

# If Kaggle available, you could download a natural-disasters dataset here (not implemented automatically for all slugs).
# Fallback:
print("Loading EuroSAT (RGB) as satellite-image proxy via TFDS...")
(ds_train, ds_val), ds_info = tfds.load('eurosat/rgb', split=['train[:90%]','train[90%:]'], with_info=True, as_supervised=True)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

train_ds = ds_train.map(preprocess).shuffle(1024).batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(BATCH).prefetch(tf.data.AUTOTUNE)
IMAGE_CLASS_NAMES = ds_info.features['label'].names
print("Loaded EuroSAT with classes (sample):", IMAGE_CLASS_NAMES[:8]) 
