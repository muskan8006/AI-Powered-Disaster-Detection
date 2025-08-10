#@title 🤗 Text model: fine-tune DistilBERT (fast)
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_texts(texts, labels, max_len=128):
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
    dataset = tf.data.Dataset.from_tensor_slices((dict(enc), labels))
    return dataset

BATCH_SIZE = 16
train_ds_text = tokenize_texts(train_texts, train_labels).shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds_text = tokenize_texts(val_texts, val_labels).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds_text = tokenize_texts(test_texts, test_labels).batch(BATCH_SIZE)

num_labels = len(set(train_labels))
print("Num text labels:", num_labels)

text_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
text_model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

EPOCHS_TEXT = 2  # keep small for speed; increase to 3-4 if you have time
history_t = text_model.fit(train_ds_text, validation_data=val_ds_text, epochs=EPOCHS_TEXT)

# Evaluate
eval_res = text_model.evaluate(test_ds_text)
print("Text model eval (loss, acc):", eval_res)

# Save
text_model.save_pretrained('/content/crisisvision_text_model')
tokenizer.save_pretrained('/content/crisisvision_tokenizer')
print("Saved text model & tokenizer to /content/")
