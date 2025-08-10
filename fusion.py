#@title 🔗 Fusion demo: run combined example predictions
import numpy as np
from tensorflow.keras.preprocessing import image
from IPython.display import display

# helper functions
def predict_image_tensor(img_tensor):
    arr = np.expand_dims(img_tensor,0)/255.0
    preds = cnn.predict(arr)
    idx = np.argmax(preds, axis=1)[0]
    prob = preds[0, idx]
    return IMAGE_CLASS_NAMES[idx], float(prob)

def predict_text_single(text):
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='tf')
    logits = text_model(enc)[0]
    pred = tf.argmax(logits, axis=1).numpy()[0]
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    return int(pred), float(probs[pred])

# pick sample image from val_ds
for batch in val_ds.take(1):
    imgs, labs = batch
    sample_img = imgs[0].numpy()
    break

img_class, img_prob = predict_image_tensor(sample_img)
print("Sample image predicted class:", img_class, "prob:", round(img_prob,3))

# sample tweets
for i in range(3):
    txt = test_texts[i]
    pred_label, pred_prob = predict_text_single(txt)
    print(f"Tweet {i+1} -> label={pred_label}, prob={round(pred_prob,3)}, text='{txt[:80]}...'")

# simple fusion rule (example)
severity_map = {c: (i%3+1) for i,c in enumerate(IMAGE_CLASS_NAMES)}
img_sev = severity_map.get(img_class,1)
# treat tweet label 0 (negative) as higher severity
text_sev = 3 if pred_label==0 else 1
severity_score = img_prob*img_sev*0.6 + pred_prob*text_sev*0.4
print("Combined severity score example:", round(severity_score,3))
