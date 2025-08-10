#@title 💾 Save artifacts and print README snippet
import json, os
artifacts = {
    "image_model": "/content/crisisvision_cnn.h5",
    "text_model": "/content/crisisvision_text_model",
    "tokenizer": "/content/crisisvision_tokenizer"
}
with open('/content/crisisvision_artifacts.json','w') as f:
    json.dump(artifacts, f)
print("Artifacts saved. Paths:", artifacts)

print("\n--- README snippet (copy into README.md in your repo) ---\n")
print("""
# CrisisVision — AI-Powered Disaster Detection

A multi-modal project that combines satellite-style image classification and social media text classification to detect and estimate disaster severity.

## How to run
1. Open CrisisVision.ipynb in Google Colab.
2. (Optional) Upload kaggle.json to use original Kaggle datasets.
3. Runtime -> Run all. Results, plots & models will be produced.

## What I used
- Image model: CNN trained on EuroSAT (satellite-image proxy)
- Text model: DistilBERT fine-tuned on tweet_eval sentiment (proxy for disaster urgency)
- Fusion: Simple example combining image + text confidences into a severity score

## Results
Please run the notebook and paste accuracy plots and confusion matrices here.
""")
