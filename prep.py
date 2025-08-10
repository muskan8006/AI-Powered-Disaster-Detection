#@title 📝 Prepare Text Dataset (fallback uses tweet_eval sentiment as disaster proxy)
from datasets import load_dataset
print("Loading tweet_eval sentiment as a proxy dataset (treat 'negative' as disaster-urgent).")
ds = load_dataset("tweet_eval", "sentiment")
train_texts = ds['train']['text']
train_labels = ds['train']['label']
val_texts = ds['validation']['text']
val_labels = ds['validation']['label']
test_texts = ds['test']['text']
test_labels = ds['test']['label']
print("Loaded tweet_eval (train samples):", len(train_texts)) 
