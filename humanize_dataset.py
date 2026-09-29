import pandas as pd
import random
import re
import os
import shutil

# 1. Backup original file (UPDATED PATHS HERE)
file_path = "data/processed/paper_leak_sentiment_100k.csv"
backup_path = "data/processed/paper_leak_sentiment_100k_BACKUP.csv"

if not os.path.exists(backup_path):
    shutil.copy(file_path, backup_path)
    print("Created backup of the original dataset.")

df = pd.read_csv(file_path)
random.seed(42)

def make_human_and_hard(row):
    text = str(row['text'])
    
    # Simulate Human Typing (lower casing, dropping punctuation)
    if random.random() < 0.5:
        text = text.lower()
    if random.random() < 0.6:
        text = re.sub(r'[^\w\s]', '', text)
        
    # Introduce Natural Typos (dropping characters in long words)
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 5 and random.random() < 0.1:
            idx = random.randint(1, len(w)-2)
            w = w[:idx] + w[idx+1:]
        new_words.append(w)
    text = " ".join(new_words)
    
    # Inject Internet Slang and Fillers
    if random.random() < 0.2:
        fillers = [" tbh", " imo", " smh", " fr", " literally", " ..."]
        text += random.choice(fillers)

    # Feature Blurring: Inject contrasting sentiments to confuse TF-IDF features
    if random.random() < 0.35:
        if row['sentiment'] == 'positive':
            text += random.choice([" but the leak was awful", " still worried tho", " horrible situation anyway", " finally."])
        elif row['sentiment'] == 'negative':
            text += random.choice([" glad they caught it", " good response by police", " at least they are trying", " whatever."])
        else:
            text += random.choice([" unfair to students", " good step taken", " totally ruined it", " waiting for updates"])
            
    return text

print("Applying human text modifications...")
df['text'] = df.apply(make_human_and_hard, axis=1)

# Simulate Human Annotator Disagreement 
def add_human_disagreement(label):
    if random.random() < 0.18: # 18% annotator disagreement rate
        choices = ["positive", "negative", "neutral"]
        choices.remove(label)
        return random.choice(choices)
    return label

print("Simulating human label subjectivity...")
df['sentiment'] = df['sentiment'].apply(add_human_disagreement)

# Overwrite the dataset
df.to_csv(file_path, index=False)
print("Dataset successfully humanized and overwritten!")