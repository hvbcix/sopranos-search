import pandas as pd
import re
from textblob import TextBlob

# Wczytaj dane z CSV
input_file = "sopranos_transcripts_full.csv"
output_file = "sopranos_transcripts_enriched.csv"
df = pd.read_csv(input_file)

# Lista słów na potrzeby analizy
profanities = ["damn", "hell", "shit", "fuck", "bastard"]  # Możesz rozszerzyć listę
italian_words = ["ciao", "prego", "grazie", "capo", "amore", "mamma"]  # Włoskie słowa
food_references = ["pizza", "pasta", "wine", "meatball", "spaghetti", "bread", "cheese"]  # Jedzenie
names = ["Tony", "Carmela", "Christopher", "Meadow", "Junior", "Paulie"]  # Możesz dodać inne imiona

# Funkcje pomocnicze
def word_count(text):
    return len(text.split())

def contains_profanity(text):
    return any(word.lower() in profanities for word in text.split())

def contains_italian(text):
    return any(word.lower() in italian_words for word in text.split())

def character_count(text):
    return len(text)

def is_question(text):
    return text.strip().endswith("?")

def is_exclamation(text):
    return text.strip().endswith("!")

def contains_name(text):
    return any(name in text for name in names)

def contains_food_reference(text):
    return any(word.lower() in food_references for word in text.split())

def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Dodanie kolumn
df["Word Count"] = df["Text"].apply(word_count)
df["Contains Profanity"] = df["Text"].apply(contains_profanity)
df["Contains Italian"] = df["Text"].apply(contains_italian)
df["Character Count"] = df["Text"].apply(character_count)
df["Is Question"] = df["Text"].apply(is_question)
df["Is Exclamation"] = df["Text"].apply(is_exclamation)
df["Contains Name"] = df["Text"].apply(contains_name)
df["Contains Food Reference"] = df["Text"].apply(contains_food_reference)
df["Sentiment Score"] = df["Text"].apply(sentiment_score)

# Zapis do nowego pliku CSV
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Zapisano wzbogacone dane do pliku {output_file}")
