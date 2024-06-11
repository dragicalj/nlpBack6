import math
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import requests
import re
from dotenv import load_dotenv
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def calculate_entropy(frequencies):
    total_count = sum(frequencies.values())
    entropy = 0.0
    for count in frequencies.values():
        probability = count / total_count
        entropy += probability * math.log2(probability)
    entropy = -entropy
    return entropy

def map_tag_to_category(tag, word):
    if word in string.punctuation:
        return 'Punctuation'
    elif tag.startswith('NN'):
        return 'Noun'
    elif tag.startswith('VB'):
        return 'Verb'
    elif tag.startswith('JJ'):
        return 'Adjective'
    elif tag.startswith('RB'):
        return 'Adverb'
    elif tag in ['PRP', 'PRP$', 'WP', 'WP$']:
        return 'Pronoun'
    elif tag == 'IN':
        return 'Preposition'
    elif tag == 'CC':
        return 'Conjunction'
    elif tag == 'UH':
        return 'Interjection'
    elif tag in ['DT', 'PDT']:
        return 'Article'
    else:
        return 'Other'

def tokenize_and_categorize(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    
    tagged_tokens = nltk.pos_tag(tokens)
    
    tokens_with_categories = [(word, map_tag_to_category(tag, word)) for word, tag in tagged_tokens]
    
    word_counts = Counter(tokens_with_categories)
    
    return word_counts

def calculate_shannon_value(entropy, N):
    Hmax = math.log2(N)
    equitability = entropy / Hmax if Hmax > 0 else 0
    return equitability

def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

def chat_with_gpt(input_text, model="gpt-3.5-turbo"):
    input_text = remove_special_characters(input_text)
    #load_dotenv()
    #api_key = os.getenv("OPENAI_API_KEY")
    api_key=""
    prefix = ""  
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": f"{model}",
        "messages": [{"role": "user", "content": prefix + input_text}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        print(response_data)
        if response.ok:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response_data['error']['message']}"
    except Exception as e:
        return f"Error: {str(e)}"