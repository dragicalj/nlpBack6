import math
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import requests
import re
from dotenv import load_dotenv
import os
import google.generativeai as GoogleAI
from openai import OpenAI
from meta_ai_api import MetaAI
import anthropic
import cohere
import torch
from transformers import pipeline

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

def chat_with_chatgpt(prompt):
    prompt = remove_special_characters(prompt)
    client = OpenAI(api_key="API_KEY")
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

def chat_with_llama(prompt):
    # ----------API-------------
    # prompt = remove_special_characters(prompt)
    # ai = MetaAI()
    # response = ai.prompt(message=prompt)
    # return response
    # --------Lokalno-----------
    model = "PATH_TO_MODEL"
    pipe = pipeline(
        "text-generation",
        model=model,
        model_kwargs={
            "torch_dtype": torch.bfloat16
        },
        device="cuda"
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    outputs = pipe(
        messages,
        max_new_tokens=4096,
        do_sample=False
    )
    return outputs[0]["generated_text"][-1]["content"]

def chat_with_gemini(prompt):
    prompt = remove_special_characters(prompt)
    GoogleAI.configure(api_key="API_KEY")
    model_config = {
        "temperature": 1,
    }
    model = GoogleAI.GenerativeModel(
        "gemini-2.0-flash-exp", 
        generation_config=model_config
    )
    response = model.generate_content(remove_special_characters(prompt))
    return response.text

def chat_with_claude(prompt):
    prompt = remove_special_characters(prompt)
    client = anthropic.Anthropic(api_key="API_KEY",)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

def chat_with_command(prompt):
    prompt = remove_special_characters(prompt)
    co = cohere.ClientV2("API_KEY")
    response = co.chat(
        model="command-r-plus",
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response

def chat_with_qwen(prompt):
    prompt = remove_special_characters(prompt)
    client = OpenAI(
    api_key="API_KEY", 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        temperature=1,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    # return completion.model_dump_json()
    return completion.choices[0].message.content.strip()

def chat_with_deepseek(prompt):
    client = OpenAI(api_key="API_KEY", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=1
    )
    return response.choices[0].message.content.strip()