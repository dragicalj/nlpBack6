import random
from rest_framework import status, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Text, Token, Category, CustomUser
from .serializers import TextSerializer, CategorySerializer, UserSerializer
from .utils import calculate_entropy, calculate_shannon_value, tokenize_and_categorize
from .utils import chat_with_chatgpt, chat_with_llama, chat_with_gemini, chat_with_command, chat_with_claude, chat_with_qwen, chat_with_deepseek, chat_with_perplexity
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
import nltk.data
from collections import Counter
import string
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth.hashers import make_password, check_password
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.db.models import Count
from django.db import models
from django.db.models import Avg, Count

@api_view(['POST'])
def create_text(request):
    if request.method == 'POST':
        data = request.data.copy()
        serializer = TextSerializer(data=data)
        if serializer.is_valid():
            content = serializer.validated_data['content']
            
            sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
            sentences = sent_detector.tokenize(content)
            tokens = []
            for sentence in sentences:
                tokens.extend(word_tokenize(sentence))
            tokens = [token.lower() for token in tokens]
            num_occurrences=len(tokens)

            token_counts = Counter(tokens)
            num_tokens=len(token_counts)
            entropy = calculate_entropy(token_counts)
            shannon_value = calculate_shannon_value(entropy, num_tokens)

            top_token = token_counts.most_common(1)[0][0]
            words_only = [token for token in tokens if token not in string.punctuation]
            word_counts = Counter(words_only)
            top_word = word_counts.most_common(1)[0][0]

            # Sačuvaj tekst zajedno sa izračunatim vrednostima
            text = serializer.save(entropy=entropy, shannon_value=shannon_value, num_tokens=num_tokens, num_occurrences=num_occurrences, top_token=top_token, top_word=top_word)

            return Response({'id': text.id}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_generated_text(request):
    prompt = request.query_params.get('prompt', 'Default prompt if none provided')
    model = request.query_params.get('model', 'default_model')
    try:
        match(model):
            case "chatgpt":
                generated_text = chat_with_chatgpt(prompt)
            case "llama":
                generated_text = chat_with_llama(prompt)
            case "gemini":
                generated_text = chat_with_gemini(prompt)
            case "claude":
                generated_text = chat_with_claude(prompt)
            case "command":
                generated_text = chat_with_command(prompt)
            case "qwen":
                generated_text = chat_with_qwen(prompt)
            case "deepseek":
                generated_text = chat_with_deepseek(prompt)
            case "perplexity":
                generated_text = chat_with_perplexity(prompt)
            case _:
                generated_text = chat_with_chatgpt(prompt)
        return Response({'generated_text': generated_text}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_all_texts(request, user_id=None):
    if user_id:
        texts = Text.objects.filter(user_id=user_id).values('id', 'title')
    else:
        texts = Text.objects.all().values('id', 'title')
    
    return Response(texts)

@api_view(['GET'])
def get_text_content(request, text_id):
    try:
        text = Text.objects.get(id=text_id)
        return Response({'content': text.content})
    except Text.DoesNotExist:
        return Response({'error': 'Text not found'}, status=404)

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

@api_view(['POST'])
def tokenize_text(request):
    text = request.data.get('text', '')
    if not text:
        return Response({'error': 'No text provided'}, status=400)
    
    word_counts = tokenize_and_categorize(text)
    
    table_data = [
        {'text': word, 'concordance': count, 'type': word_type}
        for (word, word_type), count in word_counts.items()
    ]
    
    return Response(table_data)

@api_view(['GET'])
def get_text_metadata(request, id):
    try:
        text = Text.objects.get(id=id)
    except Text.DoesNotExist:
        return Response({'error': 'Text not found'}, status=status.HTTP_404_NOT_FOUND)

    data = {
        'num_tokens': text.num_tokens,
        'num_occurrences': text.num_occurrences,
        'top_token': text.top_token,
        'top_word': text.top_word
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_text_entropy_and_shannon_value(request, id):
    try:
        text = Text.objects.get(id=id)
    except Text.DoesNotExist:
        return Response({'error': 'Text not found'}, status=status.HTTP_404_NOT_FOUND)

    data = {
        'entropy': text.entropy,
        'shannon_value': text.shannon_value
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['POST'])
def tokenize_text_only(request):
    if request.method == 'POST':
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        
        token_counts = Counter(tokens)
        sorted_token_counts = token_counts.most_common()

        frequency_data = {token: count for token, count in sorted_token_counts}
        
        return Response(frequency_data, status=status.HTTP_200_OK)
    
@api_view(['POST'])
def create_category(request):
    if request.method == 'POST':
        serializer = CategorySerializer(data=request.data)
        if serializer.is_valid():
            category = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_categories(request):
    categories = Category.objects.all()
    serializer = CategorySerializer(categories, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def register_user(request):
    if request.method == 'POST':
        data = request.data.copy()
        data['password'] = make_password(data.get('password'))  # Enkripcija lozinke
        serializer = UserSerializer(data=data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({'id': user.id}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['POST'])
def login_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    
    try:
        user = CustomUser.objects.get(username=username)
    except CustomUser.DoesNotExist:
        return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
    
    if check_password(password, user.password):
        return Response({'id': user.id}, status=status.HTTP_200_OK)
    else:
        return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['GET'])
def get_texts_by_category(request, category_id):
    user_id = request.query_params.get('userId')
    if user_id is None:
        return Response({'detail': 'User ID is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        category = Category.objects.get(id=category_id)
        texts = Text.objects.filter(category=category, user_id=user_id)
        serializer = TextSerializer(texts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Category.DoesNotExist:
        return Response({'detail': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def get_categories_with_text_count(request):
    user_id = request.query_params.get('userId')
    if user_id is None:
        return Response({'detail': 'User ID is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        categories = Category.objects.annotate(
            text_count=Count('texts', filter=models.Q(texts__user_id=user_id))
        )
        serializer = CategorySerializer(categories, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_user_texts(request, user_id):
    try:
        texts = Text.objects.filter(user_id=user_id, category__isnull=True).values('id', 'title')
        return Response(list(texts), status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def add_text_to_category(request):
    text_id = request.data.get('text_id')
    category_id = request.data.get('category_id')

    if text_id is None or category_id is None:
        return Response({'detail': 'Text ID and Category ID are required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        text = Text.objects.get(id=text_id)
        category = Category.objects.get(id=category_id)
        text.category = category
        text.save()
        return Response({'detail': 'Text successfully added to category'}, status=status.HTTP_200_OK)
    except Text.DoesNotExist:
        return Response({'detail': 'Text not found'}, status=status.HTTP_404_NOT_FOUND)
    except Category.DoesNotExist:
        return Response({'detail': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
def update_category(request, category_id):
    try:
        category = Category.objects.get(id=category_id)
    except Category.DoesNotExist:
        return Response({'detail': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = CategorySerializer(category, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_categories(request):
    categories = Category.objects.all()
    serializer = CategorySerializer(categories, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_category_statistics(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
    categories = Category.objects.all()
    category_stats = []

    for category in categories:
        texts = Text.objects.filter(category=category, user=user)
        if texts.exists():
            mean_entropy = texts.aggregate(Avg('entropy'))['entropy__avg']
            mean_shannon_value = texts.aggregate(Avg('shannon_value'))['shannon_value__avg']
            mean_num_tokens = texts.aggregate(Avg('num_tokens'))['num_tokens__avg']

            top_tokens = [text.top_token for text in texts]
            top_words = [text.top_word for text in texts]

            most_common_top_token = Counter(top_tokens).most_common(1)[0][0]
            most_common_top_word = Counter(top_words).most_common(1)[0][0]

            category_stats.append({
                'id': category.id,
                'name': category.name,
                'mean_entropy': mean_entropy,
                'mean_shannon_value': mean_shannon_value,
                'mean_num_tokens': mean_num_tokens,
                'top_token': most_common_top_token,
                'top_word': most_common_top_word
            })

    return Response(category_stats)

@api_view(['GET'])
def get_shannon_equitability_by_category(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
    categories = Category.objects.all()
    data = []

    for category in categories:
        texts = Text.objects.filter(category=category, user=user).values_list('shannon_value', flat=True)
        for shannon_value in texts:
            data.append({
                'category': category.name,
                'shannon_value': shannon_value
            })

    return Response(data)

@api_view(['GET'])
def get_frequent_nouns_by_category(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    categories = Category.objects.all()
    category_nouns = []
    category_colors = {}

    for category in categories:
        if category.name not in category_colors:
            category_colors[category.name] = '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        
        texts = Text.objects.filter(category=category, user=user)
        combined_content = ' '.join([text.content for text in texts])
        word_counts = tokenize_and_categorize(combined_content)
        
        noun_counts = {word: count for (word, category), count in word_counts.items() if category == 'Noun'}
        most_common_nouns = Counter(noun_counts).most_common(10)
        
        for noun, frequency in most_common_nouns:
            category_nouns.append({
                'text': noun,
                'value': frequency,
                'category': category.name,
                'color': category_colors[category.name]
            })

    return Response(category_nouns)

@api_view(['POST'])
def text_entropy_shannon_by_content(request):
    text = request.data.get('text')
    if not text:
        return Response({'error': 'No text provided'}, status=400)
    
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    token_counts = Counter(tokens)
    num_tokens=len(token_counts)
    entropy = calculate_entropy(token_counts)
    shannon_value = calculate_shannon_value(entropy, num_tokens)
    
    return Response({'entropy': entropy, 'shannon_value': shannon_value})
