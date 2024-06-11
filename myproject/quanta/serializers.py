from rest_framework import serializers
from .models import Text, Category, CustomUser

class TextSerializer(serializers.ModelSerializer):
    class Meta:
        model = Text
        fields = ['title', 'content', 'user', 'category', 'entropy', 'shannon_value', 'num_tokens', 'num_occurrences', 'top_token', 'top_word']
        read_only_fields = ['entropy', 'shannon_value', 'num_tokens', 'num_occurrences', 'top_token', 'top_word']

class CategorySerializer(serializers.ModelSerializer):
    text_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'description', 'text_count']

from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'first_name', 'last_name', 'username', 'password']
        extra_kwargs = {'password': {'write_only': True}}
