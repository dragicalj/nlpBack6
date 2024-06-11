from django.contrib import admin
from .models import CustomUser, Category, Text, Token

admin.site.register(CustomUser)
admin.site.register(Category)
admin.site.register(Text)
admin.site.register(Token)

