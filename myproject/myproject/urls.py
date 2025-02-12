"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from quanta import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/create_text/', views.create_text, name='create_text'),
    path('api/generate_text/', views.get_generated_text, name='generate_text'),
    path('api/texts/<int:user_id>/', views.get_all_texts, name='get_all_texts'),
    path('api/texts2/<int:text_id>/', views.get_text_content, name='get_text_content'),
    path('api/tokenize_text/', views.tokenize_text, name='tokenize_text'),
    path('api/text_metadata/<int:id>/', views.get_text_metadata, name='get_text_metadata'),
    path('api/text_entropy_shannon/<int:id>/', views.get_text_entropy_and_shannon_value, name='get_text_entropy_shannon'),
    path('api/tokenize_text_only/', views.tokenize_text_only, name='tokenize_text_only'),
    path('api/create_category/', views.create_category, name='create_category'),
    path('api/categories/', views.get_categories, name='get_categories'),
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/categories/<int:category_id>/texts/', views.get_texts_by_category, name='get_texts_by_category'),
    path('api/categories_with_text_count/', views.get_categories_with_text_count, name='get_categories_with_text_count'),
    path('api/user_texts/<int:user_id>/', views.get_user_texts, name='get_user_texts'),
    path('api/add_text_to_category/', views.add_text_to_category, name='add_text_to_category'),
    path('api/categories/<int:category_id>/', views.update_category, name='update_category'),
    path('api/categories/', views.get_categories, name='get_categories'),
    path('api/category_statistics/<int:user_id>/', views.get_category_statistics, name='get_category_statistics'),
    path('api/shannon_equitability/<int:user_id>/', views.get_shannon_equitability_by_category, name='get_shannon_equitability_by_category'),
    path('api/frequent_nouns/<int:user_id>/', views.get_frequent_nouns_by_category, name='get_frequent_nouns_by_category'),
    path('api/text_entropy_shannon_by_content/', views.text_entropy_shannon_by_content, name='text_entropy_shannon_by_content'),
    path("api/validate_text/", views.validate_text, name="validate_text"),
    path("api/fix_text_errors/", views.fix_selected_errors, name="fix_text_errors")
]
