from django.db import models

class CustomUser(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)

    def __str__(self):
        return self.username

class Category(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.name

class Text(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    entropy = models.FloatField(default=0.0)
    shannon_value = models.FloatField(default=0.0)
    num_tokens = models.IntegerField(default=0)
    num_occurrences = models.IntegerField(default=0)
    top_token = models.CharField(max_length=255, default='')
    top_word = models.CharField(max_length=255, default='')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, related_name='texts', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.title

class Token(models.Model):
    text = models.ForeignKey(Text, on_delete=models.CASCADE, related_name='tokens')
    token = models.CharField(max_length=255)
    frequency = models.IntegerField()
    word_type = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.token} ({self.word_type})"
