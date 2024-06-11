# Generated by Django 5.0.6 on 2024-05-27 11:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quanta', '0002_alter_text_category_alter_text_entropy_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='text',
            name='num_occurrences',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='text',
            name='num_tokens',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='text',
            name='top_token',
            field=models.CharField(default='', max_length=255),
        ),
        migrations.AddField(
            model_name='text',
            name='top_word',
            field=models.CharField(default='', max_length=255),
        ),
    ]
