# Generated by Django 3.0.5 on 2020-05-05 13:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0013_auto_20200505_2054'),
    ]

    operations = [
        migrations.AddField(
            model_name='model',
            name='size',
            field=models.IntegerField(default=0, max_length=20),
        ),
    ]
