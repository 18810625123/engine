# Generated by Django 3.0.5 on 2020-05-07 10:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0015_auto_20200505_2203'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='model',
            name='category',
        ),
        migrations.DeleteModel(
            name='ModelCategory',
        ),
    ]