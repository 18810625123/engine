# Generated by Django 3.0.5 on 2020-05-03 06:14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0007_auto_20200503_1413'),
    ]

    operations = [
        migrations.RenameField(
            model_name='img',
            old_name='label_id',
            new_name='label',
        ),
        migrations.RenameField(
            model_name='imglabel',
            old_name='label_id',
            new_name='label',
        ),
        migrations.RenameField(
            model_name='model',
            old_name='category_id',
            new_name='category',
        ),
    ]