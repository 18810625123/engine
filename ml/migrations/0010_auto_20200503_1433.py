# Generated by Django 3.0.5 on 2020-05-03 06:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0009_auto_20200503_1430'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='imglabel',
            options={},
        ),
        migrations.AlterModelTable(
            name='img',
            table='img',
        ),
        migrations.AlterModelTable(
            name='imglabel',
            table='img_lable',
        ),
        migrations.AlterModelTable(
            name='model',
            table='model',
        ),
        migrations.AlterModelTable(
            name='modelcategory',
            table='model_category',
        ),
    ]
