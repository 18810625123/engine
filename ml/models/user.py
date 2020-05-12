from django.db import models
from django.core import serializers
import datetime

class User(models.Model):
    email = models.CharField(max_length=50)
    uid = models.CharField(max_length=30)