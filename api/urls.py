"""HelloWorld URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path
from api.v1.model import ModelApi
from api.v1.img import ImgApi
from api.v1.img_label import ImgLabelApi
from api.v1.dataset import DatasetApi
import re

def get_lower_case_name(text):
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            lst.append("_")
        lst.append(char)
    return "".join(lst).lower()

api_classes = [ModelApi, ImgApi, ImgLabelApi, DatasetApi]
# 自动加载所有API
urls = []
for api_class in api_classes:
    arr = get_lower_case_name(api_class.__name__).split('_')
    arr.pop()
    model_path = "_".join(arr)
    for api_method in dir(api_class):
        if not re.match('^_', api_method):
            urls.append(path('api/v1/%s/%s' % (model_path, api_method), getattr(api_class, api_method)))
print(urls)
urlpatterns = urls

# urlpatterns = [
#     path('api/v1/model/search', ModelApi.search),
# ]
