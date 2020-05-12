# -*- coding: utf-8 -*-
import datetime
import time
import traceback
import pdb
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
from io import BytesIO

import os
# import tensorflow as tf
# Dense = tf.keras.layers.Dense
# Conv2D = tf.keras.layers.Conv2D
# Flatten = tf.keras.layers.Flatten
# MaxPooling2D = tf.keras.layers.MaxPooling2D
# Dropout = tf.keras.layers.Dropout
# Sequential = tf.keras.models.Sequential

import requests as req
from PIL import Image as PILImage

from django.core.paginator import Paginator, Page # 分页
from django.http import HttpResponse

from ml.models import *




def custom_success(data = {}, message = 'success !!'):
    return HttpResponse(json.dumps({
        'code': 0,
        'message': message,
        'data': data
    }))

def custom_error(code, message = 'server error !!', data = {}):
    return HttpResponse(json.dumps({
        'code': code,
        'message': message,
        'data': data
    }))


def get_params(request):
    if request.method == 'GET':
        return request.GET
    elif request.method == 'POST':
        return eval(request.body)



# 分页
def to_json(self):
    if self.__len__() > 0:
        return [model.to_json() for model in self]
    else:
        return []
Page.to_json = to_json


def pg(models, params):
    page = params['page'] if params['page'] else 1
    limit = params['limit'] if params['limit'] else 10
    return Paginator(models, limit).page(page)
