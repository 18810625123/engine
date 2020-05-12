import cv2
import os
from PIL import Image
import numpy as np
import random
import pdb
import datetime

import tensorflow as tf
import keras
import pdb
import numpy as np
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential
EarlyStopping=tf.keras.callbacks.EarlyStopping
ModelCheckpoint=tf.keras.callbacks.ModelCheckpoint
TensorBoard=tf.keras.callbacks.TensorBoard
Adam=tf.keras.optimizers.Adam
seed = 7
np.random.seed(seed)

# 修改文件名称
def rename_file_name():
    i=1
    for name in images:
        print(i)
        old = '%s/101obj-360-360/%s'%(base_path,name)
        new = '%s/101obj-360-360/%s.jpg'%(base_path,i)
        os.rename(old, new)
        i+=1
# rename_file_name()

# 修改文件名称
def rename_dir_name():
    for dir in os.listdir('../../static/images/101obj-360-360'):
        i = dir.split('_')[0]
        i = '%.4i'%int(i)
        dir2 = dir.split('_')[1:]
        dir2 = '_'.join(dir2)
        print(dir)
        os.rename('../../static/images/101obj-360-360/'+dir, '../../static/images/101obj-360-360/'+i+'_'+dir2)
# rename_dir_name()

def resize_img_and_save_method(img, new_file_path):
    shape=[360,360]
    if img.shape[0] >= img.shape[1]:
        r = img.shape[1] / img.shape[0]
        x = int(360 * r)
        bo_a = int((360 - x) / 2)
        bo_b = 360 - x - bo_a
        resize_img = cv2.resize(img, (x, 360))
        resize_img = cv2.copyMakeBorder(resize_img, top=0, bottom=0, left=bo_a, right=bo_b,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        r = img.shape[0] / img.shape[1]
        x = int(360 * r)
        bo_a = int((360 - x) / 2)
        bo_b = 360 - x - bo_a
        resize_img = cv2.resize(img, (360, x))
        resize_img = cv2.copyMakeBorder(resize_img, top=bo_a, bottom=bo_b, left=0, right=0,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(new_file_path, resize_img)
    return True

# 修改目录中所有图片 只限当前目录
def resize_img_dir_and_save_method(dir_path):
    for file in os.listdir(dir_path):
        file_path = dir_path+'/'+file
        img = cv2.imread(file_path)
        os.remove(file_path)
        resize_img_and_save_method(img, file_path)

# 修改目录中所有图片 迭代子目录一层
def resize_img_dir_and_save_each_method(dir_path):
    for dir in os.listdir(dir_path):
        print(dir)
        resize_img_dir_and_save_method(dir_path+'/'+dir)
# resize_img_dir_and_save_each_method('../../static/images/101obj-360-360')
# exit()
# 数据增强
def img_strenthen(dir_path):
    fs=os.listdir(dir_path)
    if fs.__len__() < 100:
        for f in fs:
            img = cv2.imread(dir_path+'/'+f)
            gauss_img = cv2.GaussianBlur(img, (11, 11), 0)
            cv2.imwrite(dir_path+'/%s_gauss.jpg'%(f.split('.')[0]), gauss_img)

# 数据增强each
def img_strenthen_each(dir_path):
    for dir in os.listdir(dir_path):

        img_strenthen(dir_path+'/'+dir)
# img_strenthen_each('../../static/images/101obj-360-360')

# 从目录获取训练数据与标签 （要求目录名称是标签）
def get_train_x_y(dir_path,class_n=101,train_rate=0.75,dtype=np.float16):
    train_y=[]
    train_x=[]
    test_y = []
    test_x = []
    i=1
    dirs=os.listdir(dir_path)
    test_labels = {}
    train_labels = {}
    for dir in dirs:
        if i>class_n: break
        y=int(dir.split('_')[0])
        print(i,dir,y)
        i+=1
        label='_'.join(dir.split('_')[1:])
        test_labels[y]={
            'sum':0,
            'label': label
        }
        train_labels[y] = {
            'sum': 0,
            'label': label
        }
        fs = os.listdir(dir_path+'/'+dir)
        train_sum = int(train_rate * fs.__len__())
        # 随机打乱顺序
        random.shuffle(fs)
        for f in fs[:train_sum]:
            img = cv2.imread(dir_path+'/'+dir+'/'+f)
            arr=np.array(img)
            train_x.append(arr)
            train_y.append(y)
            train_labels[y]['sum'] += 1
        for f in fs[train_sum:]:
            img = cv2.imread(dir_path+'/'+dir+'/'+f)
            arr = np.array(img)
            test_x.append(arr)
            test_y.append(y)
            test_labels[y]['sum']+=1
    if train_x.__len__() != train_y.__len__(): raise Exception('train数据长度不一致')
    if test_x.__len__() != test_y.__len__(): raise Exception('test数据长度不一致')
    return np.array(train_x,dtype=dtype),np.array(train_y,dtype=np.int8),train_labels,np.array(test_x,dtype=dtype),np.array(test_y,dtype=np.int8),test_labels

def top5(model, test_x, test_y, test_labels):
    results = model.predict(test_x)
    errors = {}
    ok = 0
    for i in range(results.__len__()):
        y = np.argmax(test_y[i])
        y_ = np.argmax(results[i])
        if y != y_:
            if not errors.get(y): errors[y] = []
            errors[y].append(y_)
        else:
            ok += 1
    for y in errors:
        print(y, test_labels[y]['label'], errors[y].__len__(),
              round(errors[y].__len__() / test_labels[y]['sum'] * 100, 2))
    print('all_acc:', ok / results.__len__())

# 训练模型并保存
def train_vgg16_model_and_save(data_path,input_shape,arr,class_n,lr,epochs,batch_size,dense_size):
    model = Sequential()
    for i in range(arr.__len__()):
        a=arr[i]
        print(a)
        if i==0:
            model.add(Conv2D(a[0], a[1], strides=(1, 1), input_shape=input_shape, padding='same', activation='relu', kernel_initializer='uniform'))
        else:
            model.add(Conv2D(a[0], a[1], strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    if dense_size:
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(0.3))
    model.add(Dense(class_n, activation='softmax'))
    rms = tf.keras.optimizers.RMSprop(learning_rate=lr)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    model.summary()

    # 获取数据集
    train_x,train_y,train_labels,test_x,test_y,test_labels=get_train_x_y(dir_path=data_path, class_n=class_n, train_rate=0.75)
    train_y=keras.utils.to_categorical(train_y, class_n)
    test_y=keras.utils.to_categorical(test_y, class_n)

    # 训练
    tensorboard = TensorBoard(r'E:\tensorboard', write_images=True)
    checkpoint = ModelCheckpoint(r'E:\models\%s.h5'%epochs, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 评估1
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 保存
    model.save('../../static/models/%s-%s-%s-%s-%s-%s-%s-%s.h5' % (class_n,epochs,int(score[1] * 100),score[0],batch_size,arr,'rms',lr))

    # 评估2
    train_x, train_y, train_labels, test_x, test_y, test_labels = get_train_x_y(dir_path=data_path,class_n=class_n, train_rate=0.75)
    test_y = keras.utils.to_categorical(test_y, class_n)
    top5(model, test_x, test_y, test_labels)

def load_model_train(h5_path):
    model=tf.keras.models.load_model(h5_path, custom_objects=None, compile=True)
    class_n=101
    epochs = 150
    batch_size = 100
    train_x,train_y,train_labels,test_x,test_y,test_labels=get_train_x_y(dir_path='../../static/images/101obj-360-360',class_n=class_n, train_rate=0.75)
    # one_hot
    train_y = keras.utils.to_categorical(train_y, class_n)
    test_y = keras.utils.to_categorical(test_y, class_n)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('../../static/models/vgg-%s-%s-%s-%s-%s-%s-%s-%s.h5' % (class_n, epochs, int(score[1] * 100), score[0], batch_size, 'old_train', dense_size,'rms',lr))
    top5(model, test_x, test_y, test_labels)

def load_model_test(h5_path):
    model=tf.keras.models.load_model(h5_path, custom_objects=None, compile=True)
    class_n=101
    train_x,train_y,train_labels,test_x,test_y,test_labels=get_train_x_y(dir_path='../../static/images/101obj-360-360',class_n=class_n, train_rate=0.75)
    train_y = keras.utils.to_categorical(train_y, class_n)
    test_y = keras.utils.to_categorical(test_y, class_n)
    # 评估
    print(train_x.shape,train_y.shape)
    print(test_x.shape,test_y.shape)
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    top5(model, test_x,test_y,test_labels)

# load_model_test('../../static/models/101-100-89-1.0977249406707041-60-[[16, (7, 7)], [32, (5, 5)], [32, (3, 1)], [64, (3, 3)], [128, (3, 3)], [256, (3, 3)]]-rms-0.0005.h5')


train_vgg16_model_and_save('../../static/images/101obj',(224,224,3),[
        [16,(7,7)],
        [32,(3,1)],
        [64,(3,3)],
        [128,(3,3)],
        [256,(3,3)],
    ],class_n=101,lr=0.00005,epochs=10,batch_size=100,dense_size=200)
