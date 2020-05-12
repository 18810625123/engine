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
        old = '%s/101obj/%s'%(base_path,name)
        new = '%s/101obj/%s.jpg'%(base_path,i)
        os.rename(old, new)
        i+=1
# rename_file_name()

# 修改文件名称
def rename_dir_name():
    for dir in os.listdir('../../static/images/101obj'):
        i = dir.split('_')[0]
        i = '%.4i'%int(i)
        dir2 = dir.split('_')[1:]
        dir2 = '_'.join(dir2)
        print(dir)
        os.rename('../../static/images/101obj/'+dir, '../../static/images/101obj/'+i+'_'+dir2)
# rename_dir_name()

def resize_img_and_save_method(old_file_path, new_file_path):
    img = cv2.imread(old_file_path)
    if img.shape[0] >= img.shape[1]:
        r = img.shape[1] / img.shape[0]
        x = int(224 * r)
        bo_a = int((224 - x) / 2)
        bo_b = 224 - x - bo_a
        resize_img = cv2.resize(img, (x, 224))
        resize_img = cv2.copyMakeBorder(resize_img, top=0, bottom=0, left=bo_a, right=bo_b,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        r = img.shape[0] / img.shape[1]
        x = int(224 * r)
        bo_a = int((224 - x) / 2)
        bo_b = 224 - x - bo_a
        resize_img = cv2.resize(img, (224, x))
        resize_img = cv2.copyMakeBorder(resize_img, top=bo_a, bottom=bo_b, left=0, right=0,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(new_file_path, resize_img)
    return True

# 修改目录中所有图片 只限当前目录
def resize_img_dir_and_save_method(dir_path):
    i=0
    for file in os.listdir(dir_path):
        i+=1
        file_path = dir_path+'/'+file
        new_path = dir_path+'/'+str(i)+'.jpg'
        if resize_img_and_save_method2(file_path,new_path) == True: os.remove(file_path)

# 修改目录中所有图片 迭代子目录一层
def resize_img_dir_and_save_each_method(dir_path):
    for dir in os.listdir(dir_path):
        print(dir)
        resize_img_dir_and_save_method(dir_path+'/'+dir)
# resize_img_dir_and_save_each_method('../../static/images/101obj')

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
# img_strenthen_each('../../static/images/101obj')

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


# import tensorflow as tf
# import numpy as np
# Dense = tf.keras.layers.Dense
# Conv2D = tf.keras.layers.Conv2D
# Flatten = tf.keras.layers.Flatten
# MaxPooling2D = tf.keras.layers.MaxPooling2D
# Dropout = tf.keras.layers.Dropout
# ZeroPadding2D = tf.keras.layers.ZeroPadding2D
# BatchNormalization=tf.keras.layers.BatchNormalization
# AveragePooling2D = tf.keras.layers.AveragePooling2D
# Sequential = tf.keras.models.Sequential
# EarlyStopping=tf.keras.callbacks.EarlyStopping
# ModelCheckpoint=tf.keras.callbacks.ModelCheckpoint
# TensorBoard=tf.keras.callbacks.TensorBoard
# Adam=tf.keras.optimizers.Adam
seed = 7
np.random.seed(seed)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2  # 这里加入l2正则化目的是为了防止过拟合
from keras.utils.vis_utils import plot_model
import keras.backend as K

# 训练模型并保存
# resnet做加和操作，因此用add函数，
# googlenet以及densenet做filter的拼接，因此用concatenate
# add和concatenate的区别参考链接：https://blog.csdn.net/u012193416/article/details/79479935
class ResNet():
    @staticmethod
    def residual_module(x, K, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5,
                        bnMom=0.9):  # 结构参考Figure 12.3右图,引入了shortcut概念，是主网络的侧网络
        """
        The residual module of the ResNet architecture.
        Parameters:
            x: The input to the residual module.
            K: The number of the filters that will be learned by the final CONV in the bottlenecks.最终卷积层的输出
            stride: Controls the stride of the convolution, help reduce the spatial dimensions of the volume *without*
                resorting to max-pooling.
            chanDim: Define the axis which will perform batch normalization.
            reduce: Cause not all residual module will be responsible for reducing the dimensions of spatial volums -- the
                red boolean will control whether reducing spatial dimensions (True) or not (False).是否降维，
            reg: Controls the regularization strength to all CONV layers in the residual module.
            bnEps: Controls the ε responsible for avoiding 'division by zero' errors when normalizing inputs.防止BN层出现除以0的异常
            bnMom: Controls the momentum for the moving average.
        Return:
            x: Return the output of the residual module.
        """

        # The shortcut branch of the ResNet module should be initialize as the input(identity) data.
        shortcut = x

        # The first block of the ResNet module -- 1x1 CONVs.
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        act1 = Activation("relu")(bn1)
        # Because the biases are in the BN layers that immediately follow the convolutions, so there is no need to introduce
        # a *second* bias term since we had changed the typical CONV block order, instead of using the *pre-activation* method.
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)  # filter=K*0.25,kernel_size=(1,1),stride=(1,1)

        # The second block of the ResNet module -- 3x3 CONVs.
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # The third block of the ResNet module -- 1x1 CONVs.
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # If we would like to reduce the spatial size, apply a CONV layer to the shortcut.
        if reduce:  # 是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # Add together the shortcut (shortcut branch) and the final CONV (main branch).
        x = add([conv3,
                 shortcut])  # 这个与googlenet的concatenate函数不同，add函数做简单加法，concatenate函数做横向拼接.该函数仅仅将shortcut部分和非shortcut部分相加在一起

        # Return the addition as the output of the Residual module.
        return x  # f(x)输出结果=conv3+shortcut

    @staticmethod
    def train():
        #初始化输入形状为“最后的通道”和通道维度本身。
        lr=0.0001
        bnEps=0.00002
        bnMom = 0.9
        input_shape = (224,224,3)
        chanDim = -1
        class_n=101
        stages = [3, 4, 6]
        epochs=30
        batch_size=40
        filters=[6, 12, 24, 48]
        # Set the input and apply BN layer.
        input = Input(shape=input_shape)
        # 使用BN层作为第一层，作为附加的归一化层。 在这里第一层使用BN层而不是使用conv，这样可以替代取平均值的操作
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

        x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(lr))(x)

        # 循环阶段的数量(块名)。
        for i in range(0, len(stages)):  # 每阶段的遍历
            # Initialize the stride, then apply a residual module used to reduce the spatial size of the input volume.

            # If this is the first entry in the stage, we’ll set the stride to (1, 1), indicating that no downsampling
            # should be performed. However, for every subsequent stage we’ll apply a residual module with a stride of (2, 2),
            # which will allow us to decrease the volume size.
            stride = (1, 1) if i == 0 else (2, 2)

            # Once we have stacked stages[i] residual modules on top of each other, our for loop brings us back up to here
            # where we decrease the spatial dimensions of the volume and repeat the process.
            x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, reduce=True, bnEps=bnEps,bnMom=bnMom)  # 进行降维

            # Loop over the number of layers in the stage.
            for j in range(0, stages[i] - 1):  # 每层的遍历
                # Apply a residual module.
                print(i)
                x = ResNet.residual_module(x, filters[i + 1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)  # 不进行降维

        # After stacked all the residual modules on top of each other, we would move to the classifier stage.
        # Apply BN=>ACT=>POOL, in order to avoid using dense/FC layers we would instead apply Global Averager POOL to reduce
        # the volume size to 1x1xclass_n.
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier.
        x = Flatten()(x)
        x = Dense(class_n, kernel_regularizer=l2(lr))(x)
        x = Activation("softmax")(x)
        # Construct the model.
        model = Model(input, x, name="ResNet")
        model.summary()  # 输出网络结构信息
        # exit()
        rms = tf.keras.optimizers.RMSprop(learning_rate=lr)
        adam = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

        train_x, train_y, train_labels, test_x, test_y, test_labels = get_train_x_y(dir_path='../../static/images/101obj', class_n=class_n, train_rate=0.75)
        train_y = keras.utils.to_categorical(train_y, class_n)
        test_y = keras.utils.to_categorical(test_y, class_n)
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        # 评估1
        score = model.evaluate(test_x, test_y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # 保存
        model.save('../../static/models/resnet-%s-%s-%s-%s-%s-%s.h5' % (class_n, epochs, int(score[1] * 100), score[0], batch_size, 'rms'))

model = ResNet.train()

