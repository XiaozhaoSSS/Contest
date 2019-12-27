import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
import time
color = sns.color_palette()

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 设置哈希种子
os.environ['PYTHONHASHSEED'] = '0'

# 设置随机种子
np.random.seed(111)

# 禁用多线程
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# 设置tensorflow的随机种子
tf.set_random_seed(111)

# 配置tensorflow session
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# 设置keras session
K.set_session(sess)

# 设置增强序列种子
aug.seed(111)

data_dir = Path('../datasets')       # 数据集路径
train_dir = data_dir / 'train'       # 训练集路径
val_dir = data_dir / 'val'           # 验证集路径
test_dir = data_dir / 'test'         # 测试集路径


# 获取训练集中正常数据和肺炎数据的路径
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'

# 获取所有图片的路径列表
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

train_data = []

# 绑定正常图片路径及其标签
for img in normal_cases:
    train_data.append((str(img),0))

# 绑定肺炎图片路径及其标签
for img in pneumonia_cases:
    train_data.append((str(img), 1))

# 将所有训练数据路径及标签存储在dataframe中
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# 将训练数据打乱
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# 训练数据前5行
train_data.head()

# 统计每类图片的个数
cases_count = train_data['label'].value_counts()
print(cases_count)

plt.figure(figsize=(10,4))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()

# 展示部分图片
pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[30:35]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[30:35]).tolist()

samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

f, ax = plt.subplots(2,5, figsize=(20,6))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()

# 获取验证集图片路径
normal_cases_dir = val_dir / 'NORMAL'
pneumonia_cases_dir = val_dir / 'PNEUMONIA'
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# 生成验证集数据及标签
valid_data = []
valid_labels = []

# 将原始图片处理为所需要维度的数组
for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)       
for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)

print("Total number of validation examples: ", valid_data.shape)
print("Total number of labels:", valid_labels.shape)

print(normal_cases)

# 图像增强
seq = iaa.OneOf([
    iaa.Fliplr(1), # 水平翻转
    iaa.Affine(rotate=20), # 旋转角度
    iaa.Multiply((1.2, 1.5))]) #亮度处理


def data_gen(data, batch_size):
    n = len(data)
    steps = n//batch_size
    
    # 定义输送给神经网络的数组
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)

    # 数据索引
    indices = np.arange(n)
    
    i =0
    while True:
        np.random.shuffle(indices)

        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # 转化为独热编码
            encoded_label = to_categorical(label, num_classes=2)
            # 读入图片，转化为所需要的大小
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))
            
            # 检查是否是灰度图片
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # 将图片转为RGB模式
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 归一化
            orig_img = img.astype(np.float32)/255.
            
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            
            # 图像增强
            if label==0 and count < batch_size-2:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2
            
            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0

def build_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model

model =  build_model()
model.summary()

# 打开VGG16预训练模型
f = h5py.File('../models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# 对某些层参数赋予预训练模型的参数值

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
model.summary() 

opt = Adam(lr=0.0001, decay=1e-5)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

'''
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='Pneumonia_vgg16_model', save_best_only=True, save_weights_only=True)
'''
from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./logs/vgg16',  # 默认保存在当前文件夹下的logs文件夹之下
histogram_freq=0,
batch_size=32,
write_graph=True,  #默认是True，默认是显示graph的。
write_grads=False,
write_images=False,
embeddings_freq=0,
embeddings_layer_names=None,
embeddings_metadata=None,
embeddings_data=None,
update_freq='batch')

batch_size = 4
nb_epochs = 50

# 生成训练数据
train_data_gen = data_gen(data=train_data, batch_size=batch_size)

# 每轮迭代的训练次数
nb_train_steps = train_data.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))

# 训练模型
history1 = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(valid_data, valid_labels),callbacks=[tbCallBack],
                               class_weight={0:1.0, 1:0.4})


model.save_weights("Pneumonia_vgg16_model.h5")

train_loss = history1.history['loss']
train_acc = history1.history['acc']

valid_loss = history1.history['val_loss']
valid_acc = history1.history['val_acc']

loss_acc=pd.DataFrame(train_loss,columns=['train_loss'])
loss_acc.to_csv('loss_acc_vgg16.csv',index=False)
loss_acc=pd.read_csv("loss_acc_vgg16.csv")
loss_acc['train_acc']=train_acc
loss_acc['valid_loss']=valid_loss
loss_acc['valid_acc']=valid_acc
loss_acc.to_csv('loss_acc_vgg16.csv',index=False)