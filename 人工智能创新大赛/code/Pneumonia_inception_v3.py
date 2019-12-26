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
from keras.applications.inception_v3 import InceptionV3, preprocess_input
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
from sklearn.metrics import accuracy_score
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

train_df = []

# 绑定正常图片路径及其标签
for img in normal_cases:
    train_df.append((str(img),0))

# 绑定肺炎图片路径及其标签
for img in pneumonia_cases:
    train_df.append((str(img), 1))

# 将所有训练数据路径及标签存储在dataframe中
train_df = pd.DataFrame(train_df, columns=['image', 'label'],index=None)

# 将训练数据打乱
train_df = train_df.sample(frac=1.).reset_index(drop=True)

# 统计每类图片的个数
cases_count = train_df['label'].value_counts()
print(cases_count)

plt.figure(figsize=(10,4))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()

# 展示部分图片
pneumonia_samples = (train_df[train_df['label']==1]['image'].iloc[30:35]).tolist()
normal_samples = (train_df[train_df['label']==0]['image'].iloc[30:35]).tolist()

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

# 图像增强
seq = iaa.OneOf([
    iaa.Fliplr(1), # 水平翻转
    iaa.Affine(rotate=20), # 旋转角度
    iaa.Multiply((1.2, 1.5))]) #亮度处理

# 设置输入图片大小
img_rows, img_cols, img_channels = 299,299,3
# 设置batch size
batch_size=16

def data_generator(data, batch_size):

    n = len(data)
    nb_batches = int(np.ceil(n/batch_size))

    # 数据索引
    indices = np.arange(n)
    
    # 定义输送给神经网络的数组
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)
    
    while True:
        # 随机打乱
        np.random.shuffle(indices)
            
        for i in range(nb_batches):
            # 获取下一组数据的索引
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            
            # 生成下一组数据
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = seq.augment_image(img)
                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                label = data.iloc[idx]["label"]
                
                batch_data[j] = img
                batch_labels[j] = label
            
            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels

# 获取训练数据
train_data_gen = data_generator(train_df, batch_size)

def read_images(images, label):
    """
    将原始图片处理为所需要维度的数组，并对应其标签
    """
    data = []
    for img in images:
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
        data.append(img)
    
    labels = [label]*len(data)
    data = np.array(data).astype(np.float32)
    data = preprocess_input(data)
    return data, labels

def prepare_data(data_dir):
    """
    生成需要输送给神经网络的数据
    """
    normal_cases_dir = data_dir / 'NORMAL'
    pneumonia_cases_dir = data_dir / 'PNEUMONIA'

    normal_cases = list(normal_cases_dir.glob('*.jpeg'))
    pneumonia_cases = list(pneumonia_cases_dir.glob('*.jpeg'))
    print(f"Found {len(normal_cases)} normal cases and {len(pneumonia_cases)} pneumonia_cases")
    
    normal_cases_data, normal_cases_labels = read_images(normal_cases, 0)
    pneumonia_cases_data, pneumonia_cases_labels = read_images(pneumonia_cases, 1)
    
    # 判定数据个数与标签个数是否相等
    assert len(normal_cases_data) == len(normal_cases_labels), "You had one job!"
    assert len(pneumonia_cases_data) == len(pneumonia_cases_labels), "You can't get it right, can you?"
    
    data = np.vstack((normal_cases_data, pneumonia_cases_data))
    labels = np.array((normal_cases_labels + pneumonia_cases_labels)).astype(np.float32)
    
    return data, labels

# 生成验证集数据及其标签
validation_data, validation_labels = prepare_data(val_dir)
print(f"Number of validation images: {len(validation_data)} and labels: {len(validation_labels)}")
print(validation_data.shape, validation_labels.shape)

# 生成测试集数据
test_datas=list(test_dir.glob('*.jpeg'))
print(f"Found {len(test_datas)} test_datas")

ImageId=[str(x).split('\\')[-1].split('.')[0] for x in test_datas]
submission=pd.DataFrame(ImageId,columns=['ImageId'])
submission.to_csv('submission.csv',index=False)

test_data = []
for img in test_datas:
    img = cv2.imread(str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
    test_data.append(img)
test_data = np.array(test_data).astype(np.float32)
test_data = preprocess_input(test_data)
print("Number of samples in test data: ", len(test_data))
print(test_data.shape)

# 将预训练模型和自定义模型拼接在一起
def get_fine_tuning_model(base_model, feature_model,top_model, inputs, learning_type=None):
    if learning_type=='transfer_learning':
        # 预训练模型参数不参与训练
        print("Doing transfer learning")
        K.set_learning_phase(0)
        base_model.trainable = False
        base_res = base_model(inputs)
        features = feature_model(base_res)
        outputs = top_model(features)
    else:
        # 放开预训练模型参数，训练之
        print("Doing fine-tuning")
        base_model.trainable = True
        base_res = base_model(inputs)
        features = feature_model(base_res)
        outputs = top_model(features)
    return Model(inputs, outputs)

# 获取基础的预训练模型（InceptionV3）
base_model = InceptionV3(input_shape=(img_rows, img_cols, img_channels), 
                       weights='imagenet', 
                       include_top=False, 
                       pooling='avg')

# 自定义特征提取层
feature_inputs = Input(shape=base_model.output_shape, name='feature_model_input')
x = Dense(1024, activation='relu', name='fc1')(feature_inputs)
x = Dropout(0.5,name='drop')(x)
x = Dense(512, activation='relu', name='fc2')(feature_inputs)
x = Dropout(0.5,name='drop')(x)
feature_model = Model(feature_inputs, x, name='feature_model')
# 自定义输出层
top_inputs = Input(shape=feature_model.output_shape, name='top_model_input')
outputs = Dense(1, activation='sigmoid', name='fc3')(top_inputs)
top_model = Model(top_inputs, outputs, name='top_model')

# 生成整体模型
inputs = Input(shape=(img_rows, img_cols, img_channels))
model = get_fine_tuning_model(base_model, feature_model,top_model, inputs, "transfer_learning")
model.summary()

# focal loss 
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

optimizer = RMSprop(0.0001)
model.compile(loss=focal_loss(), optimizer=optimizer, metrics=['accuracy'])

'''
# 设置earlystopping
es = EarlyStopping(patience=5, restore_best_weights=True)
chkpt = ModelCheckpoint(filepath="resnet50_model", save_best_only=True)
'''
# 不设置earlystopping，并保存所有结果
from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./logs',  # 默认保存在当前文件夹下的logs文件夹之下
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

# 每轮迭代的训练个数
nb_train_steps = int(np.ceil(len(train_df)/batch_size))
# 迭代次数
nb_epochs=50

# 训练模型
history1 = model.fit_generator(train_data_gen, 
                              epochs=nb_epochs, 
                              steps_per_epoch=nb_train_steps, 
                              validation_data=(validation_data, validation_labels),
                              callbacks=[tbCallBack],
                              class_weight={0:1.0, 1:0.4})

model.save_weights("Pneumonia_inception_v3_model.h5")

# 获取训练过程中的loss和accuracy，并保存
train_loss = history1.history['loss']
train_acc = history1.history['acc']

valid_loss = history1.history['val_loss']
valid_acc = history1.history['val_acc']

loss_acc=pd.DataFrame(train_loss,columns=['train_loss'])
loss_acc.to_csv('loss_acc_inception_v3.csv',index=False)
loss_acc=pd.read_csv("loss_acc_inception_v3.csv")
loss_acc['train_acc']=train_acc
loss_acc['valid_loss']=valid_loss
loss_acc['valid_acc']=valid_acc
loss_acc.to_csv('loss_acc_inception_v3.csv',index=False)
