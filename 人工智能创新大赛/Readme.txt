目录结构：
|----code
|----datasets
       |----test
       |----train
             |----NORMAL
             |----PNEUMONIA
       |----val
             |----NORMAL
             |----PNEUMONIA
|----models

code文件夹存在所有代码文件、模型结果；
datasets文件夹存放test、train、val数据集；
models文件夹存放预训练模型；
submission.csv为提交结果；
肺部X光图像分类项目说明书.docx为说明书；
视频_代码讲解.mov为代码讲解视频。

code下其他文件内容：
loss_acc_inception_v3.csv、loss_acc_resnet50.csv、loss_acc_vgg16.csv为三种预训练模型loss、accuracy训练过程的变化；
Pneumonia_resnet50_model.h5、Pneumonia_inception_v3_model.h5、Pneumonia_vgg16_model.h5为训练后模型；
Pneumonia_resnet50_feature.csv为训练集图片经训练后模型提取出的特征序列；
Pneumonia_resnet50_test_feature.csv为测试集图片经训练后模型提取出的特征序列；
Pneumonia_resnet50_results.csv为(xgboost/svm/rf/sigmoid)四种方法在test集特征序列上的分类结果；
submission.csv为最终test集上的预测概率。

在code下：
运行Pneumonia_resnet50.ipynb查看项目方案；
运行Pneumonia_inception_v3.py和Pneumonia_vgg16.py查看这两种预训练模型训练情况。
