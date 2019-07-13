# encoding:utf-8
from datasets import flowers
import tensorflow as tf
import pylab

slim = tf.contrib.slim

# flowers数据集目录
DATA_DIR = 'images_data/flowers/'

# 指定获取“validation”下的数据
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

#在session下读取数据，并用pylab显示图片
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #启动队列
    tf.train.start_queue_runners()
    image_batch,label_batch = sess.run([image, label])
    #显示图片
    pylab.imshow(image_batch)
    pylab.show()
