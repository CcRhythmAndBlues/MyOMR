import tensorflow as tf
import skimage
import numpy as np
from skimage import color, transform, io
from sklearn.utils import shuffle
import keras
import os
from playsound import playsound


'''
功能：载入模型，并对设置的图片进行识别 并播放

2019-12-17：
    终于可以跑,并且输出最可能的一个类型，原来前面问题在于获取tensor是用get_tensor_by_name而不是跟训练时一样创建
    但目前问题是 7个类型和数组的7个元素不能对应，因为当时输入是乱序的
    但我可以用训练数据进行测试，因为训练数据基本百分百准确率
2019-12-18：
    标签和下标对应：
	    C---[2]
	    D---[3]
	    E---[4]
	    F---[5]
	    G---[6]
	    A---[0]
	    B---[1]
2019-12-20：
    找到音乐素材，利用playsound成功播放
    目前的问题:速度很慢
'''
# restored的地址， import_meta_graph 则根据下面地址加.meta
Model_PATH = "./MyModel/model2.ckpt"
# 要预测的图片存放地址
ImgDir_PATH = './MyImg/'
# 钢琴mp3文件路径
MusicDir_PATH = "./MyMusic/"
depth_in = 1  # 图片的通道数
n_classes = 7  # 数据的类别数

imgs = os.listdir(ImgDir_PATH)
img_PATH = ImgDir_PATH+imgs[0]
PitchNameList = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def load_data_one(img_path):
    images = []  # 新建一个空列表用于存放图片数集
    labels = []  # 新建一个空列表用于存放标签数集

    images.append(skimage.io.imread(img_path))
    labels.append(1)
    return images, labels
def load_data_many(dir_path):
    images = []  # 新建一个空列表用于存放图片数集
    labels = []  # 新建一个空列表用于存放标签数集

    n = 0
    imgs = os.listdir(dir_path)
    for i in imgs:
        # img_path = dir_path + "/" + i
        img_path = dir_path + i
        images.append(skimage.io.imread(img_path))
        labels.append(int(n))
        n += 1
    return images, labels
def cut_image(images, h, w):
    new_images = [skimage.transform.resize(I, (h, w)) for I in images]
    return new_images
def prepare_data(images, labels, n_classes):
    images64 = cut_image(images, 100, 40)  # 裁剪图片
    train_x = color.rgb2gray(np.array(images64))  # 转灰度
    train_x = np.reshape(train_x, (-1, 100, 40, 1))
    # train_x = np.array(images64)
    train_y = np.array(labels)
    indx = np.arange(0, train_y.shape[0])
    indx = shuffle(indx)
    train_x = train_x[indx]
    train_y = train_y[indx]
    train_y = keras.utils.to_categorical(train_y, n_classes)  # one-hot独热编码
    return train_x, train_y

# 这个获取参数的方法是错误的
# x = tf.placeholder(tf.float32, [None, 100, 40, depth_in])
# y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32)
def recognize(img_path):
    with tf.Session() as sess:
        # 恢复graph、再恢复ckpt
        saver = tf.train.import_meta_graph(Model_PATH+'.meta')
        saver.restore(sess, Model_PATH)

        # 写输入数据
        # 与训练时图像预处理操作相同
        graph = tf.get_default_graph()
        # 一次只载入一张图片
        images_test, labels_test = load_data_one(img_path)  # 测试集
        # images_test, labels_test = load_data_many(ImgDir_PATH)  # 测试集
        test_x, test_y = prepare_data(images_test, labels_test, n_classes)

        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        test_feed = {x: test_x, y: test_y, keep_prob: 0.8}

        prediction = graph.get_tensor_by_name("prediction:0")
        y1 = sess.run(prediction, feed_dict=test_feed)
        result = np.argmax(y1, axis=1)

        result_pitch_name = PitchNameList[result[0]]
        print("识别为：", result_pitch_name)
        playsound(MusicDir_PATH+result_pitch_name+".mp3")
    return result_pitch_name

if __name__ == '__main__':
    recognize(img_PATH)
