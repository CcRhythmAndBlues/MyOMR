import tensorflow as tf
import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, transform, io
from sklearn.utils import shuffle
import keras
# from keras.utils import np_utils

# 这个是测试版本
# 对原本的会有细节的修改
#
# 已经修改的差不多了，就做正式版本好了

# 可能会有的坑/待解决的问题:
# (已解决)1、数据集尚且不规范，(1)图片大小不一致(2)不清楚这套轮子要怎样规格的图片
# (已解决)1.5、对图像转灰度，不知道有没有坑，姑且能跑
# 2、对于五线谱的特征识别，采用正方形5*5卷积核不知道会不会出问题
#    因为特征是黑点在五线谱的哪个位置，东西都是一样的，所以这样的卷积核不知道能不能成功识别出来
#    不行的话可能得选择用高度大于五线谱高度的矩形卷积核来卷积了

# 修改记录：
# 1、调整了灰度(预处理函数)以及通道数
# 2、修改了种类数目为7 (暂定高音区 从小字一组C到小字一组B)
# 3、修改了代码块数为 (原128)
# 4、对图片做了裁剪resize

# 注:数据集理论上应该要有很多，但是做省略，只有1.印刷体图片
#    (放弃)训练应该有高音区和低音区两个
#    (放弃)训练集包括升降符

# 预处理
# 获取对应路径文件夹下的图片数据

# 设定为数据集所在目录
# os.chdir('E://OMR')
os.chdir('../OMR')

def load_data(dir_path):
    images = []  # 新建一个空列表用于存放图片数集
    labels = []  # 新建一个空列表用于存放标签数集

    lab = os.listdir(dir_path)
    n = 0
    for l in lab:
        img = os.listdir(dir_path + l)  # img为对应路径下的文件夹
        for i in img:
            img_path = dir_path + l + '/' + i  # 是的话获取图片路径
            labels.append(int(n))  # 将外循环的迭代数n存于labels中
            images.append(skimage.io.imread(img_path))  # 读取对应路径图像存放于datasets中
        n += 1
    return images, labels  # 返回的images内的图片存放顺序与实际文件夹中存放的顺序不同


images_20, labels_20 = load_data('./Training/')  # 训练集

images_test_20, labels_test_20 = load_data('./Test/')  # 测试集


# 使用列表推导式完成图像的批量裁剪
def cut_image(images, h, w):
    new_images = [skimage.transform.resize(I, (h, w)) for I in images]
    return new_images


# 预处理数据函数（数组化，乱序）
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

# 参数定义
n_classes = 7  # 数据的类别数

# 训练集数据预处理
train_x, train_y = prepare_data(images_20, labels_20, n_classes)

# 测试数据集与标签的数组化和乱序
test_x, test_y = prepare_data(images_test_20, labels_test_20, n_classes)
print(len(test_x))
print(len(test_y))
# for i in test_x:
#     i = np.reshape(i, (100, 40))
#     plt.imshow(i, cmap=plt.get_cmap('gray'))
#     plt.show()
# 配置神经网络的参数
depth_in = 1  # 图片的通道数
batch_size = 28  # 训练块的大小

kernel_h = 7
kernel_w = 7  # 卷积核尺寸
dropout = 0.8  # dropout的概率
depth_out1 = 64  # 第一层卷积的卷积核个数
depth_out2 = 128  # 第二层卷积的卷积核个数
# print("train_x.shape:", train_x.shape)
# 注意这里只取[1]是因为默认输入图像是一个正方形
# 后续算真实输入尺寸的时候用image_size*image_size
# image_size = train_x.shape[1]  # 图片尺寸1  对应正方形输入 100*100
image_size = train_x.shape[1] * train_x.shape[2]  # 图片尺寸2 对应图片尺寸为100*64
n_sample = train_x.shape[0]  # 训练样本个数
t_sample = test_x.shape[0]  # 测试样本个数

# feed给神经网络的图像数据类型与shape,四维，第一维训练的数据量，第二、三维图片尺寸，第四维图像通道数
x = tf.placeholder(tf.float32, [None, 100, 40, depth_in], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name='y')  # feed到神经网络的标签数据的类型和shape
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout的placeholder(解决过拟合)
flatten = int((image_size / 16) * depth_out2)  # 用于扁平化处理的参数经过两层卷积池化后的图像大小*第二层的卷积核个数
# 定义各卷积层和全连接层的权重变量
Weights = {"con1_w": tf.Variable(tf.random_normal([kernel_h, kernel_w, depth_in, depth_out1])),
           "con2_w": tf.Variable(tf.random_normal([kernel_h, kernel_w, depth_out1, depth_out2])),
           "fc_w1": tf.Variable(tf.random_normal([int((image_size / 16) * depth_out2), 1024])),
           "fc_w2": tf.Variable(tf.random_normal([1024, 512])),
           "out": tf.Variable(tf.random_normal([512, n_classes]))}

# 定义各卷积层和全连接层的偏置变量
bias = {"conv1_b": tf.Variable(tf.random_normal([depth_out1])),
        "conv2_b": tf.Variable(tf.random_normal([depth_out2])),
        "fc_b1": tf.Variable(tf.random_normal([1024])),
        "fc_b2": tf.Variable(tf.random_normal([512])),
        "out": tf.Variable(tf.random_normal([n_classes]))}

# 定义卷积层的生成函数
def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
# 定义池化层的生成函数
def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding="SAME")
# 定义卷积神经网络生成函数
def conv_net(x, Weights, Biases, dropout):
    # 卷积层 1
    ConLayer1 = conv2d(x, Weights['con1_w'], Biases['conv1_b'])  # 当前尺寸100*40*64
    # 经过最大池化层1后的 shape：50*20*64
    ConLayer1 = maxpool2d(ConLayer1, 2)

    # 卷积层 2
    ConLayer2 = conv2d(ConLayer1, Weights['con2_w'], Biases['conv2_b'])  # 当前尺寸50*20*128
    # 经过最大池化层2后的 shape：25*10*128
    ConLayer2 = maxpool2d(ConLayer2, 2)

    # 全连接层 2
    global flatten
    # 扁平化处理.flatten层，在池化层到全连接层的步骤。
    fla = tf.reshape(ConLayer2, [-1, flatten])
    fully_Conn1 = tf.add(tf.matmul(fla, Weights['fc_w1']), Biases['fc_b1'])
    fully_Conn1 = tf.nn.relu(fully_Conn1)  # 激活函数使用relu

    print(fully_Conn1.shape)

    # 全连接层 1
    # 计算公式：输出y = 输入x * 权值k + 偏置b
    fully_Conn2 = tf.add(tf.matmul(fully_Conn1, Weights['fc_w2']), Biases['fc_b2'])
    fully_Conn2 = tf.nn.relu(fully_Conn2)  # 激活

    # print(fully_Conn2.shape)

    # 使用Dropout层来防止预测数据过拟合,丢失率初设0.8
    fully_Conn2 = tf.nn.dropout(fully_Conn2, dropout)
    # 输出预测参数
    pred = tf.add(tf.matmul(fully_Conn2, Weights['out']), Biases['out'], name="prediction")
    return pred


# 有很多优化器,之间效果性能会不一样
# 优化预测准确率
prediction = conv_net(x, Weights, bias, keep_prob)  # 生成卷积神经网络
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))  # 交叉熵损失函数
optimizer = tf.train.AdamOptimizer(0.0009).minimize(cross_entropy)  # 自适应梯度优化器以及学习率
# 跑过 根本用不了 得不到损失函数
# optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 跑过  损失值下降很慢  准确率上升很慢
# optimizer=tf.train.AdagradOptimizer(0.001).minimize(cross_entropy) # 叫不上名的优化器以及学习率

# 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 训练块数据生成器
def gen_small_data(inputs, batch_size):
    i = 0
    while True:
        small_data = inputs[i:(batch_size + i)]
        i += batch_size
        yield small_data


# 初始会话并开始训练过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    test_x = test_x[0:400]
    test_y = test_y[0:400]
    test_feed = {x: test_x, y: test_y, keep_prob: 0.8}

    result = {}
    for i in range(300):
        train_x, train_y = prepare_data(images_20, labels_20, n_classes)  # 重新预处理数据
        train_x = gen_small_data(train_x, batch_size)  # 生成图像块数据
        train_y = gen_small_data(train_y, batch_size)  # 生成标签块数据
        for j in range(int(n_sample / batch_size) + 1):
            # 每一次迭代 这里都会更新权值
            x_ = next(train_x)
            y_ = next(train_y)
            # 准备验证数据
            validate_feed = {x: x_, y: y_, keep_prob: 0.8}
            if i % 1 == 0:
                sess.run(optimizer, feed_dict=validate_feed)
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: x_, y: y_, keep_prob: 0.8})
                print("Epoch:", '%04d' % (i + 1), "cost=", "{:.5f}".format(loss), "Training accuracy",
                      "{:.4f}".format(acc))
                # 下面是自己加的，方便迭代中查看准确率的提升
                if i % 10 == 1 and i != 0 and j == 1:
                    brand = np.argmax(test_y, axis=1)
                    y1 = sess.run(prediction, feed_dict=test_feed)
                    test_classes = np.argmax(y1, axis=1)
                    for k in range(len(brand)):
                        if brand[k] != test_classes[k]:
                            print(brand[k], ":", test_classes[k])
                    TestAccuracy = sess.run(accuracy, feed_dict=test_feed)
                    print('Testing Accuracy:', TestAccuracy)
                    result[i] = TestAccuracy

    print('Optimization Completed')
    # 准备测试数据
    brand = np.argmax(test_y, axis=1)
    print("test_y      :", brand)
    y1 = sess.run(prediction, feed_dict=test_feed)
    test_classes = np.argmax(y1, axis=1)
    print("test_classes:", test_classes)

    for k in range(len(test_classes)):
        if brand[k] != test_classes[k]:
            print(brand[k], "-", test_classes[k], end=" ")

    print('Testing Accuracy:', sess.run(accuracy, feed_dict=test_feed))
    print('result:', result)


    # saver = tf.train.Saver()
    # os.chdir("E://pycharm project/tensorflow_test/OMR2")
    # saver.save(sess, './MyModel/model2.ckpt')
'''
以下为运行结果，记录不同次的识别结果好观察准确率的提升，以及是否有个别样例难以识别


'''
'''

Optimization Completed
test_y      : [0 2 6 2 0 5 4 4 3 1 5 5 4 6 2 3 6 1 0 1 3]
test_classes: [0 3 0 3 0 5 4 4 3 1 5 6 4 6 2 3 6 1 0 1 3]
2 - 3 6 - 0 2 - 3 5 - 6 Testing Accuracy: 0.85714287
result: {1: 0.0952381, 21: 0.5714286, 41: 0.7619048, 61: 0.61904764, 81: 0.71428573
, 101: 0.7619048, 121: 0.71428573, 141: 0.71428573, 161: 0.7619048, 181: 0.71428573
, 201: 0.8095238, 221: 0.8095238, 241: 0.61904764, 261: 0.7619048, 281: 0.7619048
, 301: 0.71428573, 321: 0.85714287, 341: 0.71428573, 361: 0.7619048, 381: 0.9047619
, 401: 0.71428573, 421: 0.71428573, 441: 0.85714287, 461: 0.8095238, 481: 0.7619048
, 501: 0.85714287, 521: 0.85714287, 541: 0.8095238, 561: 0.85714287, 581: 0.7619048
, 601: 0.8095238, 621: 0.71428573, 641: 0.8095238, 661: 0.85714287, 681: 0.8095238
, 701: 0.9047619, 721: 0.71428573, 741: 0.8095238, 761: 0.8095238, 781: 0.8095238
, 801: 0.7619048, 821: 0.85714287, 841: 0.85714287, 861: 0.71428573, 881: 0.7619048
, 901: 0.71428573, 921: 0.7619048, 941: 0.85714287, 961: 0.6666667, 981: 0.85714287}

'''
'''
3*3卷积核 迭代300次
Optimization Completed
test_y      : [6 3 1 6 2 1 5 2 2 5 0 1 3 4 0 5 4 6 4 3 0]
test_classes: [5 5 1 5 3 1 5 3 5 5 0 1 3 5 0 5 3 5 4 3 0]
6 - 5 3 - 5 6 - 5 2 - 3 2 - 3 2 - 5 4 - 5 4 - 3 6 - 5 Testing Accuracy: 0.7619048
result: {1: 0.14285715, 11: 0.0952381, 21: 0.52380955, 31: 0.52380955, 41: 0.6666667, 51: 0.71428573, 61: 0.61904764, 
71: 0.71428573, 81: 0.52380955, 91: 0.8095238, 101: 0.8095238, 111: 0.71428573, 121: 0.8095238, 131: 0.7619048, 
141: 0.7619048, 151: 0.71428573, 161: 0.71428573, 171: 0.71428573, 181: 0.61904764, 191: 0.71428573, 201: 0.71428573, 
211: 0.8095238, 221: 0.61904764, 231: 0.85714287, 241: 0.85714287, 251: 0.8095238, 261: 0.7619048, 271: 0.71428573, 
281: 0.7619048, 291: 0.7619048}


'''
'''
5*5卷积核 迭代200次
Optimization Completed
test_y      : [4 6 5 4 6 4 3 3 0 1 2 0 2 5 2 0 5 1 1 3 6]
test_classes: [4 6 5 4 5 4 3 3 0 1 3 0 2 5 3 1 6 1 1 3 6]
6 - 5 2 - 3 2 - 3 0 - 1 5 - 6 Testing Accuracy: 0.7619048
result: {1: 0.2857143, 11: 0.47619048, 21: 0.7619048, 31: 0.52380955, 41: 0.7619048, 51: 0.6666667, 61: 0.6666667, 
71: 0.71428573, 81: 0.9047619, 91: 0.6666667, 101: 0.7619048, 111: 0.8095238, 121: 0.8095238, 131: 0.6666667, 
141: 0.8095238, 151: 0.7619048, 161: 0.7619048, 171: 0.7619048, 181: 0.71428573, 191: 0.8095238}


'''
'''
7*7卷积核 迭代200次
Optimization Completed
test_y      : [4 5 6 4 2 2 3 1 0 5 0 3 2 1 6 3 6 0 4 5 1]
test_classes: [4 5 6 4 3 3 3 1 0 5 0 3 2 1 6 3 6 0 4 5 1]
2 - 3 2 - 3 Testing Accuracy: 0.7619048
result: {1: 0.23809524, 11: 0.33333334, 21: 0.6666667, 31: 0.5714286, 41: 0.71428573, 51: 0.7619048, 61: 0.95238096, 
71: 0.8095238, 81: 0.85714287, 91: 0.7619048, 101: 0.8095238, 111: 0.9047619, 121: 0.71428573, 131: 0.8095238, 
141: 0.85714287, 151: 0.8095238, 161: 0.71428573, 171: 0.8095238, 181: 0.71428573, 191: 0.9047619}


'''