from builtins import print

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob

# TensorFlow推奨フォーマット「TFRecord」の作成と読み込み方法
# https://www.tdi.co.jp/miso/tensorflow-tfrecord-01


mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


def test_show_img():
    directory_path = os.getcwd()
    img = Image.open(directory_path + "/mnistVisualize/0-7.jpg")
    img = img.convert('L')
    plt.figure("Image")
    plt.imshow(img, cmap='gray')
    plt.axis('on')
    plt.title("image")
    plt.show()


def matrix_to_image(imageMatrix, imageShape, dirName, labal):
    directory_path = os.getcwd()
    imageMatrix = imageMatrix * 255  # 画像データの値を0～255の範囲に変更する
    for i in range(0, imageMatrix.shape[0]):
        print("i={}".format(i))
        imageArray = imageMatrix[i].reshape(imageShape)
        outImg = Image.fromarray(imageArray)
        outImg = outImg.convert("L")  # グレースケール
        print("qiulongquan_path=",directory_path+os.sep+dirName + os.sep + str(i) + "-" + str(np.argmax(labal[i])))
        # print(labal[i])　　[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
        # np.argmax表示从数组中选择最大的数字的index返回，注意返回的是index不是数值
        outImg.save(directory_path+os.sep+dirName + os.sep + str(i) + "-" + str(np.argmax(labal[i])) + ".jpg", format="JPEG")


def CreateTensorflowReadFile(img_files, out_file):
    with tf.python_io.TFRecordWriter(out_file) as writer:
        for f in img_files:
            # ファイルを開く
            with Image.open(f).convert("L") as image_object:  # グレースケール
                image = np.array(image_object)

                height = image.shape[0]
                width = image.shape[1]
                image_raw = image.tostring()
                label = int(f[f.rfind("-") + 1: -4])  # ファイル名からラベルを取得

                example = tf.train.Example(features=tf.train.Features(feature={
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_object.tobytes()]))
                }))

            # レコード書込
            writer.write(example.SerializeToString())


matrix_to_image(mnist.test.images, imageShape=(28, 28), dirName="mnistVisualize", labal=mnist.test.labels)
test_show_img()
OUTPUT_TFRECORD_NAME = "test_tf_file.tfrecord"  # アウトプットするTFRecordファイル名
# 書き込み
files = glob.glob("mnistVisualize" + os.sep + "*.jpg")
# CreateTensorflowReadFile(files, OUTPUT_TFRECORD_NAME)