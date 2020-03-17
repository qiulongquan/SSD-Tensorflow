import os
import random

trainval_percent = 0.8
train_percent = 0.8
xmlfilepath = '/home/qiulongquan/github/SSD-Tensorflow/elephant/Annotations'
txtsavepath = '/home/qiulongquan/github/SSD-Tensorflow/elephant/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
print("tv={}".format(tv))
tr = int(tv * train_percent)
print("tr={}".format(tr))
trainval = random.sample(list, tv)
print("trainval len={}".format(len(trainval)))
train = random.sample(trainval, tr)
print("train len={}".format(len(train)))

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()