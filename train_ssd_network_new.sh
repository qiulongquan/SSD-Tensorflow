<< COMMENTOUT
# 通过加载预训练好的vgg16模型，对“voc07trainval+voc2012”进行训练
# 通过checkpoint_exclude_scopes指定哪些层的参数不需要从vgg16模型里面加载进来
# 通过trainable_scopes指定哪些层的参数是需要训练的，未指定的参数保持不变,若注释掉此命令，所有的参数均需要训练
DATASET_DIR=/home/doctorimage/kindlehe/common/dataset/VOC0712/
TRAIN_DIR=.././log_files/log_finetune/train_voc0712_20170816_1654_VGG16/
CHECKPOINT_PATH=../checkpoints/vgg_16.ckpt
 
python3 ../train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \      #训练生成模型的存放路径
    --dataset_dir=${DATASET_DIR} \  #数据存放路径
    --dataset_name=pascalvoc_2007 \ #数据名的前缀
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \      #加载的模型的名字
    --checkpoint_path=${CHECKPOINT_PATH} \  #所加载模型的路径
    --checkpoint_model_scope=vgg_16 \   #所加载模型里面的作用域名
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \  #每60s保存一下日志
    --save_interval_secs=600 \  #每600s保存一下模型
    --weight_decay=0.0005 \     #正则化的权值衰减的系数
    --optimizer=adam \          #选取的最优化函数
    --learning_rate=0.001 \     #学习率
    --learning_rate_decay_factor=0.94 \ #学习率的衰减因子
    --batch_size=24 \   
    --gpu_memory_fraction=0.9   #指定占用gpu内存的百分比

原文链接：https://blog.csdn.net/liuyan20062010/article/details/78905517
COMMENTOUT

current_directory=`pwd`
echo $current_directory
DATASET_DIR=$current_directory"/tfrecords/"
echo $DATASET_DIR
TRAIN_DIR=$current_directory"/logs/"
echo $TRAIN_DIR


CHECKPOINT_PATH=$current_directory"/checkpoints/vgg_16.ckpt"
echo $CHECKPOINT_PATH

python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32 \
    --gpu_memory_fraction=0.8