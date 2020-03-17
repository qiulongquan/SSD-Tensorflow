# 把这个sh文件放在SSD-Tensorflow下面，和VOC2007同一层然后终端运行sh
# 这个是用来检测模型精确度的shell,原始程序有点问题，会产生TypeError: _variable_v2_call() got an unexpected keyword argument 'collections'这个错误，参考下面的解决方法
# https://github.com/balancap/SSD-Tensorflow/issues/321
<< COMMENTOUT
used 1.13rc1,
change tf_extended/metrics.py line 51 to variables.VariableV1
add a function


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

and flatten the parameter of slim.evaluation.evaluate_once as "eval_op=flatten(list(names_to_updates.values())),"
COMMENTOUT

# Model	  Training data	                   Testing data	mAP
# SSD-300 VGG-based	VOC07+12+COCO trainval	VOC07 test	0.817 
# SSD-512 VGG-based	VOC07+12+COCO trainval	VOC07 test	0.837
# model的下载地址
# SSD-300   https://drive.google.com/file/d/0B0qPCUZ-3YwWT1RCLVZNN3RTVEU/view
# SSD-512   https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view

# 验证数据test set的下载地址
# http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit

current_directory=`pwd`
echo $current_directory
DATASET_DIR=$current_directory"/tfrecords/"
echo $DATASET_DIR
EVAL_DIR=$current_directory"/logs/"
echo $EVAL_DIR

if [[ ! -d "$EVAL_DIR" ]]; then
	mkdir "$EVAL_DIR"
	echo "文件夹不存在,已经创建文件夹"
else
	echo "文件夹存在"
fi

#CHECKPOINT_PATH=$current_directory"/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt"
CHECKPOINT_PATH=$current_directory"/checkpoints/ssd_300_vgg.ckpt"
echo $CHECKPOINT_PATH
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1