current_directory=`pwd`
echo $current_directory
DATASET_DIR=$current_directory"/tfrecords/"
echo $DATASET_DIR
TRAIN_DIR=$current_directory"/logs/"
echo $TRAIN_DIR

if [[ ! -d "$TRAIN_DIR" ]]; then
	mkdir "$TRAIN_DIR"
	mkdir "${TRAIN_DIR}"+"/eval"
	echo "文件夹不存在,已经创建文件夹"
else
	echo "文件夹存在"
fi

#CHECKPOINT_PATH=$current_directory"/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt"
CHECKPOINT_PATH=$current_directory"/checkpoints/ssd_300_vgg.ckpt"
echo $CHECKPOINT_PATH

EVAL_DIR=${TRAIN_DIR}"/eval/"
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500