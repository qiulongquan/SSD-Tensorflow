current_directory=`pwd`
echo $current_directory
DATASET_DIR=$current_directory"/tfrecords/"
echo $DATASET_DIR
TRAIN_DIR=$current_directory"/logs/"
echo $TRAIN_DIR


CHECKPOINT_PATH=$current_directory"/checkpoints/ssd_300_vgg.ckpt"
echo $CHECKPOINT_PATH

python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32