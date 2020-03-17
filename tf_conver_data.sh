# 把这个sh文件放在SSD-Tensorflow下面，和VOC2007同一层然后终端运行sh
# 它会把VOC2007文件夹里面的图片转换成tfrecord文件放在 SSD-Tensorflow下面的tfrecords文件夹中
# voc2007的下载地址
# http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

current_directory=`pwd`
echo $current_directory
DATASET_DIR=$current_directory"/VOC2007/"
echo $DATASET_DIR
OUTPUT_DIR=$current_directory"/tfrecords"
echo $OUTPUT_DIR

if [[ ! -d "$OUTPUT_DIR" ]]; then
	mkdir "$OUTPUT_DIR"
	echo "文件夹不存在,已经创建文件夹"
else
	echo "文件夹存在"
fi


python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}