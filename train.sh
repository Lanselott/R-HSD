# Modify to your own path
/research/byu2/rchen/protoc_3.3/bin/protoc object_detection/protos/*proto --python_out=.

# Modify to your own path
export PYTHONPATH=$PYTHONPATH:/research/byu2/rchen/proj/cuhsd/hsd-od/slim/

# CUDA_VISIBLE_DEVICES: Choose GPUs, use 'nvidia-smi' to check number of GPUs you have
# --pipeline_config_path: Load configure path
# --train_dir: Model save path

CUDA_VISIBLE_DEVICES=$1   python ./object_detection/train.py \
--pipeline_config_path=./config/multi_branch_encoder_decoder_faster_rcnn_inception_v2_coco_dropout_l2regular_IOU_UPDATE_with_iou_loss_2x.config \
--train_dir=./result/

