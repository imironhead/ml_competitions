PATH_DATASET_RAW=/home/ironhead/datasets/ntire2019_raw/RealSR/
PATH_DATASET_TRAINING_SOURCE=$(PATH_DATASET_RAW)Train_LR/
PATH_DATASET_TRAINING_TARGET=$(PATH_DATASET_RAW)Train_GT/
PATH_DATASET_VALIDATION_SOURCE=$(PATH_DATASET_RAW)Validation_LR/
PATH_DATASET_VALIDATION_TARGET=$(PATH_DATASET_RAW)Validation_GT/
PATH_DATASET_TESTING_SOURCE=$(PATH_DATASET_RAW)Test_LR/

PATH_PROJECT=/home/ironhead/projects/ml_competitions/ntire2019_image_super_resolution/


deploy:
	scp makefile *.py $(HOST):$(PATH_PROJECT)


EXP_NAME=baseline_all
EXP_TRAIN_PATCH_SIZE=128
EXP_TRAIN_CHECKPOINT_PATH=$(PATH_PROJECT)checkpoints/$(EXP_NAME)/
EXP_TRAIN_SUMMARY_PATH=gs://hecate-research-ml-results/ntire2019_sisr/summary_final/$(EXP_NAME)/
EXP_TRAIN_TRAINING_BATCH_SIZE=36
EXP_TRAIN_TRAINING_BATCH_SIZE_MULTIPLIER=5
EXP_TRAIN_TRAINING_IMAGE_SIZE=$(EXP_TRAIN_PATCH_SIZE)
EXP_TRAIN_TRAINING_STOP_STEP=100000000
EXP_TRAIN_TRAINING_LOSS_WEIGHT_SHIFTING_BEGIN=10000
EXP_TRAIN_TRAINING_LOSS_WEIGHT_SHIFTING_END=20000
EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_BEGIN=20000
EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_CYCLE=10000
EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_RATE=0.7
EXP_TRAIN_TRAINING_LEARNING_RATE=0.0001
EXP_TRAIN_TRAINING_LRS_STOP_STEP=0
EXP_TRAIN_TRAINING_LRS_INITIAL_RATE=0.0
EXP_TRAIN_TRAINING_LRS_INCREASING_RATE=0.0
EXP_TRAIN_VALIDATION_CYCLE=1000
EXP_TRAIN_VALIDATION_BATCH_SIZE=32
EXP_TRAIN_VALIDATION_BATCH_SIZE_MULTIPLIER=1
EXP_TRAIN_VALIDATION_IMAGE_SIZE=$(EXP_TRAIN_PATCH_SIZE)
EXP_TRAIN_TESTING_CYCLE=2000
EXP_TRAIN_TESTING_BATCH_SIZE=32
EXP_TRAIN_TESTING_PATCH_SIZE=$(EXP_TRAIN_PATCH_SIZE)
EXP_TRAIN_TESTING_OVERLAPPING_SIZE=32
EXP_TRAIN_TESTING_RESULT_PATH=./results/$(EXP_NAME).zip
EXP_TRAIN_SAVE_CYCLE=5000

train:
	mkdir -p $(EXP_TRAIN_CHECKPOINT_PATH)
	mkdir -p ./results/
	python3 experiment_train.py \
		--checkpoint_path=$(EXP_TRAIN_CHECKPOINT_PATH) \
		--summary_path=$(EXP_TRAIN_SUMMARY_PATH) \
		--training_image_source_dir_path=$(PATH_DATASET_TRAINING_SOURCE) \
		--training_image_target_dir_path=$(PATH_DATASET_TRAINING_TARGET) \
		--training_batch_size=$(EXP_TRAIN_TRAINING_BATCH_SIZE) \
		--training_batch_size_multiplier=$(EXP_TRAIN_TRAINING_BATCH_SIZE_MULTIPLIER) \
		--training_image_size=$(EXP_TRAIN_TRAINING_IMAGE_SIZE) \
		--training_stop_step=$(EXP_TRAIN_TRAINING_STOP_STEP) \
		--training_loss_weight_shifting_begin=$(EXP_TRAIN_TRAINING_LOSS_WEIGHT_SHIFTING_BEGIN) \
		--training_loss_weight_shifting_end=$(EXP_TRAIN_TRAINING_LOSS_WEIGHT_SHIFTING_END) \
		--training_learning_rate_decay_begin=$(EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_BEGIN) \
		--training_learning_rate_decay_cycle=$(EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_CYCLE) \
		--training_learning_rate_decay_rate=$(EXP_TRAIN_TRAINING_LEARNING_RATE_DECAY_RATE) \
		--training_learning_rate=$(EXP_TRAIN_TRAINING_LEARNING_RATE) \
		--training_lrs_stop_step=$(EXP_TRAIN_TRAINING_LRS_STOP_STEP) \
		--training_lrs_initial_rate=$(EXP_TRAIN_TRAINING_LRS_INITIAL_RATE) \
		--training_lrs_increasing_rate=$(EXP_TRAIN_TRAINING_LRS_INCREASING_RATE) \
		--validation_image_source_dir_path=$(PATH_DATASET_VALIDATION_SOURCE) \
		--validation_image_target_dir_path=$(PATH_DATASET_VALIDATION_TARGET) \
		--validation_cycle=$(EXP_TRAIN_VALIDATION_CYCLE) \
		--validation_batch_size=$(EXP_TRAIN_VALIDATION_BATCH_SIZE) \
		--validation_batch_size_multiplier=$(EXP_TRAIN_VALIDATION_BATCH_SIZE_MULTIPLIER) \
		--validation_image_size=$(EXP_TRAIN_VALIDATION_IMAGE_SIZE) \
		--testing_image_source_dir_path=$(PATH_DATASET_TESTING_SOURCE) \
		--testing_cycle=$(EXP_TRAIN_TESTING_CYCLE) \
		--testing_batch_size=$(EXP_TRAIN_TESTING_BATCH_SIZE) \
		--testing_patch_size=$(EXP_TRAIN_TESTING_PATCH_SIZE) \
		--testing_overlapping_size=$(EXP_TRAIN_TESTING_OVERLAPPING_SIZE) \
		--testing_result_path=$(EXP_TRAIN_TESTING_RESULT_PATH) \
		--save_cycle=$(EXP_TRAIN_SAVE_CYCLE)


test:
	mkdir -p ./results/
	python3 experiment_test.py \
		--checkpoint_path=./checkpoints/model.ckpt-130000 \
		--testing_image_source_dir_path=./Test_LR/ \
		--testing_patch_size=$(EXP_TRAIN_TESTING_PATCH_SIZE) \
		--testing_overlapping_size=$(EXP_TRAIN_TESTING_OVERLAPPING_SIZE) \
		--testing_result_path=./results/results.zip


search-learning-rate: EXP_TRAIN_SUMMARY_PATH=$(PATH_PROJECT)summary/lrs_$(EXP_NAME)/
search-learning-rate: EXP_TRAIN_TRAINING_LRS_STOP_STEP=10000
search-learning-rate: EXP_TRAIN_TRAINING_LRS_INITIAL_RATE=0.0
search-learning-rate: EXP_TRAIN_TRAINING_LRS_INCREASING_RATE=0.000001
search-learning-rate: train

