#set hyperparameters
OUTPUT_ROOT_DIR=/work/mhessent/TextBrewer/examples/mnli_example/outputs/training
DATA_ROOT_DIR=/work/mhessent/TextBrewer/examples/mnli_example


accu=1
ep=3
lr=3
temperature=8
batch_size=32
length=128
torch_seed=1909

taskname='mnli'
NAME=${taskname}_base_lr${lr}e${ep}_bs${batch_size}_teacher
DATA_DIR=${DATA_ROOT_DIR}/MNLI
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}

mkdir -p $OUTPUT_DIR
model_config_json_file=TrainBertTeacher.json
cp jsons/${model_config_json_file} ${OUTPUT_DIR}/${model_config_json_file}.run


python3 -u main.trainer.py \
    --data_dir  $DATA_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 2 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --task_name ${taskname} \
    --fp16 \
    --model_config_json ${OUTPUT_DIR}/${model_config_json_file}.run
