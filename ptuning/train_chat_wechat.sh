PRE_SEQ_LEN=128
LR=1e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file WechatData/train.json \
    --validation_file WechatData/train.json \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path ../weight/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-2048-1e-4 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --pre_seq_len 2048 \
    --quantization_bit 4

