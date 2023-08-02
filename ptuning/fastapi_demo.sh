PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python3 fastapi_demo.py \
    --model_name_or_path /root/zhucanxiang/model/chatglm2-6b \
    --ptuning_checkpoint output/apa-chatglm2-6b-pt-128-2e-2/checkpoint-4000 \
    --pre_seq_len $PRE_SEQ_LEN

