# export ASCEND_LAUNCH_BLOCKING=1   # for DEBUG
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # select used NPU

accelerate launch \
    --num_processes 8 \
    --main_process_port 25001 \
    --config_file configs/dpsp_z2.yaml \
    train_sft.py \
    --config configs/qwen2.5-0.5B-lora-sft.yaml
   