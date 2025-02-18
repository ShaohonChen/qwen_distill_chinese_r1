import time
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
import peft
import torch
from threading import Thread

# 加载 Qwen2.5-7B-Instruct 模型和分词器
# model_name = "/mnt/work/weights/Qwen2__5-0__5B-Instruct"  # origin
model_name = "/mnt/work/output/Qwen-05B-Chinese-R1-Distill"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
try:
    model = peft.PeftModel.from_pretrained(model, model_name)
    print("---use peft model---")
except:
    print("---not found peft model---")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = "npu:0"  # for Ascend NPU
# device = "cuda:0"  # for Nvidia GPU
device = "cpu"  # for CPU
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


def qwen_chat_stream(message, history):
    # 组合历史记录形成上下文
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        }
    ]
    for h in history:
        messages += [
            {"role": "user", "content": h[0]},
            {"role": "assistant", "content": h[1]},
        ]
    messages.append({"role": "user", "content": message})

    streamer = TextIteratorStreamer(
        pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        text_inputs=messages, max_new_tokens=512, streamer=streamer
    )

    # 在后台线程中运行生成任务
    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        if (
            generated_text == tokenizer.pad_token
            or generated_text == tokenizer.eos_token
        ):
            return
        yield generated_text  # 逐步返回生成内容


# 创建 Gradio 对话界面
demo = gr.ChatInterface(
    qwen_chat_stream,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch()
