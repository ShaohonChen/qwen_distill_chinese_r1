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
model_name = "/mnt/work/output/Qwen-05BI-Chinese-R1-Distill/"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
try:
    model = peft.PeftModel.from_pretrained(model, model_name)
    print("---use peft model---")
except:
    print("---not found peft model---")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# thinking chat_template
tokenizer.chat_template = """{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- messages[0][\'content\'] }}\n    {%- else %}\n        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}\n    {%- endif %}\n    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}\n    {%- else %}\n        {{- \'<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {{- \'<|im_start|>\' + message.role }}\n        {%- if message.content %}\n            {{- \'\\n\' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- \'\\n<tool_call>\\n{"name": "\' }}\n            {{- tool_call.name }}\n            {{- \'", "arguments": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \'}\\n</tool_call>\' }}\n        {%- endfor %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\nlet\\'s think step by step<|think_start|>\\n\' }}\n{%- endif %}\n"""

device = "npu:0"  # for Ascend NPU
# device = "cuda:0"  # for Nvidia GPU
# device = "cpu"  # for CPU
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


def qwen_chat_stream(message, history):
    # 组合历史记录形成上下文
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        }
    ]
    # for h in history:
    #     messages += [
    #         {"role": "user", "content": h[0]},
    #         {"role": "assistant", "content": h[1]},
    #     ]
    messages += history
    messages.append({"role": "user", "content": message})

    streamer = TextIteratorStreamer(
        pipe.tokenizer, skip_prompt=True, skip_special_tokens=False
    )
    generation_kwargs = dict(
        text_inputs=messages, max_new_tokens=1024, streamer=streamer
    )

    # 在后台线程中运行生成任务
    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

    generated_text = "### 开始思考\n"
    for new_text in streamer:
        if "<|think_end|>" in new_text:
            new_text.replace("<|think_end|>", "\n### 结束思考 END\n### 开始回答\n")
        if "<|im_end|>" in new_text:
            new_text.replace("<|im_end|>", "")
            break
        generated_text += new_text
        yield generated_text  # 逐步返回生成内容
    generated_text += new_text
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
