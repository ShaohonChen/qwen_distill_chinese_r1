# DeepSeek-r1数据蒸馏Qwen模型呢

## 数据集来源

* [中文基于满血DeepSeek-R1蒸馏数据集](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k)

## 模型来源

* [Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B)

* [Qwen2.5-3B](https://modelscope.cn/models/Qwen/Qwen2.5-3B)

## 环境安装

```bash
pip install -r requirements.txt
```

## 运行训练

运行如下命令下载数据集&模型（注意在脚本中修改下载路径）

```bash
bash scripts/download_model_and_datasets.sh
```

在`configs/qwen2.5-0.5B-lora-sft.yaml`文件中设置模型&数据集路径。

使用如下命令开始运行Qwen0.5B蒸馏

```bash
python train_sft.py --config configs/qwen2.5-0.5B-lora-sft.yaml
```

使用如下命令开始运行Qwen0.5B蒸馏（8GPU，zero2）

```bash
bash scripts/train_sft_05BI_8npu.sh
```

## 运行gradio的案例

需要记得在代码中改一下模型地址

```bash
python demo.py
```