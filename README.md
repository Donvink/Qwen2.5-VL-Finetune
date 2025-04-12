# Qwen2.5-VL-Finetune

[教程文档](https://blog.csdn.net/sinat_16020825/article/details/147163785?spm=1001.2014.3001.5501)

## 1. 环境配置

1. 确保你的电脑上至少有一张英伟达显卡，并已安装好了CUDA环境。
2. 安装Python（版本>=3.9）以及能够调用CUDA加速的PyTorch。
3. 安装与Qwen2.5-VL微调相关的第三方库，可以使用以下命令：

```
pip install modelscope transformers sentencepiece accelerate datasets peft swanlab qwen-vl-utils pandas
```


## 2. 数据准备

依次执行：

```bash
python data/data2csv.py
```

```bash
python data/csv2json.py
```

```bash
python data/split_data.py
```

## 3. 开始微调

```bash
python train.py
```

## 4. 推理

```bash
python test.py
```
