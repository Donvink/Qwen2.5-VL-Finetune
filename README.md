# Qwen2.5-VL-Finetune

[教程文档](https://blog.csdn.net/sinat_16020825/article/details/147163785?spm=1001.2014.3001.5501)

## 1. 环境配置

1. 确保你的电脑上至少有一张英伟达显卡，并已安装好了CUDA环境。
2. 安装Python（版本>=3.9）以及能够调用CUDA加速的PyTorch。
3. 安装与Qwen2.5-VL微调相关的第三方库，可以使用以下命令：

```
pip install modelscope transformers sentencepiece accelerate datasets peft swanlab qwen-vl-utils pandas
```


## 2. Prepare Datas

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

## 3. Train

Single GPU:

```bash
CUDA_VISIBLE_DEVICES="2" accelerate launch train_with_lora_accel.py \
    --pretrained_model ./Qwen2.5-VL-7B-Instruct \
    --train_dataset_path ./coco_2014/data_vl_train.json \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --epochs 4 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./output/Qwen2.5-VL-7B
```

Multi-GPU:

```bash
accelerate launch --num_processes=4 train_with_lora_accel.py \
    --pretrained_model ./Qwen2.5-VL-7B-Instruct \
    --train_dataset_path ./coco_2014/data_vl_train.json \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --epochs 4 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./output/Qwen2.5-VL-7B-accel
```


## 4. Inference

```bash
python test.py
```
