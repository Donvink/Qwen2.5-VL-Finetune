import argparse
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json

from accelerate import Accelerator

accelerator = Accelerator()

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Pretrained model to use.")
    parser.add_argument("--train_dataset_path", type=str, default="coco_2014/data_vl_train.json", help="Train dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=4, help="Train epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=64, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/Qwen2.5-VL-7B-accel",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    args = parser.parse_args()

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.pretrained_model)

    # Load pretrained model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16,
    )
    # model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    train_ds = Dataset.from_json(args.train_dataset_path)
    train_dataset = train_ds.map(process_func)

    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # for train
        r=args.lora_rank,  # Lora rank
        lora_alpha=args.lora_alpha,  # Lora alaph
        lora_dropout=args.lora_dropout,  # Dropout
        bias="none",
    )

    # 获取LoRA模型
    peft_model = get_peft_model(model, config)

    # 配置训练参数
    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=args.epochs,
        save_steps=100,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        # gradient_checkpointing=True,
        report_to="none",
        bf16=True,
    )

    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    peft_model, trainer = accelerator.prepare(peft_model, trainer)

    # 开启模型训练
    trainer.train()
