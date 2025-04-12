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


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


if __name__ == "__main__":
    # # 在modelscope上下载Qwen2.5-VL模型到本地目录下
    # model_dir = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="./", revision="master")
    model_path = "./Qwen2.5-VL-7B-Instruct"

    # 使用Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)

    # 加载 Qwen2.5-VL-7B-Instruct
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    # ====================测试模式===================
    # 配置测试参数
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,  # 测试模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )

    # 获取测试模型
    val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2.5-VL-7B/checkpoint-56", config=val_config)

    # 读取测试数据
    with open("coco_2014/data_vl_test.json", "r") as f:
        test_dataset = json.load(f)

    test_image_list = []
    for item in test_dataset:
        input_image_prompt = item["conversations"][0]["value"]
        # 去掉前后的<|vision_start|>和<|vision_end|>
        origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        
        messages = [{
            "role": "user", 
            "content": [
                {
                "type": "image", 
                "image": origin_image_path
                },
                {
                "type": "text",
                "text": "COCO Yes:"
                }
            ]}]
        
        response = predict(messages, val_peft_model)
        messages.append({"role": "assistant", "content": f"{response}"})
        print(messages[-1])
