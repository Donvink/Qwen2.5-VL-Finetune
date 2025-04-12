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
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    train_ds = Dataset.from_json("coco_2014/data_vl_train.json")
    train_dataset = train_ds.map(process_func)

    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )

    # 获取LoRA模型
    peft_model = get_peft_model(model, config)

    # 配置训练参数
    args = TrainingArguments(
        output_dir="./output/Qwen2.5-VL-7B",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
            
    # # 设置SwanLab回调
    # swanlab_callback = SwanLabCallback(
    #     project="Qwen2.5-VL-finetune",
    #     experiment_name="qwen2-vl-coco2014",
    #     config={
    #         "model": "https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct",
    #         "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
    #         "github": "https://github.com/datawhalechina/self-llm",
    #         "prompt": "COCO Yes: ",
    #         "train_data_number": len(train_data),
    #         "lora_rank": 64,
    #         "lora_alpha": 16,
    #         "lora_dropout": 0.1,
    #     },
    # )

    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        # callbacks=[swanlab_callback],
    )

    # 开启模型训练
    trainer.train()

    # exit()

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

        # test_image_list.append(swanlab.Image(origin_image_path, caption=response))

    # swanlab.log({"Prediction": test_image_list})

    # # 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
    # swanlab.finish()
