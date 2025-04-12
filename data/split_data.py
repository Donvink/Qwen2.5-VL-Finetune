import json

if __name__ == "__main__":
    # 处理数据集：读取json文件
    # 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
    train_json_path = "coco_2014/data_vl.json"
    with open(train_json_path, 'r') as f:
        data = json.load(f)
        train_data = data[:-50]
        test_data = data[-50:]

    with open("coco_2014/data_vl_train.json", "w") as f:
        json.dump(train_data, f)

    with open("coco_2014/data_vl_test.json", "w") as f:
        json.dump(test_data, f)

