import json

def transform_and_split_data(original_file_path, train_path, val_path, test_path, train_ratio=0.9, val_ratio=0.05):
    # 读取并转换原始数据
    with open(original_file_path, 'r', encoding='utf-8') as file:
        original_data = json.load(file)

    transformed_data = []
    for item in original_data:
        context = item["instruction"]
        if item["input"]:
            context += " " + item["input"]
        target = item["output"]
        transformed_data.append({"context": context, "target": target})

    # 计算分割点
    total_size = len(transformed_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # 分割数据集
    train_data = transformed_data[:train_size]
    val_data = transformed_data[train_size:train_size + val_size]
    test_data = transformed_data[train_size + val_size:]

    # 保存数据到文件
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)
    with open(val_path, 'w', encoding='utf-8') as file:
        json.dump(val_data, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)

# 调用函数
transform_and_split_data('alpaca_data.json', 'train_data.json', 'val_data.json', 'test_data.json')
