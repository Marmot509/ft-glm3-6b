import json
import random

# Function to transform data format
def transform_data(data):
    return {
        "prompt": "请续写一首完整的歌词，开头为："+"，".join(data["lyric"][:3]),
        "response": "，".join(data["lyric"]) + "。"
    }

# Function to read data from the file
def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Function to split data into train, validation, and test sets
def split_data(data, train_ratio=0.77, validation_ratio=0.18, test_ratio=0.05):
    random.shuffle(data)  # Shuffling the data

    total = len(data)
    train_end = int(train_ratio * total)
    validation_end = train_end + int(validation_ratio * total)

    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    test_data = data[validation_end:]

    return train_data, validation_data, test_data

# Function to transform and save data
def save_transformed_data(data, file_name):
    transformed_data = []
    for item in data:
        transformed_item = transform_data(item)
        transformed_data.append(transformed_item)

    with open(file_name, 'w', encoding='utf-8') as file:
        for item in transformed_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

# File path of the original data
file_path = 'lyric_data_for_CL_no_id.jsonl'

# Reading the data from the file
original_data = read_data(file_path)

# Splitting the data
train_data, validation_data, test_data = split_data(original_data)
mini_val_data = validation_data[:100]

# Saving the transformed data
save_transformed_data(train_data, 'train_data.jsonl')
save_transformed_data(validation_data, 'validation_data.jsonl')
save_transformed_data(test_data, 'test_data.jsonl')
save_transformed_data(mini_val_data, 'mini_val_data.jsonl')
