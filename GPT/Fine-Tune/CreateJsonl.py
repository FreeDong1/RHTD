import csv
import json


def csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, 'r') as csv_file, open(output_jsonl, 'w') as jsonl_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            # 提取特征值（除了最后一列）
            features = ','.join(row[:-1])
            # 提取标签（最后一列）
            label = row[-1]

            # 构建messages结构
            messages = [
                {
                    "role": "system",
                    "content": "You are a classifier for HTTP/3 traffic. Classify the given traffic features into one of five categories: 0, 1, 2, 3, or 4."
                },
                {
                    "role": "user",
                    "content": f"Traffic features: {features}"
                },
                {
                    "role": "assistant",
                    "content": label
                }
            ]

            # 写入JSONL文件
            json_line = json.dumps({"messages": messages})
            jsonl_file.write(json_line + '\n')


# 使用函数
csv_to_jsonl('packet_50.csv', 'packet_http3_training.jsonl')