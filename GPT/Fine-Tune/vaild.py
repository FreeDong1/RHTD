import openai
import csv
import json
import time
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from openai import OpenAI

# ==================== 配置区域 ====================
# 显式设置你的API密钥
API_KEY = "xxx";
BASE_URL = "xxx"; 

# 模型配置
FINE_TUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:default::DEu2RQkI"  # 替换为你的微调模型名称
#all:ft:gpt-4.1-mini-2025-04-14:default::DEvIq4WZ,对应all_100-TEST
# 测试数据配置
TEST_CSV_PATH = "xxx"  # 测试数据文件路径
MAX_TEST_SAMPLES = None  # 最大测试样本数，设为None使用全部数据，或设为整数如100
SAMPLE_RATE = 1.0  # 采样率，1.0表示使用全部，0.5表示使用50%

# 输出配置
SAVE_PREDICTIONS = True  # 是否保存预测结果
PREDICTIONS_OUTPUT_PATH = "predictions.csv"  # 预测结果输出路径

# API调用配置
BATCH_SIZE = 5  # 批处理大小，避免API限流
REQUEST_DELAY = 1  # 请求延迟（秒），避免触发速率限制
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试等待时间（秒）
# =================================================

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def load_test_data(csv_path, max_samples=None, sample_rate=1.0):
    """
    加载测试数据
    """
    print(f"正在加载测试数据从: {csv_path}")

    features_list = []
    true_labels = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)

        for i, row in enumerate(csv_reader):
            # 采样逻辑
            if sample_rate < 1.0 and np.random.random() > sample_rate:
                continue

            # 提取特征（除了最后一列）
            features = ','.join(row[:-1])
            # 提取标签（最后一列）
            label = row[-1]

            features_list.append(features)
            true_labels.append(label)

            # 如果达到最大样本数，停止加载
            if max_samples and len(features_list) >= max_samples:
                break

    print(f"成功加载 {len(features_list)} 条测试数据")

    # 显示标签分布
    label_dist = Counter(true_labels)
    print("标签分布:")
    for label, count in sorted(label_dist.items()):
        print(f"  类别 {label}: {count} 条 ({count / len(true_labels) * 100:.2f}%)")

    return features_list, true_labels


def predict_single(features, retry_count=0):
    """
    单条预测
    """
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a classifier for HTTP/3 traffic. Classify the given traffic features into one of five categories: 0, 1, 2, 3, or 4. Only respond with the category number. Do not include any other text or explanation."},
                {"role": "user", "content": f"Traffic features: {features}"}
            ],
            temperature=0,
            max_tokens=10
        )

        predicted = response.choices[0].message.content.strip()

        # 清理预测结果，只保留数字
        predicted = ''.join(filter(str.isdigit, predicted))

        if predicted in ['0', '1', '2', '3', '4']:
            return predicted
        else:
            print(f"警告: 模型返回了无效的预测结果: '{predicted}'，将视为预测失败")
            return None

    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"预测失败，{RETRY_DELAY}秒后重试 ({retry_count + 1}/{MAX_RETRIES})... 错误: {e}")
            time.sleep(RETRY_DELAY)
            return predict_single(features, retry_count + 1)
        else:
            print(f"预测失败，已达最大重试次数: {e}")
            return None


def batch_predict(features_list, batch_size=5, delay=1):
    """
    批量预测
    """
    print(f"\n开始批量预测，共 {len(features_list)} 条数据，批次大小: {batch_size}")

    predictions = []
    failed_indices = []

    for i in range(0, len(features_list), batch_size):
        batch = features_list[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(features_list))))

        print(
            f"处理批次 {i // batch_size + 1}/{(len(features_list) - 1) // batch_size + 1} (样本 {i + 1}-{min(i + batch_size, len(features_list))})")

        for idx, features in zip(batch_indices, batch):
            predicted = predict_single(features)

            if predicted is not None:
                predictions.append((idx, predicted))
                print(f"  样本 {idx + 1}: 预测={predicted}")
            else:
                failed_indices.append(idx)
                print(f"  样本 {idx + 1}: 预测失败")

            time.sleep(delay)  # 避免触发API限流

    # 按原始顺序整理预测结果
    ordered_predictions = [None] * len(features_list)
    for idx, pred in predictions:
        ordered_predictions[idx] = pred

    print(f"\n批量预测完成。成功: {len(predictions)}/{len(features_list)}, 失败: {len(failed_indices)}")

    return ordered_predictions, failed_indices


def calculate_metrics(true_labels, predictions):
    """
    计算评估指标
    """
    # 过滤掉预测失败的样本
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [predictions[i] for i in valid_indices]

    if len(valid_true) == 0:
        print("错误: 没有有效的预测结果")
        return None, None, None

    # 转换为整数
    valid_true_int = [int(t) for t in valid_true]
    valid_pred_int = [int(p) for p in valid_pred]

    # 计算准确率
    accuracy = accuracy_score(valid_true_int, valid_pred_int)

    # 计算F1分数（宏平均和加权平均）
    f1_macro = f1_score(valid_true_int, valid_pred_int, average='macro')
    f1_weighted = f1_score(valid_true_int, valid_pred_int, average='weighted')

    # 为每个类别计算F1分数
    labels = sorted(set(valid_true_int))
    f1_per_class = f1_score(valid_true_int, valid_pred_int, labels=labels, average=None)

    return accuracy, f1_macro, f1_weighted, f1_per_class, labels, valid_true_int, valid_pred_int


def save_predictions(features_list, true_labels, predictions, output_path):
    """
    保存预测结果到CSV
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['样本序号', '特征(前50个字符)', '真实标签', '预测标签', '是否正确'])

        for i, (features, true_label, pred) in enumerate(zip(features_list, true_labels, predictions)):
            features_preview = features[:50] + '...' if len(features) > 50 else features
            is_correct = '是' if pred is not None and pred == true_label else '否' if pred is not None else '预测失败'
            writer.writerow([i + 1, features_preview, true_label, pred if pred else 'N/A', is_correct])

    print(f"预测结果已保存到: {output_path}")


def print_detailed_report(true_labels, predictions, labels, f1_per_class):
    """
    打印详细的评估报告
    """
    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)

    # 计算每个类别的统计信息
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [predictions[i] for i in valid_indices]

    valid_true_int = [int(t) for t in valid_true]
    valid_pred_int = [int(p) for p in valid_pred]

    # 使用sklearn的分类报告
    report = classification_report(valid_true_int, valid_pred_int,
                                   target_names=[f'类别 {l}' for l in labels],
                                   digits=4)
    print(report)

    # 打印混淆矩阵
    cm = confusion_matrix(valid_true_int, valid_pred_int, labels=labels)
    print("\n混淆矩阵:")
    print("预测→")
    print("真实↓")

    # 打印表头
    print("     ", end="")
    for l in labels:
        print(f"  {l}  ", end="")
    print()

    for i, l in enumerate(labels):
        print(f"  {l}  ", end="")
        for j in range(len(labels)):
            print(f"  {cm[i][j]:3d} ", end="")
        print()

    # 计算每个类别的准确率
    print("\n每个类别的准确率:")
    for i, l in enumerate(labels):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            print(f"  类别 {l}: {class_acc:.4f} ({cm[i, i]}/{np.sum(cm[i, :])})")

    # 计算失败样本分布
    failed_indices = [i for i, p in enumerate(predictions) if p is None]
    if failed_indices:
        failed_labels = [true_labels[i] for i in failed_indices]
        failed_dist = Counter(failed_labels)
        print(f"\n预测失败的样本分布 ({len(failed_indices)} 条):")
        for label, count in sorted(failed_dist.items()):
            print(f"  类别 {label}: {count} 条")


def test_model():
    """
    主测试函数
    """
    print("=" * 60)
    print("HTTP/3 流量分类模型评估")
    print("=" * 60)

    # 检查配置
    if API_KEY == "你的OpenAI API密钥":
        print("❌ 错误: 请先设置你的OpenAI API密钥")
        return

    if FINE_TUNED_MODEL == "ft:gpt-3.5-turbo-0613:personal::xxxxx":
        print("❌ 错误: 请先设置你的微调模型名称")
        return

    print(f"\n📊 配置信息:")
    print(f"  - 模型: {FINE_TUNED_MODEL}")
    print(f"  - 测试数据: {TEST_CSV_PATH}")
    print(f"  - 最大测试样本: {MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else '全部'}")
    print(f"  - 采样率: {SAMPLE_RATE}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 请求延迟: {REQUEST_DELAY}秒")

    # 加载测试数据
    print("\n📂 步骤1: 加载测试数据")
    features_list, true_labels = load_test_data(TEST_CSV_PATH, MAX_TEST_SAMPLES, SAMPLE_RATE)

    if len(features_list) == 0:
        print("错误: 没有加载到测试数据")
        return

    # 进行预测
    print("\n🤖 步骤2: 执行预测")
    predictions, failed_indices = batch_predict(features_list, BATCH_SIZE, REQUEST_DELAY)

    # 计算指标
    print("\n📈 步骤3: 计算评估指标")
    result = calculate_metrics(true_labels, predictions)

    if result[0] is not None:
        accuracy, f1_macro, f1_weighted, f1_per_class, labels, valid_true_int, valid_pred_int = result

        # 打印主要指标
        print("\n" + "=" * 60)
        print("主要评估指标")
        print("=" * 60)
        print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"F1分数 (Macro): {f1_macro:.4f}")
        print(f"F1分数 (Weighted): {f1_weighted:.4f}")

        # 打印详细报告
        print_detailed_report(true_labels, predictions, labels, f1_per_class)

    # 保存预测结果
    if SAVE_PREDICTIONS:
        print("\n💾 步骤4: 保存预测结果")
        save_predictions(features_list, true_labels, predictions, PREDICTIONS_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)


def quick_test():
    """
    快速测试函数：只测试少量样本
    """
    global MAX_TEST_SAMPLES, SAMPLE_RATE

    # 临时设置
    original_max = MAX_TEST_SAMPLES
    original_sample = SAMPLE_RATE

    MAX_TEST_SAMPLES = 20  # 只测试20条
    SAMPLE_RATE = 1.0

    print("\n🔍 执行快速测试 (20条样本)...")
    test_model()

    # 恢复原始设置
    MAX_TEST_SAMPLES = original_max
    SAMPLE_RATE = original_sample


if __name__ == "__main__":
    import sys

    print("HTTP/3 分类模型测试程序")
    print("1. 完整测试")
    print("2. 快速测试 (20条样本)")
    print("3. 退出")

    choice = "1"

    if choice == "1":
        test_model()
    elif choice == "2":
        quick_test()
    else:
        print("退出程序")
        sys.exit(0)