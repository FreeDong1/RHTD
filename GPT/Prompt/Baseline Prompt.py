import openai
import csv
import time
import json
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from openai import OpenAI

# ==================== 配置区域 ====================
# 显式设置你的API密钥
API_KEY = "xx"  # 请替换为你的实际API密钥
BASE_URL = "xx"

# 模型配置
MODEL = "gpt-4.1-mini-2025-04-14"  # 或使用 gpt-3.5-turbo-16k 处理长上下文
# MODEL = "gpt-3.5-turbo-16k"  # 备选方案

# 文件路径配置
TRAIN_FILE_PATH = "xx"  # 训练示例文件（50条带标签数据）
TEST_FILE_PATH = "xx"  # 测试文件（带标签）
OUTPUT_FILE_PATH = "1_packet_only.csv"  # 输出结果文件
METRICS_OUTPUT_PATH = "metrics_packet_only.txt"  # 评估指标输出文件

# 提示词模板配置
USE_FEW_SHOT = True  # 是否在提示词中包含示例
MAX_EXAMPLES_PER_CLASS = 5  # 每类最多选取的示例数量
SHOW_RAW_PACKET_SIZES = False  # 是否显示原始包大小值（True显示原始值，False显示统计信息）

# API调用配置
BATCH_SIZE = 5  # 批处理大小
REQUEST_DELAY = 1  # 请求延迟（秒）
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试等待时间（秒）
# =================================================

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 类别标签映射
CLASS_LABELS = {
    '0': '正常流量',
    '1': '网络攻击（通用）',
    '2': 'HTTP3 Flood攻击',
    '3': 'HTTP3 Stream攻击',
    '4': 'HTTP3 LORIS攻击'
}


def load_training_data(file_path, max_examples_per_class=5):
    """
    加载训练数据，只提取32维包大小序列
    """
    print(f"正在加载训练数据从: {file_path}")

    # 按类别存储数据
    class_data = {str(i): [] for i in range(5)}

    with open(file_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            if len(row) < 93:  # 确保有足够的列
                continue

            # 只提取32维包大小序列（列61-92）
            packet_sizes = row[60:92]  # 索引60到91是32个包大小
            label = row[92]

            # 验证标签是否有效
            if label in class_data:
                class_data[label].append({
                    'packet_sizes': packet_sizes,
                    'full_row': row
                })

    # 为每个类别选择代表性样本
    selected_examples = []
    for label in range(5):
        label_str = str(label)
        examples = class_data[label_str]

        if len(examples) > max_examples_per_class:
            # 如果样本过多，随机选择
            indices = np.random.choice(len(examples), max_examples_per_class, replace=False)
            selected = [examples[i] for i in indices]
        else:
            selected = examples

        for ex in selected:
            ex['class_name'] = CLASS_LABELS[label_str]

        selected_examples.extend(selected)

        print(f"  类别 {label} ({CLASS_LABELS[label_str]}): 选择了 {len(selected)}/{len(examples)} 条示例")

    print(f"总计选择了 {len(selected_examples)} 条训练示例")
    return selected_examples


def load_test_data(file_path):
    """
    加载测试数据，只提取32维包大小序列
    """
    print(f"\n正在加载测试数据从: {file_path}")

    test_data = []
    true_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)

        for i, row in enumerate(csv_reader):
            if len(row) < 93:  # 测试文件有标签列，应该有93列
                print(f"警告: 第 {i + 1} 行列数不足 ({len(row)}), 跳过")
                continue

            # 只提取32维包大小序列（列61-92）
            packet_sizes = row[60:92]  # 索引60到91是32个包大小
            label = row[92]  # 最后一列是标签

            test_data.append({
                'index': i,
                'packet_sizes': packet_sizes,
                'full_row': row
            })
            true_labels.append(label)

    print(f"成功加载 {len(test_data)} 条测试数据")

    # 显示真实标签分布
    label_dist = Counter(true_labels)
    print("真实标签分布:")
    for label in sorted(label_dist.keys()):
        class_name = CLASS_LABELS.get(label, '未知')
        print(
            f"  类别 {label} ({class_name}): {label_dist[label]} 条 ({label_dist[label] / len(true_labels) * 100:.2f}%)")

    return test_data, true_labels


def analyze_packet_sequence(packet_sizes):
    """
    分析包大小序列，提取关键特征
    """
    # 将字符串转换为浮点数
    try:
        sizes_float = [float(p) for p in packet_sizes if p]
    except:
        sizes_float = []

    if not sizes_float:
        return "包大小序列为空"

    # 转换为numpy数组便于计算
    arr = np.array(sizes_float)

    # 计算各种统计特征
    analysis = []

    # 基本统计
    analysis.append(f"序列长度: {len(arr)}")
    analysis.append(f"平均值: {np.mean(arr):.4f}")
    analysis.append(f"标准差: {np.std(arr):.4f}")
    analysis.append(f"最小值: {np.min(arr):.4f}")
    analysis.append(f"最大值: {np.max(arr):.4f}")
    analysis.append(f"中位数: {np.median(arr):.4f}")

    # 变异系数 (CV = 标准差/平均值)，反映数据的离散程度
    if np.mean(arr) != 0:
        cv = np.std(arr) / np.mean(arr)
        analysis.append(f"变异系数: {cv:.4f}")

    # 偏度和峰度，反映分布形状
    from scipy import stats
    if len(arr) > 3:  # 需要足够的样本计算偏度和峰度
        skewness = stats.skew(arr)
        kurtosis = stats.kurtosis(arr)
        analysis.append(f"偏度: {skewness:.4f}")
        analysis.append(f"峰度: {kurtosis:.4f}")

    # 四分位数
    q1, q3 = np.percentile(arr, [25, 75])
    analysis.append(f"Q1 (25%): {q1:.4f}")
    analysis.append(f"Q3 (75%): {q3:.4f}")
    analysis.append(f"IQR (四分位距): {q3 - q1:.4f}")

    # 序列变化特征
    diffs = np.diff(arr)
    analysis.append(f"相邻差值平均值: {np.mean(np.abs(diffs)):.4f}")
    analysis.append(f"相邻差值最大值: {np.max(np.abs(diffs)):.4f}")

    # 零值比例
    zero_ratio = np.sum(arr == 0) / len(arr)
    analysis.append(f"零值比例: {zero_ratio:.4f}")

    # 大于0的比例
    positive_ratio = np.sum(arr > 0) / len(arr)
    analysis.append(f"正值比例: {positive_ratio:.4f}")

    # 趋势分析
    # 简单线性回归斜率
    x = np.arange(len(arr))
    if len(arr) > 1:
        slope = np.polyfit(x, arr, 1)[0]
        analysis.append(f"趋势斜率: {slope:.4f}")

    # 周期性检测（简单方法：自相关）
    if len(arr) > 10:
        # 计算一阶自相关
        autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1] if len(arr) > 1 else 0
        analysis.append(f"一阶自相关: {autocorr:.4f}")

    # 异常值检测（超过3倍标准差）
    mean = np.mean(arr)
    std = np.std(arr)
    outliers = np.sum(np.abs(arr - mean) > 3 * std)
    analysis.append(f"异常值数量: {outliers}")

    return "\n    ".join(analysis)


def generate_packet_sequence_description(packet_sizes):
    """
    生成包大小序列的描述
    """
    if SHOW_RAW_PACKET_SIZES:
        # 显示原始包大小值
        return f"包大小序列 (32维): [{', '.join(packet_sizes)}]"
    else:
        # 显示分析结果
        analysis = analyze_packet_sequence(packet_sizes)
        return f"包大小序列分析:\n    {analysis}"


def generate_few_shot_prompt(training_examples):
    """
    生成包含小样本示例的提示词，专注于包大小序列分析
    """
    prompt = """你是一个网络安全领域的专家，精通流量检测任务。你需要基于HTTP/3流量的包大小序列进行分类。

包大小序列是32个连续数据包的大小（已归一化），按时间顺序排列。不同的攻击类型会在包大小序列上表现出独特的模式：

0：正常流量 - 包大小分布较均匀，无明显异常模式
1：网络攻击（通用） - 包大小可能出现异常波动
2：HTTP3 Flood攻击 - 包大小可能较大且密集，呈现高流量特征
3：HTTP3 Stream攻击 - 包大小序列可能呈现特定的流模式变化
4：HTTP3 LORIS攻击 - 包大小可能较小且稀疏，呈现慢速攻击特征

请通过分析包大小序列的统计特征和变化模式来判断流量类别。

"""

    if training_examples:
        prompt += "以下是各类流量的包大小序列示例分析：\n\n"

        for i, ex in enumerate(training_examples):
            prompt += f"示例 {i + 1} (类别 {ex['class_name']}):\n"
            prompt += generate_packet_sequence_description(ex['packet_sizes']) + "\n\n"

    prompt += """\n现在，请对以下新的包大小序列进行分类。请仔细分析其统计特征和变化模式，参考上述示例，判断它属于哪一类。

重要规则：
1. 只输出一个数字（0-4），不要输出任何其他文字
2. 不要输出分析过程
3. 不要输出格式标记

新的包大小序列：
"""

    return prompt


def generate_zero_shot_prompt():
    """
    生成零样本学习的提示词（不包含示例）
    """
    prompt = """你是一个网络安全领域的专家，精通流量检测任务。你需要基于HTTP/3流量的包大小序列进行分类。

包大小序列是32个连续数据包的大小（已归一化），按时间顺序排列。不同的攻击类型会在包大小序列上表现出独特的模式：

0：正常流量 - 正常的HTTP/3通信流量，包大小分布较均匀，无明显异常模式
1：网络攻击（通用） - 一般性的网络攻击流量，包大小可能出现异常波动
2：HTTP3 Flood攻击 - 大量HTTP/3请求的泛洪攻击，包大小可能较大且密集，呈现高流量特征
3：HTTP3 Stream攻击 - 针对HTTP/3流的攻击，包大小序列可能呈现特定的流模式变化，如周期性波动
4：HTTP3 LORIS攻击 - Slow Loris类型的慢速攻击，包大小可能较小且稀疏，呈现慢速攻击特征

分类规则：
- 分析包大小序列的统计特征（平均值、标准差、变异系数等）
- 观察序列的变化模式（趋势、周期性、突变等）
- 识别异常值分布和零值比例
- 比较与各类攻击的典型模式

重要规则：
1. 只输出一个数字（0-4），不要输出任何其他文字
2. 不要输出分析过程
3. 不要输出格式标记

新的包大小序列：
"""

    return prompt


def predict_single(packet_sizes, base_prompt, retry_count=0):
    """
    单条预测，只基于包大小序列
    """
    # 构建完整的提示词
    packet_description = generate_packet_sequence_description(packet_sizes)
    full_prompt = base_prompt + "\n" + packet_description

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "你是一个网络安全专家，专注于HTTP/3流量分类。你只输出分类数字，不输出其他内容。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,  # 低温度使输出更确定
            max_tokens=10
        )

        predicted = response.choices[0].message.content.strip()

        # 清理预测结果，只保留数字
        predicted = ''.join(filter(str.isdigit, predicted))

        if predicted in ['0', '1', '2', '3', '4']:
            return predicted
        else:
            print(f"警告: 模型返回了无效的预测结果: '{predicted}'，将使用默认值0")
            return '0'  # 默认返回正常流量

    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"预测失败，{RETRY_DELAY}秒后重试 ({retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return predict_single(packet_sizes, base_prompt, retry_count + 1)
        else:
            print(f"预测失败，已达最大重试次数: {e}")
            return '0'  # 失败时默认返回正常流量


def batch_predict(test_data, base_prompt, batch_size=5, delay=1):
    """
    批量预测
    """
    print(f"\n开始批量预测，共 {len(test_data)} 条数据，批次大小: {batch_size}")

    predictions = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]

        print(
            f"处理批次 {i // batch_size + 1}/{(len(test_data) - 1) // batch_size + 1} (样本 {i + 1}-{min(i + batch_size, len(test_data))})")

        for item in batch:
            predicted = predict_single(item['packet_sizes'], base_prompt)
            predictions.append((item['index'], predicted))

            # 显示预测结果
            class_name = CLASS_LABELS.get(predicted, '未知')
            print(f"  样本 {item['index'] + 1}: 预测类别 {predicted} ({class_name})")

            time.sleep(delay)  # 避免触发API限流

    # 按原始顺序排序
    predictions.sort(key=lambda x: x[0])
    ordered_predictions = [pred for _, pred in predictions]

    print(f"\n批量预测完成。共预测 {len(predictions)} 条数据")

    return ordered_predictions


def calculate_metrics(true_labels, predictions):
    """
    计算评估指标
    """
    # 转换为整数
    true_labels_int = [int(t) for t in true_labels]
    predictions_int = [int(p) for p in predictions]

    # 计算准确率
    accuracy = accuracy_score(true_labels_int, predictions_int)

    # 计算F1分数（宏平均和加权平均）
    f1_macro = f1_score(true_labels_int, predictions_int, average='macro')
    f1_weighted = f1_score(true_labels_int, predictions_int, average='weighted')

    # 为每个类别计算F1分数
    labels = sorted(set(true_labels_int))
    f1_per_class = f1_score(true_labels_int, predictions_int, labels=labels, average=None)

    return accuracy, f1_macro, f1_weighted, f1_per_class, labels, true_labels_int, predictions_int


def print_detailed_report(true_labels, predictions, labels, f1_per_class, accuracy, f1_macro, f1_weighted):
    """
    打印详细的评估报告
    """
    print("\n" + "=" * 60)
    print("📊 详细评估报告 (仅使用包大小序列)")
    print("=" * 60)

    true_labels_int = [int(t) for t in true_labels]
    predictions_int = [int(p) for p in predictions]

    # 打印主要指标
    print(f"\n📈 主要指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  宏平均F1分数 (Macro F1): {f1_macro:.4f}")
    print(f"  加权平均F1分数 (Weighted F1): {f1_weighted:.4f}")

    # 使用sklearn的分类报告
    print("\n📋 分类报告:")
    report = classification_report(true_labels_int, predictions_int,
                                   target_names=[f'类别 {l} ({CLASS_LABELS[str(l)]})' for l in labels],
                                   digits=4)
    print(report)

    # 打印混淆矩阵
    cm = confusion_matrix(true_labels_int, predictions_int, labels=labels)
    print("\n🔄 混淆矩阵:")
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
    print("\n🎯 每个类别的准确率:")
    for i, l in enumerate(labels):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            print(f"  类别 {l} ({CLASS_LABELS[str(l)]}): {class_acc:.4f} ({cm[i, i]}/{np.sum(cm[i, :])})")

    # 计算错误分类统计
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_labels_int, predictions_int)) if t != p]
    if errors:
        print(f"\n❌ 错误分类统计 (共 {len(errors)} 条):")
        error_by_pair = Counter([(t, p) for _, t, p in errors])
        for (true, pred), count in sorted(error_by_pair.items()):
            print(f"  真实类别 {true} → 预测类别 {pred}: {count} 条")


def save_predictions_and_metrics(predictions, true_labels, accuracy, f1_macro, f1_weighted,
                                 predictions_path, metrics_path):
    """
    保存预测结果和评估指标
    """
    # 保存预测结果
    with open(predictions_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['预测标签'])  # 添加表头
        for pred in predictions:
            writer.writerow([pred])

    print(f"\n预测结果已保存到: {predictions_path}")

    # 保存评估指标
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("HTTP/3 流量分类模型评估报告 (仅使用包大小序列)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"📈 主要指标:\n")
        f.write(f"  准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"  宏平均F1分数 (Macro F1): {f1_macro:.4f}\n")
        f.write(f"  加权平均F1分数 (Weighted F1): {f1_weighted:.4f}\n\n")

        # 保存分类报告
        true_labels_int = [int(t) for t in true_labels]
        predictions_int = [int(p) for p in predictions]
        labels = sorted(set(true_labels_int))

        f.write("📋 分类报告:\n")
        report = classification_report(true_labels_int, predictions_int,
                                       target_names=[f'类别 {l} ({CLASS_LABELS[str(l)]})' for l in labels],
                                       digits=4)
        f.write(report + "\n")

        # 保存混淆矩阵
        cm = confusion_matrix(true_labels_int, predictions_int, labels=labels)
        f.write("\n🔄 混淆矩阵:\n")
        f.write("预测→\n")
        f.write("真实↓\n")

        f.write("     ")
        for l in labels:
            f.write(f"  {l}  ")
        f.write("\n")

        for i, l in enumerate(labels):
            f.write(f"  {l}  ")
            for j in range(len(labels)):
                f.write(f"  {cm[i][j]:3d} ")
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"评估指标已保存到: {metrics_path}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("HTTP/3 流量分类 - 仅使用包大小序列（对比实验）")
    print("=" * 60)

    # 检查API密钥
    if API_KEY == "你的OpenAI API密钥":
        print("❌ 错误: 请先设置你的OpenAI API密钥")
        return

    print(f"\n📊 配置信息:")
    print(f"  - 模型: {MODEL}")
    print(f"  - 训练文件: {TRAIN_FILE_PATH}")
    print(f"  - 测试文件: {TEST_FILE_PATH}")
    print(f"  - 输出文件: {OUTPUT_FILE_PATH}")
    print(f"  - 指标文件: {METRICS_OUTPUT_PATH}")
    print(f"  - 小样本学习: {'是' if USE_FEW_SHOT else '否'}")
    print(f"  - 输入特征: 仅使用32维包大小序列")
    if USE_FEW_SHOT:
        print(f"  - 每类示例数: {MAX_EXAMPLES_PER_CLASS}")
    print(f"  - 显示原始包大小: {'是' if SHOW_RAW_PACKET_SIZES else '否 (使用统计分析)'}")

    # 加载训练数据
    print("\n📂 步骤1: 加载训练数据")
    training_examples = load_training_data(TRAIN_FILE_PATH, MAX_EXAMPLES_PER_CLASS)

    # 生成提示词
    print("\n📝 步骤2: 生成分类提示词")
    if USE_FEW_SHOT and training_examples:
        base_prompt = generate_few_shot_prompt(training_examples)
        print(f"已生成小样本提示词 (包含 {len(training_examples)} 条示例)")
    else:
        base_prompt = generate_zero_shot_prompt()
        print("已生成零样本提示词")

    # 显示提示词预览
    print("\n提示词预览 (前500字符):")
    print("-" * 40)
    print(base_prompt[:500] + "...")
    print("-" * 40)

    # 加载测试数据
    print("\n📂 步骤3: 加载测试数据")
    test_data, true_labels = load_test_data(TEST_FILE_PATH)

    if len(test_data) == 0:
        print("错误: 没有加载到测试数据")
        return

    # 执行预测
    print("\n🤖 步骤4: 执行分类预测")
    predictions = batch_predict(test_data, base_prompt, BATCH_SIZE, REQUEST_DELAY)

    # 计算评估指标
    print("\n📊 步骤5: 计算评估指标")
    accuracy, f1_macro, f1_weighted, f1_per_class, labels, true_labels_int, predictions_int = calculate_metrics(
        true_labels, predictions)

    # 打印详细报告
    print_detailed_report(true_labels, predictions, labels, f1_per_class, accuracy, f1_macro, f1_weighted)

    # 保存结果
    print("\n💾 步骤6: 保存结果")
    save_predictions_and_metrics(predictions, true_labels, accuracy, f1_macro, f1_weighted,
                                 OUTPUT_FILE_PATH, METRICS_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("✅ 对比实验完成！")
    print("=" * 60)


def quick_test():
    """
    快速测试函数：只测试少量样本
    """
    global TEST_FILE_PATH, OUTPUT_FILE_PATH, METRICS_OUTPUT_PATH

    # 创建临时测试文件
    temp_test_file = "temp_test.csv"
    temp_output_file = "temp_output_packet.csv"
    temp_metrics_file = "temp_metrics_packet.txt"

    print("\n🔍 执行快速测试...")

    # 读取原测试文件的前10行
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()[:10]

    with open(temp_test_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(lines)

    # 临时替换文件路径
    original_test = TEST_FILE_PATH
    original_output = OUTPUT_FILE_PATH
    original_metrics = METRICS_OUTPUT_PATH

    TEST_FILE_PATH = temp_test_file
    OUTPUT_FILE_PATH = temp_output_file
    METRICS_OUTPUT_PATH = temp_metrics_file

    main()

    # 恢复原路径
    TEST_FILE_PATH = original_test
    OUTPUT_FILE_PATH = original_output
    METRICS_OUTPUT_PATH = original_metrics


if __name__ == "__main__":
    import sys

    print("HTTP/3 流量分类 - 仅使用包大小序列（对比实验）")
    print("1. 完整评估")
    print("2. 快速测试 (10条样本)")
    print("3. 退出")

    choice = input("请选择 (1-3): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    else:
        print("退出程序")
        sys.exit(0)