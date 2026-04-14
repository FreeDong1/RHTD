# =====================================================
# 基准模型混淆矩阵输出代码
# 作用：单独输出基线模型的混淆矩阵，包括数值和可视化图表
# =====================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 全局配置
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 特征维度配置
FLOW_DIM = 60
SEQ_DIM = 32
NUM_CLASSES = 5

# 路径配置
MODEL_DIR = "saved_models"
TEST_DATA_DIR = "test_data"
TRAIN_DATA_DIR = "train_data"
RESULT_DIR = "baseline_results"

# 关键文件路径
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "enhanced_model_final.pth")
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "fixed_test_data.pkl")
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "train_data.pkl")
ALL_CSV_PATH = "data/all.csv"

# 标签映射
LABEL_MAP = {
    "be": 0,
    "web": 1,
    "flood": 2,
    "loris": 3,
    "stream": 4
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES = ["be", "web", "flood", "loris", "stream"]
CLASS_NAMES_CN = ["良性", "Web", "洪水", "慢速", "流"]  # 中文标签

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# 1. 模型定义（与原代码保持一致）
# =====================================================
class FeatureEncoder(nn.Module):
    """特征提取器"""

    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class PrototypicalNetwork(nn.Module):
    """原型网络模型"""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.flow_encoder = FeatureEncoder(FLOW_DIM, 64)
        self.seq_encoder = FeatureEncoder(SEQ_DIM, 64)
        self.feature_dim = feature_dim

    def forward(self, flow, seq):
        flow_features = self.flow_encoder(flow)
        seq_features = self.seq_encoder(seq)
        return torch.cat([flow_features, seq_features], dim=1)

    def compute_prototypes(self, support_features, support_labels):
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototypes[label.item()] = class_features.mean(dim=0)
        return prototypes

    def predict(self, query_features, prototypes):
        predictions = []
        distances = []
        for query_feature in query_features:
            class_distances = []
            for label, prototype in prototypes.items():
                distance = torch.norm(query_feature - prototype, p=2)
                class_distances.append((label, distance))
            class_distances.sort(key=lambda x: x[1])
            predictions.append(class_distances[0][0])
            distances.append(class_distances[0][1])
        return torch.tensor(predictions).to(query_features.device), torch.tensor(distances).to(query_features.device)


# =====================================================
# 2. 数据加载函数
# =====================================================
def load_test_data():
    """加载测试数据"""
    print("\n📥 加载测试数据...")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"测试数据不存在: {TEST_DATA_PATH}")

    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    print(f"✅ 测试数据加载完成")
    print(f"   测试集样本数: {len(test_data['labels'])}")

    # 统计各类别样本数
    label_counts = {}
    for label in test_data['labels']:
        class_name = INV_LABEL_MAP[label]
        label_counts[class_name] = label_counts.get(class_name, 0) + 1

    print(f"   测试集类别分布:")
    for cls in CLASSES:
        print(f"     {cls}: {label_counts.get(cls, 0)} 个样本")

    return test_data


def load_train_data_for_prototypes():
    """加载训练数据用于构建原型"""
    print("\n📥 加载训练数据（用于构建原型）...")

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_DATA_PATH}")

    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)

    print(f"✅ 训练数据加载完成")
    print(f"   训练集类别分布: {[f'{cls}: {len(train_data[cls])}' for cls in CLASSES]}")

    return train_data


# =====================================================
# 3. 混淆矩阵计算和可视化函数
# =====================================================
def compute_baseline_confusion_matrix(model, test_data, train_data):
    """
    计算基准模型的混淆矩阵
    返回：混淆矩阵（原始计数）、归一化混淆矩阵、准确率、各类别准确率
    """
    model.eval()
    all_true = []
    all_pred = []

    print("\n🔄 正在计算基准模型预测结果...")

    with torch.no_grad():
        # Step 1: 从训练数据构建原型
        print("   构建类别原型...")
        all_support_features = []
        all_support_labels = []

        for cls in CLASSES:
            cls_label = LABEL_MAP[cls]
            train_samples = train_data[cls]

            # 每个类别使用最多50个样本构建原型
            if len(train_samples) > 50:
                sampled_indices = random.sample(range(len(train_samples)), 50)
            else:
                sampled_indices = range(len(train_samples))

            for idx in sampled_indices:
                f, s = train_samples[idx]
                features = model(
                    torch.tensor([f], dtype=torch.float32).to(DEVICE),
                    torch.tensor([s], dtype=torch.float32).to(DEVICE)
                )
                all_support_features.append(features)
                all_support_labels.append(cls_label)

        support_features = torch.cat(all_support_features, dim=0)
        support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
        prototypes = model.compute_prototypes(support_features, support_labels)
        print(f"   原型构建完成，共 {len(prototypes)} 个类别原型")

        # Step 2: 测试集预测
        print("   测试集预测...")
        num_samples = len(test_data["labels"])
        batch_size = 64

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_flow = torch.tensor(test_data["features_flow"][i:end_idx]).to(DEVICE)
            batch_seq = torch.tensor(test_data["features_seq"][i:end_idx]).to(DEVICE)
            batch_features = model(batch_flow, batch_seq)
            batch_preds, _ = model.predict(batch_features, prototypes)

            all_true.extend(test_data["labels"][i:end_idx])
            all_pred.extend(batch_preds.cpu().numpy())

            if (i + batch_size) % 256 == 0 or end_idx == num_samples:
                print(f"     已处理 {end_idx}/{num_samples} 个样本")

    # 转换为numpy数组
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # 计算总体准确率
    overall_acc = accuracy_score(all_true, all_pred)

    # 计算各类别准确率
    class_acc = {}
    for cls in CLASSES:
        cls_label = LABEL_MAP[cls]
        mask = all_true == cls_label
        if np.sum(mask) > 0:
            class_acc[cls] = accuracy_score(all_true[mask], all_pred[mask])
        else:
            class_acc[cls] = 0.0

    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况

    print(f"\n✅ 预测完成")
    print(f"   总体准确率: {overall_acc:.4f}")

    return cm, cm_normalized, overall_acc, class_acc, all_true, all_pred


def plot_confusion_matrix(cm, cm_normalized, overall_acc, class_acc, save_path):
    """
    绘制混淆矩阵图（多种风格）
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ===== 图1: 计数混淆矩阵 =====
    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN,
                yticklabels=CLASS_NAMES_CN,
                cbar_kws={'label': '样本数量'})

    plt.title(f'基准模型混淆矩阵（计数）\n总体准确率: {overall_acc:.4f}', fontsize=14, fontweight='bold')
    plt.ylabel('真实类别', fontsize=12)
    plt.xlabel('预测类别', fontsize=12)

    # 添加类别准确率文本
    acc_text = "各类别准确率:\n"
    for i, cls in enumerate(CLASSES):
        acc_text += f"{CLASS_NAMES_CN[i]}: {class_acc[cls]:.4f}\n"

    plt.text(1.05, 0.5, acc_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix_count.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 图2: 归一化混淆矩阵 =====
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlBu_r',
                xticklabels=CLASS_NAMES_CN,
                yticklabels=CLASS_NAMES_CN,
                cbar_kws={'label': '归一化比例'})

    plt.title(f'基准模型混淆矩阵（归一化）\n总体准确率: {overall_acc:.4f}', fontsize=14, fontweight='bold')
    plt.ylabel('真实类别', fontsize=12)
    plt.xlabel('预测类别', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 图3: 简洁风格混淆矩阵 =====
    plt.figure(figsize=(8, 6))

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'基准模型混淆矩阵\n准确率: {overall_acc:.4f}', fontsize=14, fontweight='bold')
    plt.colorbar(label='归一化比例')

    tick_marks = np.arange(len(CLASS_NAMES_CN))
    plt.xticks(tick_marks, CLASS_NAMES_CN, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES_CN)

    # 添加数值标注
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2%}',
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black",
                     fontsize=11, fontweight='bold')

    plt.ylabel('真实类别', fontsize=11)
    plt.xlabel('预测类别', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix_simple.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 混淆矩阵图已保存至: {save_path}")


def save_confusion_matrix_csv(cm, cm_normalized, class_acc, overall_acc, save_path):
    """
    保存混淆矩阵为CSV文件
    """
    # 保存原始计数矩阵
    cm_df = pd.DataFrame(cm,
                         index=[f'{cls}' for cls in CLASSES],
                         columns=[f'{cls}' for cls in CLASSES])
    cm_df.to_csv(os.path.join(save_path, "confusion_matrix_count.csv"), encoding='utf-8')

    # 保存归一化矩阵
    cm_norm_df = pd.DataFrame(cm_normalized,
                              index=[f'{cls}' for cls in CLASSES],
                              columns=[f'{cls}' for cls in CLASSES])
    cm_norm_df.to_csv(os.path.join(save_path, "confusion_matrix_normalized.csv"), encoding='utf-8', float_format='%.4f')

    # 保存详细结果
    detailed_results = {
        'overall_accuracy': overall_acc,
        'class_accuracy': class_acc,
        'confusion_matrix_count': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist()
    }

    # 保存为pkl
    with open(os.path.join(save_path, "baseline_detailed_results.pkl"), 'wb') as f:
        pickle.dump(detailed_results, f)

    # 保存为txt
    with open(os.path.join(save_path, "baseline_results.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("基准模型性能评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总体准确率: {overall_acc:.4f}\n\n")

        f.write("各类别准确率:\n")
        for cls in CLASSES:
            f.write(f"  {cls}: {class_acc[cls]:.4f}\n")

        f.write("\n混淆矩阵（原始计数）:\n")
        f.write("真实\\预测\t" + "\t".join(CLASSES) + "\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"{cls}\t" + "\t".join(map(str, cm[i])) + "\n")

        f.write("\n混淆矩阵（归一化）:\n")
        f.write("真实\\预测\t" + "\t".join(CLASSES) + "\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"{cls}\t" + "\t".join([f"{v:.4f}" for v in cm_normalized[i]]) + "\n")

    print(f"✅ 混淆矩阵数值已保存至: {save_path}")


def plot_class_accuracy(class_acc, overall_acc, save_path):
    """
    绘制各类别准确率条形图
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))

    classes = CLASS_NAMES_CN
    acc_values = [class_acc[cls] for cls in CLASSES]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    bars = plt.bar(classes, acc_values, color=colors, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, acc in zip(bars, acc_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2,
                label=f'总体准确率: {overall_acc:.4f}')

    plt.ylim(0, 1.1)
    plt.ylabel('准确率', fontsize=12)
    plt.title('基准模型各类别准确率', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "class_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 类别准确率图已保存至: {save_path}")


# =====================================================
# 4. 主函数
# =====================================================
def main():
    print("=" * 80)
    print("基准模型混淆矩阵输出程序")
    print("=" * 80)

    # Step 1: 加载模型
    print("\n🔧 加载基准模型...")
    model = PrototypicalNetwork().to(DEVICE)

    if os.path.exists(BASE_MODEL_PATH):
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
        print(f"✅ 成功加载预训练模型: {BASE_MODEL_PATH}")
    else:
        print(f"⚠️ 预训练模型不存在，使用随机初始化模型")

    model.eval()

    # Step 2: 加载数据
    test_data = load_test_data()
    train_data = load_train_data_for_prototypes()

    # Step 3: 计算混淆矩阵
    print("\n📊 开始计算混淆矩阵...")
    cm, cm_normalized, overall_acc, class_acc, all_true, all_pred = compute_baseline_confusion_matrix(
        model, test_data, train_data
    )

    # Step 4: 保存混淆矩阵数值
    print("\n💾 保存混淆矩阵数值...")
    save_confusion_matrix_csv(cm, cm_normalized, class_acc, overall_acc, RESULT_DIR)

    # Step 5: 绘制混淆矩阵图
    print("\n🎨 绘制混淆矩阵可视化图表...")
    plot_confusion_matrix(cm, cm_normalized, overall_acc, class_acc, RESULT_DIR)

    # Step 6: 绘制类别准确率图
    plot_class_accuracy(class_acc, overall_acc, RESULT_DIR)

    # Step 7: 输出详细评估报告
    print("\n" + "=" * 80)
    print("基准模型评估报告")
    print("=" * 80)

    print(f"\n📈 总体性能:")
    print(f"   总体准确率: {overall_acc:.4f}")
    print(f"   总体错误率: {1 - overall_acc:.4f}")

    print(f"\n📊 各类别性能:")
    for cls in CLASSES:
        cls_label = LABEL_MAP[cls]
        mask = all_true == cls_label
        n_samples = np.sum(mask)
        print(f"   {cls}:")
        print(f"     样本数: {n_samples}")
        print(f"     准确率: {class_acc[cls]:.4f}")
        print(f"     错误率: {1 - class_acc[cls]:.4f}")

    print(f"\n📋 混淆矩阵（归一化）:")
    print("真实\\预测\t" + "\t".join(CLASSES[:3]) + "\t" + "\t".join(CLASSES[3:]))
    for i, cls in enumerate(CLASSES):
        print(f"{cls}\t" + "\t".join([f"{v:.4f}" for v in cm_normalized[i][:3]]) + "\t" + "\t".join(
            [f"{v:.4f}" for v in cm_normalized[i][3:]]))

    # 误分类分析
    print(f"\n🔍 误分类分析:")
    misclassified_total = np.sum(all_true != all_pred)
    print(f"   总误分类样本数: {misclassified_total}")

    # 找出最容易混淆的类别对
    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:
                confusion_pairs.append((i, j, cm[i][j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n   最容易混淆的类别对（真实→预测）:")
    for i, (true_idx, pred_idx, count) in enumerate(confusion_pairs[:5]):
        if count > 0:
            true_cls = CLASSES[true_idx]
            pred_cls = CLASSES[pred_idx]
            print(f"     {i + 1}. {true_cls} → {pred_cls}: {count} 个样本 ({count / len(all_true) * 100:.2f}%)")

    print(f"\n✅ 所有结果已保存至: {os.path.abspath(RESULT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()