
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, silhouette_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 全局配置
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 模型参数配置
FLOW_DIM = 60
SEQ_DIM = 32
NUM_CLASSES = 5
LABEL_MAP = {
    "be": 0,
    "web": 1,
    "flood": 2,
    "loris": 3,
    "stream": 4
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES = ["be", "web", "flood", "loris", "stream"]
WEB_CLASS_ID = LABEL_MAP["web"]

# 文件路径配置
BASE_MODEL_PATH = "saved_models/enhanced_model_final.pth"
TEST_DATA_PATH = "test_data/fixed_test_data.pkl"
TRAIN_DATA_PATH = "train_data/train_data.pkl"
RESULT_DIR = "comprehensive_evaluation_fixed_results"

# 评估配置
BATCH_SIZE = 64
SUPPORT_PER_CLASS = 50  # 每个类使用的原型样本数

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# 模型定义
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
        """计算类别原型"""
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototypes[label.item()] = class_features.mean(dim=0)
        return prototypes

    def predict(self, query_features, prototypes):
        """基于原型进行预测"""
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
# 数据加载函数
# =====================================================
def load_data():
    """加载训练集和测试集"""
    print("\n" + "=" * 80)
    print("加载数据集")
    print("=" * 80)

    # 加载测试集
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"测试数据集不存在: {TEST_DATA_PATH}")

    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    # 加载训练集
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"训练数据集不存在: {TRAIN_DATA_PATH}")

    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)

    print(f"✅ 数据加载完成")
    print(f"   测试集样本数: {len(test_data['labels'])}")
    print(f"   测试集类别分布: {Counter(test_data['class_names'])}")

    # 统计训练集每个类别的样本数
    train_counts = {}
    for cls in CLASSES:
        train_counts[cls] = len(train_data.get(cls, []))
    print(f"   训练集类别分布: {train_counts}")

    return train_data, test_data


def prepare_tensors(test_data):
    """将测试数据转换为tensor格式"""
    test_data_tensor = {
        "features_flow": torch.tensor(test_data["features_flow"], dtype=torch.float32).to(DEVICE),
        "features_seq": torch.tensor(test_data["features_seq"], dtype=torch.float32).to(DEVICE),
        "labels": torch.tensor(test_data["labels"], dtype=torch.int64).to(DEVICE),
        "class_names": test_data["class_names"]
    }
    return test_data_tensor


# =====================================================
# 模型加载函数
# =====================================================
def load_model():
    """加载预训练模型"""
    print("\n" + "=" * 80)
    print("加载预训练模型")
    print("=" * 80)

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {BASE_MODEL_PATH}")

    model = PrototypicalNetwork().to(DEVICE)
    checkpoint = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"✅ 模型加载成功: {BASE_MODEL_PATH}")
    print(f"   模型设备: {DEVICE}")

    return model


def compute_prototypes_as_feature_evaluate2(model, train_data):
    """
    完全按照feature_evaluate2.py的方式计算原型
    使用random.sample随机采样50个样本
    """
    print(f"\n📊 按照feature_evaluate2.py方式计算原型...")

    all_support_features = []
    all_support_labels = []

    with torch.no_grad():
        for cls in CLASSES:
            train_samples = train_data.get(cls, [])
            if len(train_samples) == 0:
                print(f"⚠️ 警告: {cls}类没有训练样本")
                continue

         
            if len(train_samples) > 50:            
                sampled_indices = random.sample(range(len(train_samples)), 50)
                print(f"   {cls}: 随机采样50个样本 (总样本数: {len(train_samples)})")
            else:
                sampled_indices = range(len(train_samples))
                print(f"   {cls}: 使用全部{len(train_samples)}个样本")

            for idx in sampled_indices:
                f, s = train_samples[idx]
                f_tensor = torch.tensor([f], dtype=torch.float32).to(DEVICE)
                s_tensor = torch.tensor([s], dtype=torch.float32).to(DEVICE)

                features = model(f_tensor, s_tensor)
                all_support_features.append(features)
                all_support_labels.append(LABEL_MAP[cls])

    if len(all_support_features) == 0:
        raise RuntimeError("无法计算原型：训练集无有效样本")

    # 拼接特征和标签
    support_features = torch.cat(all_support_features, dim=0)
    support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)

    # 计算原型
    prototypes = model.compute_prototypes(support_features, support_labels)

    # 打印原型向量的一部分用于验证
    print(f"\n✅ 原型计算完成:")
    for label, proto in prototypes.items():
        class_name = INV_LABEL_MAP[label]
        print(f"   {class_name}原型向量前5维: {proto[:5].cpu().numpy()}")

    return prototypes, support_features, support_labels


def evaluate_accuracy_as_feature_evaluate2(model, test_data, train_data):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        # 1. 计算原型（每次评估都重新随机采样）
        all_support_features = []
        all_support_labels = []

        for cls in CLASSES:
            train_samples = train_data.get(cls, [])
            if len(train_samples) == 0:
                continue

            # 随机采样50个样本
            if len(train_samples) > 50:
                sampled_indices = random.sample(range(len(train_samples)), 50)
            else:
                sampled_indices = range(len(train_samples))

            for idx in sampled_indices:
                f, s = train_samples[idx]
                f_tensor = torch.tensor([f], dtype=torch.float32).to(DEVICE)
                s_tensor = torch.tensor([s], dtype=torch.float32).to(DEVICE)

                features = model(f_tensor, s_tensor)
                all_support_features.append(features)
                all_support_labels.append(LABEL_MAP[cls])

        support_features = torch.cat(all_support_features, dim=0)
        support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
        prototypes = model.compute_prototypes(support_features, support_labels)

        # 2. 测试集预测
        num_samples = len(test_data["labels"])
        for i in range(0, num_samples, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, num_samples)

            batch_flow = test_data["features_flow"][i:end_idx]
            batch_seq = test_data["features_seq"][i:end_idx]
            batch_labels = test_data["labels"][i:end_idx]

            batch_features = model(batch_flow, batch_seq)
            batch_preds, _ = model.predict(batch_features, prototypes)

            all_true.extend(batch_labels.cpu().numpy())
            all_pred.extend(batch_preds.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_true, all_pred)

    # 计算每个类别的准确率
    class_acc = {}
    for cls in CLASSES:
        cls_label = LABEL_MAP[cls]
        mask = np.array(all_true) == cls_label
        if np.sum(mask) > 0:
            class_acc[cls] = accuracy_score(np.array(all_true)[mask], np.array(all_pred)[mask])
        else:
            class_acc[cls] = 0.0

    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(f"   整体准确率: {accuracy:.4f}")
    print(f"   类别准确率:")
    for cls in CLASSES:
        print(f"     {cls}: {class_acc[cls]:.4f}")

    return {
        "accuracy": accuracy,
        "class_accuracy": class_acc,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_normalized,
        "predictions": all_pred,
        "true_labels": all_true,
        "prototypes": prototypes
    }


# =====================================================
# Embedding提取函数
# =====================================================
def extract_embeddings(model, test_data):
    """提取测试集所有样本的embedding特征"""
    print(f"\n🔍 提取测试集embedding特征...")

    model.eval()
    all_embeddings = []
    all_labels = []
    num_samples = len(test_data["labels"])

    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, num_samples)

            batch_flow = test_data["features_flow"][i:end_idx]
            batch_seq = test_data["features_seq"][i:end_idx]
            batch_labels = test_data["labels"][i:end_idx]

            batch_embeddings = model(batch_flow, batch_seq)

            all_embeddings.append(batch_embeddings.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)

    print(f"✅ Embedding提取完成")
    print(f"   Embedding形状: {all_embeddings.shape}")
    print(f"   标签形状: {all_labels.shape}")

    return all_embeddings, all_labels


# =====================================================
# 评估指标计算函数
# =====================================================
def intra_class_radius(embeddings, labels, target_class):
    """指标1：类内紧致性"""
    print(f"\n📊 计算类内紧致性 (目标类: {INV_LABEL_MAP[target_class]})...")

    e = embeddings[labels == target_class]
    if len(e) == 0:
        raise ValueError(f"无目标类({target_class})样本，无法计算类内紧致性")

    center = e.mean(axis=0)
    distances = np.linalg.norm(e - center, axis=1)
    avg_radius = distances.mean()
    std_radius = distances.std()
    max_radius = distances.max()
    min_radius = distances.min()

    print(f"   样本数: {len(e)}")
    print(f"   平均半径: {avg_radius:.4f}")
    print(f"   半径标准差: {std_radius:.4f}")

    return {
        "avg_radius": avg_radius,
        "std_radius": std_radius,
        "max_radius": max_radius,
        "min_radius": min_radius,
        "center": center,
        "distances": distances,
        "sample_count": len(e)
    }


def class_margin(embeddings, labels, target_class):
    """指标2：类间Margin"""
    print(f"\n📊 计算类间Margin (目标类: {INV_LABEL_MAP[target_class]} vs 其他类)...")

    e_pos = embeddings[labels == target_class]
    e_neg = embeddings[labels != target_class]

    if len(e_pos) == 0 or len(e_neg) == 0:
        raise ValueError("目标类或非目标类无样本，无法计算Margin")

    # 计算类中心
    c_pos = e_pos.mean(axis=0)
    c_neg = e_neg.mean(axis=0)

    # 类内平均距离
    intra_distances = np.linalg.norm(e_pos - c_pos, axis=1)
    intra = intra_distances.mean()

    # 类间平均距离
    inter_distances = np.linalg.norm(e_pos - c_neg, axis=1)
    inter = inter_distances.mean()

    # Margin计算
    margin = inter - intra

    # 计算每个目标类样本的Margin
    sample_margins = inter_distances - intra_distances

    print(f"   目标类样本数: {len(e_pos)}")
    print(f"   非目标类样本数: {len(e_neg)}")
    print(f"   类内平均距离: {intra:.4f}")
    print(f"   类间平均距离: {inter:.4f}")
    print(f"   Margin: {margin:.4f}")

    return {
        "margin": margin,
        "intra_distance": intra,
        "inter_distance": inter,
        "intra_std": intra_distances.std(),
        "inter_std": inter_distances.std(),
        "c_pos": c_pos,
        "c_neg": c_neg,
        "sample_margins": sample_margins
    }


def prototype_distance_stats(embeddings, labels, target_class, prototypes):
    """指标3：Prototype距离分布统计"""
    print(f"\n📊 计算目标类样本到原型距离统计...")

    # 获取目标类原型
    proto = prototypes[target_class]
    proto_np = proto.cpu().numpy()

    # 获取测试集中目标类样本的embedding
    e_pos = embeddings[labels == target_class]

    if len(e_pos) == 0:
        raise ValueError(f"测试集中无目标类({target_class})样本")

    # 计算距离
    dists = np.linalg.norm(e_pos - proto_np, axis=1)

    # 统计
    mean_dist = dists.mean()
    p90_dist = np.percentile(dists, 90)
    std_dist = dists.std()

    print(f"   样本数: {len(dists)}")
    print(f"   平均距离: {mean_dist:.4f}")
    print(f"   90%分位数: {p90_dist:.4f}")
    print(f"   标准差: {std_dist:.4f}")

    return {
        "mean": mean_dist,
        "p90": p90_dist,
        "std": std_dist,
        "dists": dists,
        "proto": proto_np,
        "sample_count": len(dists)
    }


def calculate_silhouette_score(embeddings, labels):
    """指标4：轮廓系数"""
    print(f"\n📊 计算Embedding轮廓系数...")

    # 标准化embedding
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 计算轮廓系数
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)

    score = silhouette_score(
        embeddings_scaled[sample_indices],
        labels[sample_indices],
        metric='euclidean'
    )

    print(f"   轮廓系数: {score:.4f}")
    print(f"   计算样本数: {sample_size}")

    return {
        "silhouette_score": score,
        "sample_size": sample_size
    }


# =====================================================
# 多次运行验证函数
# =====================================================
def verify_with_multiple_runs(model, train_data, test_data_tensor, num_runs=5):
    """多次运行验证准确率的稳定性"""
    print("\n" + "=" * 80)
    print("多次运行验证准确率稳定性")
    print("=" * 80)

    accuracies = []
    all_prototypes = []

    for run in range(num_runs):
        print(f"\n🔄 第 {run + 1} 次运行:")

        # 重置随机种子，但每次用不同的种子
        random.seed(42 + run)
        np.random.seed(42 + run)

        results = evaluate_accuracy_as_feature_evaluate2(model, test_data_tensor, train_data)
        accuracies.append(results["accuracy"])

        # 保存原型向量用于比较
        all_prototypes.append(results["prototypes"])

        # 打印Web类原型向量的前几个元素
        web_proto = results["prototypes"][WEB_CLASS_ID][:5].cpu().numpy()
        print(f"   Web类原型向量前5维: {web_proto}")

    print(f"\n📊 多次运行结果统计:")
    print(f"   准确率列表: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"   平均准确率: {np.mean(accuracies):.4f}")
    print(f"   标准差: {np.std(accuracies):.4f}")
    print(f"   最小值: {np.min(accuracies):.4f}")
    print(f"   最大值: {np.max(accuracies):.4f}")

    # 检查原型是否相同
    proto_diffs = []
    for i in range(1, num_runs):
        proto1 = all_prototypes[0][WEB_CLASS_ID]
        proto2 = all_prototypes[i][WEB_CLASS_ID]
        diff = torch.norm(proto1 - proto2).item()
        proto_diffs.append(diff)

    print(f"\n   Web类原型差异:")
    print(f"   第1次与第2-{num_runs}次的平均L2距离: {np.mean(proto_diffs):.6f}")

    return accuracies, all_prototypes


# =====================================================
# 保存结果函数
# =====================================================
def save_results(metrics, accuracies=None):
    """保存评估结果"""
    result_path = os.path.join(RESULT_DIR, "evaluation_results.txt")

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("增强模型(enhanced_model_final.pth)综合评估报告\n")
        f.write("=" * 80 + "\n\n")

        # 1. 准确率评估
        f.write("1. 准确率评估\n")
        f.write("-" * 40 + "\n")
        f.write(f"   整体准确率: {metrics['accuracy_results']['accuracy']:.4f}\n")
        f.write(f"   类别准确率:\n")
        for cls, acc in metrics['accuracy_results']['class_accuracy'].items():
            f.write(f"     {cls}: {acc:.4f}\n")
        f.write("\n")

        # 如果有多次运行结果
        if accuracies is not None and len(accuracies) > 1:
            f.write("   多次运行统计:\n")
            f.write(f"     平均准确率: {np.mean(accuracies):.4f}\n")
            f.write(f"     标准差: {np.std(accuracies):.4f}\n")
            f.write(f"     最小值: {np.min(accuracies):.4f}\n")
            f.write(f"     最大值: {np.max(accuracies):.4f}\n\n")

        # 2. Web类内紧致性
        f.write("2. Web类内紧致性\n")
        f.write("-" * 40 + "\n")
        f.write(f"   样本数: {metrics['web_compactness']['sample_count']}\n")
        f.write(f"   平均半径: {metrics['web_compactness']['avg_radius']:.4f}\n")
        f.write(f"   半径标准差: {metrics['web_compactness']['std_radius']:.4f}\n")
        f.write(f"   最大半径: {metrics['web_compactness']['max_radius']:.4f}\n")
        f.write(f"   最小半径: {metrics['web_compactness']['min_radius']:.4f}\n\n")

        # 3. Web vs 非Web Margin
        f.write("3. Web vs 非Web Margin\n")
        f.write("-" * 40 + "\n")
        f.write(f"   类内平均距离: {metrics['web_margin']['intra_distance']:.4f}\n")
        f.write(f"   类间平均距离: {metrics['web_margin']['inter_distance']:.4f}\n")
        f.write(f"   Margin: {metrics['web_margin']['margin']:.4f}\n\n")

        # 4. Web到原型距离统计
        f.write("4. Web样本到原型距离统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"   样本数: {metrics['web_prototype_dist']['sample_count']}\n")
        f.write(f"   平均距离: {metrics['web_prototype_dist']['mean']:.4f}\n")
        f.write(f"   90%分位数: {metrics['web_prototype_dist']['p90']:.4f}\n")
        f.write(f"   标准差: {metrics['web_prototype_dist']['std']:.4f}\n\n")

        # 5. 轮廓系数
        f.write("5. Embedding轮廓系数\n")
        f.write("-" * 40 + "\n")
        f.write(f"   轮廓系数: {metrics['silhouette']['silhouette_score']:.4f}\n")
        f.write(f"   计算样本数: {metrics['silhouette']['sample_size']}\n\n")

        # 6. 混淆矩阵
        f.write("6. 混淆矩阵（归一化）\n")
        f.write("-" * 40 + "\n")
        cm_norm = metrics['accuracy_results']['confusion_matrix_norm']
        cm_df = pd.DataFrame(cm_norm,
                             index=[f'真实_{cls}' for cls in CLASSES],
                             columns=[f'预测_{cls}' for cls in CLASSES])
        f.write(cm_df.to_string())
        f.write("\n\n")

        # 7. 分类报告
        f.write("7. 详细分类报告\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(
            metrics['accuracy_results']['true_labels'],
            metrics['accuracy_results']['predictions'],
            target_names=CLASSES,
            digits=4
        ))

    print(f"\n✅ 评估报告已保存至: {result_path}")


# =====================================================
# 主函数
# =====================================================
def main():
    print("=" * 80)
    print("增强模型(enhanced_model_final.pth)综合评估 - 修正版")
    print("=" * 80)

    try:
        # 1. 加载数据
        train_data, test_data = load_data()
        test_data_tensor = prepare_tensors(test_data)

        # 2. 加载模型
        model = load_model()

        # 3. 设置随机种子
        random.seed(42)
        np.random.seed(42)

        # 4. 评估准确率（按照feature_evaluate2.py的方式）
        accuracy_results = evaluate_accuracy_as_feature_evaluate2(
            model, test_data_tensor, train_data
        )

        # 5. 提取测试集embeddings
        embeddings, labels = extract_embeddings(model, test_data_tensor)

        # 6. 计算各项评估指标
        print("\n" + "=" * 80)
        print("计算详细评估指标")
        print("=" * 80)

        web_compactness = intra_class_radius(embeddings, labels, WEB_CLASS_ID)
        web_margin = class_margin(embeddings, labels, WEB_CLASS_ID)
        web_prototype_dist = prototype_distance_stats(
            embeddings, labels, WEB_CLASS_ID, accuracy_results["prototypes"]
        )
        silhouette = calculate_silhouette_score(embeddings, labels)

        # 7. 汇总所有指标
        metrics = {
            "accuracy_results": accuracy_results,
            "web_compactness": web_compactness,
            "web_margin": web_margin,
            "web_prototype_dist": web_prototype_dist,
            "silhouette": silhouette
        }

        # 8. 保存结果
        save_results(metrics)

        # 9. 多次运行验证（可选）
        print("\n" + "=" * 80)
        print("是否进行多次运行验证？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            accuracies, _ = verify_with_multiple_runs(model, train_data, test_data_tensor, num_runs=5)
            save_results(metrics, accuracies)

        # 10. 输出最终总结
        print("\n" + "=" * 80)
        print("评估完成 - 结果摘要")
        print("=" * 80)

        print(f"\n📊 核心指标:")
        print(f"   整体准确率: {accuracy_results['accuracy']:.4f}")
        print(f"   Web类内平均半径: {web_compactness['avg_radius']:.4f}")
        print(f"   Web Margin: {web_margin['margin']:.4f}")
        print(f"   Web到原型平均距离: {web_prototype_dist['mean']:.4f}")
        print(f"   轮廓系数: {silhouette['silhouette_score']:.4f}")

        print(f"\n📁 所有结果已保存至: {RESULT_DIR}")

    except Exception as e:
        print(f"\n❌ 评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()