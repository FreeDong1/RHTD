
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
# =====================================================
# 全局配置（核心新增：不同微调轮数配置）
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 特征维度配置（32列seq特征 + 1列标签）
SEQ_DIM = 32
NUM_CLASSES = 5

# 鲁棒性测试配置
TRAIN_SAMPLE_COUNTS = [5, 10, 15, 20,25]  # 每个类别抽取的训练样本数
FINETUNE_EPOCHS_LIST = [5, 10, 15, 20, 25]  # 新增：测试不同微调轮数
MAX_TEST_ROWS_PER_CLASS = 400  # 每类仅取前400行作为测试数据

# 路径配置
MODEL_PATH = "saved_models/enhanced_model_final.pth"  # 预训练GAN增强模型路径
ORIGINAL_TEST_CSV = "./Metalearn_evalu/data/originData.csv"  # 替换fixed_test_data.pkl的新文件
NETWORK_CSV_PATH = "./Metalearn_evalu/data/BadNetwork.csv"  # 鲁棒性测试用数据
RESULT_SAVE_PATH = "./Metalearn_evalu/robustness_test_results.csv"  # 测试结果保存路径
DETAILED_RESULT_PATH = "./Metalearn_evalu/detailed_finetune_results.csv"  # 新增：不同轮数详细结果

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


# =====================================================
# 模型定义
# =====================================================
class FeatureEncoder(nn.Module):
    """特征提取器"""

    def __init__(self, input_dim=SEQ_DIM, output_dim=64):
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

    def __init__(self, feature_dim=64):
        super().__init__()
        self.seq_encoder = FeatureEncoder(SEQ_DIM, 64)
        self.feature_dim = feature_dim

    def forward(self, seq):
        seq_features = self.seq_encoder(seq)
        return seq_features

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
# 数据加载与预处理
# =====================================================
def load_1_csv_test_data(csv_path, max_test_rows_per_class=MAX_TEST_ROWS_PER_CLASS):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv不存在: {csv_path}")

    print(f"\n📥 加载并处理1.csv数据...")
    df = pd.read_csv(csv_path)

    if df.shape[1] != 33:
        raise ValueError(f"csv列数错误")

    # 分离特征和标签
    X_seq = df.iloc[:, :SEQ_DIM].values.astype(np.float32)  # 前32列=特征
    y = df.iloc[:, SEQ_DIM].values.astype(int)  # 第33列=标签

    # 特征归一化
    seq_scaler = StandardScaler()
    X_seq = seq_scaler.fit_transform(X_seq)

    # 按类别整理数据并仅保留每类前400行
    class_data = {c: [] for c in CLASSES}
    for s, label in zip(X_seq, y):
        cls = INV_LABEL_MAP.get(label)
        if cls in CLASSES:
            # 仅保留每类前400行
            if len(class_data[cls]) < max_test_rows_per_class:
                class_data[cls].append(s)

    # 构建测试数据集
    test_data = {
        "features_seq": [],
        "labels": [],
        "class_names": []
    }
    for cls in CLASSES:
        samples = class_data[cls]
        if not samples:
            print(f"⚠️ 1.csv中{cls}类别无数据")
            continue

        # 填充测试数据
        for s in samples:
            test_data["features_seq"].append(s)
            test_data["labels"].append(LABEL_MAP[cls])
            test_data["class_names"].append(cls)

    # 转换为numpy数组
    test_data["features_seq"] = np.array(test_data["features_seq"], dtype=np.float32)
    test_data["labels"] = np.array(test_data["labels"], dtype=int)

    # 打印统计
    print(f"✅ 1.csv处理完成:")
    print(f"   - 每类仅保留前{max_test_rows_per_class}行")
    print(f"   - 总测试样本数: {len(test_data['labels'])}")
    print(f"   - 类别分布: {Counter(test_data['class_names'])}")
    print(f"   - 特征维度: {test_data['features_seq'].shape}")

    return test_data


def load_and_process_network_data(csv_path, max_test_rows_per_class=MAX_TEST_ROWS_PER_CLASS):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"network.csv不存在: {csv_path}")

    print(f"\n📥 加载并处理network.csv数据...")
    df = pd.read_csv(csv_path)

    # 验证列数：必须33列（32特征+1标签）
    if df.shape[1] != 33:
        raise ValueError(f"network.csv列数错误，期望33列（32特征+1标签），实际{df.shape[1]}列")

    # 分离特征和标签
    X_seq = df.iloc[:, :SEQ_DIM].values.astype(np.float32)  # 前32列=特征
    y = df.iloc[:, SEQ_DIM].values.astype(int)  # 第33列=标签

    # 特征归一化
    seq_scaler = StandardScaler()
    X_seq = seq_scaler.fit_transform(X_seq)

    # 按类别整理所有数据
    all_class_data = {c: [] for c in CLASSES}
    for s, label in zip(X_seq, y):
        cls = INV_LABEL_MAP.get(label)
        if cls in CLASSES:
            all_class_data[cls].append(s)

    # 拆分：每类前400行=测试数据，剩余=训练池
    train_pool = {c: [] for c in CLASSES}
    test_set = {
        "features_seq": [],
        "labels": [],
        "class_names": []
    }

    for cls in CLASSES:
        all_samples = all_class_data[cls]
        if not all_samples:
            print(f"⚠️ network.csv中{cls}类别无数据")
            continue

        # 每类前400行作为测试数据
        test_samples = all_samples[:max_test_rows_per_class]
        # 剩余作为训练池
        train_samples = all_samples[max_test_rows_per_class:]

        # 填充测试集
        for s in test_samples:
            test_set["features_seq"].append(s)
            test_set["labels"].append(LABEL_MAP[cls])
            test_set["class_names"].append(cls)

        # 填充训练池
        train_pool[cls] = train_samples

    # 转换为numpy数组
    test_set["features_seq"] = np.array(test_set["features_seq"], dtype=np.float32)
    test_set["labels"] = np.array(test_set["labels"], dtype=int)

    # 打印统计
    print(f"✅ network.csv处理完成:")
    print(f"   - 每类仅取前{max_test_rows_per_class}行作为测试数据")
    print(f"   - 测试集样本数: {len(test_set['labels'])}")
    print(f"   - 测试集类别分布: {Counter(test_set['class_names'])}")
    print(f"   - 训练池样本数:")
    for cls in CLASSES:
        print(f"     {cls}: {len(train_pool[cls])}")

    return train_pool, test_set


def sample_train_data(train_pool, n_samples_per_class):
    train_data = {
        "features_seq": [],
        "labels": []
    }

    for cls in CLASSES:
        samples = train_pool[cls]
        if len(samples) < n_samples_per_class:
            print(f"⚠️ {cls}类别训练池样本不足，仅能抽取{len(samples)}个（要求{n_samples_per_class}个）")
            selected = samples
        else:
            selected = random.sample(samples, n_samples_per_class)

        # 填充训练数据
        for s in selected:
            train_data["features_seq"].append(s)
            train_data["labels"].append(LABEL_MAP[cls])

    # 转换为张量
    train_data["features_seq"] = torch.tensor(train_data["features_seq"], dtype=torch.float32).to(DEVICE)
    train_data["labels"] = torch.tensor(train_data["labels"], dtype=torch.long).to(DEVICE)

    print(f"\n🎯 抽取训练样本完成: 每个类别{n_samples_per_class}个，总样本数{len(train_data['labels'])}")
    return train_data


# =====================================================
# 模型评估函数
# =====================================================
def evaluate_model(model, test_data, data_name):
    """通用模型评估函数"""
    model.eval()
    all_preds = []
    all_labels = test_data["labels"]
    num_samples = len(all_labels)

    if num_samples == 0:
        print(f"❌ {data_name}无样本，评估失败")
        return 0.0, None

    # 批量处理
    batch_size = 64

    with torch.no_grad():
        # 1. 计算原型（使用测试数据中的样本）
        all_support_features = []
        all_support_labels = []

        # 为每个类别抽取部分样本计算原型
        for cls in CLASSES:
            cls_mask = np.array(test_data["class_names"]) == cls
            cls_samples_seq = test_data["features_seq"][cls_mask]

            if len(cls_samples_seq) > 20:
                sample_idx = random.sample(range(len(cls_samples_seq)), 20)
            else:
                sample_idx = range(len(cls_samples_seq))

            for idx in sample_idx:
                s = cls_samples_seq[idx]
                features = model(torch.tensor([s], dtype=torch.float32).to(DEVICE))
                all_support_features.append(features)
                all_support_labels.append(LABEL_MAP[cls])

        if not all_support_features:
            print(f"❌ 无法计算原型，{data_name}评估失败")
            return 0.0, None

        support_features = torch.cat(all_support_features, dim=0)
        support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
        prototypes = model.compute_prototypes(support_features, support_labels)

        # 2. 预测测试数据
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)

            batch_seq = torch.tensor(test_data["features_seq"][i:end_idx]).to(DEVICE)
            batch_features = model(batch_seq)
            batch_preds, _ = model.predict(batch_features, prototypes)

            all_preds.extend(batch_preds.cpu().numpy())

    # 计算准确率和分类报告
    all_preds = np.array(all_preds)
    accuracy = np.mean(all_preds == all_labels)

    print(f"\n🎯 {data_name}评估结果:")
    print(f"   样本数: {num_samples}")
    print(f"   整体准确率: {accuracy:.4f}")

    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📊 混淆矩阵:")
    print("True\\Pred", end="")
    for cls in CLASSES:
        print(f"{cls:>8}", end="")
    print()
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>8}", end="")
        for j in range(len(CLASSES)):
            print(f"{cm[i][j]:>8}", end="")
        print()

    print("\n📋 详细分类报告:")
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    return accuracy, report


# =====================================================
# 模型微调函数（适配不同轮数）
# =====================================================
def fine_tune_model(model, train_data, epochs):
    """用少量样本微调模型（支持指定不同微调轮数）"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 准备训练数据
    seq = train_data["features_seq"]
    labels = train_data["labels"]

    # 记录每轮训练的损失和准确率
    train_history = {
        "loss": [],
        "accuracy": []
    }

    # 训练循环（指定轮数）
    print(f"\n🔧 开始微调模型（{epochs}轮）...")
    for epoch in range(epochs):
        # 前向传播
        features = model(seq)

        # 计算原型
        prototypes = model.compute_prototypes(features, labels)

        # 计算损失（自监督方式）
        all_distances = []
        for feat in features:
            dists = []
            for label in prototypes.keys():
                dist = torch.norm(feat - prototypes[label], p=2)
                dists.append(-dist)
            all_distances.append(torch.stack(dists))

        logits = torch.stack(all_distances)
        loss = loss_fn(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 计算训练准确率
        preds, _ = model.predict(features, prototypes)
        train_acc = (preds == labels).float().mean().item()

        # 记录历史
        train_history["loss"].append(loss.item())
        train_history["accuracy"].append(train_acc)

        # 每5轮打印一次进度
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

    print("✅ 模型微调完成")
    return model, train_history


# =====================================================
# 主测试流程（核心新增：不同微调轮数测试）
# =====================================================
def main():
    print("=" * 80)
    print("模型鲁棒性测试（不同微调轮数：5/10/15/20/25）")
    print("=" * 80)

    # 初始化结果字典
    results = {
        "baseline_network_test_acc": 0.0,  # 新增：基线 - 原始模型在network.csv上的准确率
        "baseline_1csv_test_acc": 0.0,  # 新增：基线 - 原始模型在1.csv上的准确率
        "detailed_results": []  # 新增：存储不同样本数+不同轮数的详细结果
    }

    # =====================================================
    # 步骤1：加载预训练模型
    # =====================================================
    print("\n" + "=" * 80)
    print("步骤1：加载预训练GAN增强模型")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"预训练模型不存在: {MODEL_PATH}")

    model = PrototypicalNetwork().to(DEVICE)
    # 加载模型（过滤可能的flow_encoder参数）
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("flow_encoder")}
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"✅ 成功加载预训练模型: {MODEL_PATH}")


    print("\n" + "=" * 80)
    print("步骤2：加载并处理测试数据（每类仅前400行）")
    print("=" * 80)

   
    csv1_test_data = load_1_csv_test_data(ORIGINAL_TEST_CSV)
    network_train_pool, network_test_data = load_and_process_network_data(NETWORK_CSV_PATH)
    print("\n" + "=" * 80)
    print("步骤3：测试基线性能（原始模型在测试集上的表现）")
    print("=" * 80)

   
    print("\n📊 基线1: 原始模型在network.csv测试集上的表现")
    baseline_network_acc, baseline_network_report = evaluate_model(
        model, network_test_data,
        "基线模型-network.csv测试集（每类前400行）"
    )
    results["baseline_network_test_acc"] = baseline_network_acc

 
    print("\n" + "-" * 50)
    print("📈 network.csv基线混淆矩阵总结:")
    print("-" * 50)

  
    print("\n📊 基线2: 原始模型在1.csv测试集上的表现")
    baseline_1csv_acc, baseline_1csv_report = evaluate_model(
        model, csv1_test_data,
        "基线模型-1.csv测试集（每类前400行）"
    )
    results["baseline_1csv_test_acc"] = baseline_1csv_acc

    
    print("\n📋 基线性能总结:")
    print(f"   network.csv测试集准确率: {baseline_network_acc:.4f}")
    print(f"   1.csv测试集准确率: {baseline_1csv_acc:.4f}")

    # =====================================================
    #不同样本数+不同微调轮数的测试（核心新增）
    # =====================================================
    print("\n" + "=" * 80)
    print("步骤4：不同样本数+不同微调轮数测试")
    print("=" * 80)

    for n_samples in TRAIN_SAMPLE_COUNTS:
        print(f"\n" + "=" * 60)
        print(f"测试：每个类别抽取{n_samples}个样本")
        print("=" * 60)

        # 4.1 抽取指定数量的训练样本（一次抽取，多次微调）
        train_data = sample_train_data(network_train_pool, n_samples)

        for n_epochs in FINETUNE_EPOCHS_LIST:
            print(f"\n" + "-" * 50)
            print(f"子测试：微调{n_epochs}轮")
            print("-" * 50)

            # 4.2 复制模型（避免不同轮数相互影响）
            fine_tune_model_copy = PrototypicalNetwork().to(DEVICE)
            fine_tune_model_copy.load_state_dict(model.state_dict(), strict=False)

            # 4.3 微调模型（指定轮数）
            fine_tune_model_copy, train_history = fine_tune_model(fine_tune_model_copy, train_data, n_epochs)

            # 4.4 测试微调后的性能
            # 测试network.csv测试集
            ft_network_acc, ft_network_report = evaluate_model(
                fine_tune_model_copy, network_test_data,
                f"微调{n_epochs}轮-network.csv测试集（{n_samples}样本/类）"
            )

            # 测试1.csv测试集（验证泛化性）
            ft_1csv_acc, _ = evaluate_model(
                fine_tune_model_copy, csv1_test_data,
                f"微调{n_epochs}轮-1.csv测试集（{n_samples}样本/类）"
            )

            # 计算相对于基线的提升值（新增）
            network_improvement = ft_network_acc - baseline_network_acc
            csv1_change = ft_1csv_acc - baseline_1csv_acc

            # 记录详细结果（新增提升值字段）
            detailed_result = {
                "样本数_每类": n_samples,
                "微调轮数": n_epochs,
                "network测试集准确率": ft_network_acc,
                "相对于network基线提升": network_improvement,  # 新增

            }
            results["detailed_results"].append(detailed_result)

            # 打印本轮小结
            print(f"\n📝 本轮小结（{n_samples}样本/类，{n_epochs}轮微调）:")
            print(
                f"   network.csv准确率: {ft_network_acc:.4f} (基线{baseline_network_acc:.4f}, 提升{network_improvement:+.4f})")
            print(f"   1.csv准确率: {ft_1csv_acc:.4f} (基线{baseline_1csv_acc:.4f}, 变化{csv1_change:+.4f})")
            print(f"   最后一轮训练损失: {train_history['loss'][-1]:.4f}")
            print(f"   最后一轮训练准确率: {train_history['accuracy'][-1]:.4f}")

    # =====================================================
    # 结果汇总与保存
    # =====================================================
    print("\n" + "=" * 80)
    print("步骤5：结果汇总与保存")
    print("=" * 80)

    # 打印基线结果
    print(f"\n📊 基线性能:")
    print(f"  1.csv测试集（每类前400行）准确率: {results['baseline_1csv_test_acc']:.5f}")
    print(f"  network.csv测试集（每类前400行）准确率: {results['baseline_network_test_acc']:.5f}")

    # 打印详细结果汇总
    print(f"\n📈 不同参数组合性能汇总:")
    detailed_df = pd.DataFrame(results["detailed_results"])
    print(detailed_df.to_string(index=False))

    #  保存详细结果到CSV
    detailed_df.to_csv(DETAILED_RESULT_PATH, index=False, encoding='utf-8')
    print(f"\n✅ 详细结果已保存至: {DETAILED_RESULT_PATH}")

    # 生成汇总报告
    summary_df = detailed_df.pivot_table(
        index="样本数_每类",
        columns="微调轮数",
        values=["network测试集准确率"],
        aggfunc="mean"
    )
    # 保存汇总报告
    summary_df.to_csv(RESULT_SAVE_PATH, encoding='utf-8')
    



if __name__ == "__main__":
    # 设置随机种子（保证结果可复现）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()