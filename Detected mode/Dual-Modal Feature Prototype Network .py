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
import pickle  # 用于保存/加载测试数据集对象

# =====================================================
# 全局配置
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLOW_DIM = 60
SEQ_DIM = 32
NUM_CLASSES = 5

# 少样本配置
K_SHOT = 10
Q_QUERY = 10

TASKS_PER_BATCH = 12
META_LR = 3e-4
META_EPOCHS = 5
TARGET_ACC = 0.68  # 目标准确率阈值

# 测试数据配置（仅基于稳定域all.csv）
TEST_DATA_DIR = "test_data"  # 测试数据保存目录
TRAIN_DATA_DIR = "train_data"  # 训练数据保存目录
TEST_SPLIT_RATIO = 0.2  # 测试集占比（20%）
TEST_SAMPLE_SIZE = 400  # 每个类别最大测试样本数

# 路径配置
TEST_DATA_SAVE_PATH = os.path.join(TEST_DATA_DIR, "fixed_test_data.pkl")
TEST_CSV_SAVE_PATH = os.path.join(TEST_DATA_DIR, "fixed_test_data.csv")
TRAIN_DATA_SAVE_PATH = os.path.join(TRAIN_DATA_DIR, "train_data.pkl")

# =====================================================
# 标签映射
# =====================================================
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
# 核心：数据拆分（仅基于all.csv，训练/测试完全隔离）
# =====================================================
def split_train_test_data(raw_data, test_ratio=TEST_SPLIT_RATIO,
                          test_size=TEST_SAMPLE_SIZE, force_rebuild=False):
    """
    拆分all.csv数据为训练集和测试集（完全隔离）
    参数:
        raw_data: 仅包含稳定域的原始数据字典
        test_ratio: 测试集占比
        test_size: 每个类别最大测试样本数
        force_rebuild: 是否强制重新拆分
    返回:
        train_data: 训练集（无测试数据）
        test_data: 测试集（纯独立数据）
    """
    # 确保目录存在
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

    # 如果已有拆分好的数据且不强制重建，直接加载
    if os.path.exists(TEST_DATA_SAVE_PATH) and os.path.exists(TRAIN_DATA_SAVE_PATH) and not force_rebuild:
        print(f"\n📂 发现已拆分的训练/测试数据，直接加载...")
        with open(TEST_DATA_SAVE_PATH, 'rb') as f:
            test_data = pickle.load(f)
        with open(TRAIN_DATA_SAVE_PATH, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✅ 训练/测试数据加载完成")
        return train_data, test_data

    print(f"\n🔧 开始拆分数据（仅基于all.csv，训练/测试完全隔离）...")
    print(f"   测试集占比: {test_ratio}, 每个类别最大测试样本数: {test_size}")

    # 初始化训练/测试数据结构
    train_data = {c: [] for c in CLASSES}
    test_data = {
        "features_flow": [],  # 测试集flow特征
        "features_seq": [],  # 测试集seq特征
        "labels": [],  # 测试集标签（数字）
        "class_names": []  # 测试集类别名称
    }

    # 对每个类别进行数据拆分
    for cls in CLASSES:
        # 获取该类别的所有样本
        all_samples = raw_data[cls].copy()
        if not all_samples:
            print(f"⚠️ {cls}类别无数据，跳过拆分")
            train_data[cls] = all_samples
            continue

        # 计算测试样本数量（取比例和最大数的较小值）
        n_test = min(int(len(all_samples) * test_ratio), test_size)
        if n_test == 0:
            print(f"⚠️ {cls}类别样本数不足，测试集数量设为1")
            n_test = 1

        # 随机打乱样本（保证随机性）
        random.shuffle(all_samples)

        # 拆分：前n_test个为测试集，剩余为训练集
        test_samples = all_samples[:n_test]
        train_samples = all_samples[n_test:]

        # 填充测试集
        for f, s in test_samples:
            test_data["features_flow"].append(f)
            test_data["features_seq"].append(s)
            test_data["labels"].append(LABEL_MAP[cls])
            test_data["class_names"].append(cls)

        # 填充训练集
        train_data[cls] = train_samples

    # 转换测试集为numpy数组
    test_data["features_flow"] = np.array(test_data["features_flow"], dtype=np.float32)
    test_data["features_seq"] = np.array(test_data["features_seq"], dtype=np.float32)
    test_data["labels"] = np.array(test_data["labels"], dtype=int)

    # 保存拆分后的数据（持久化）
    with open(TEST_DATA_SAVE_PATH, 'wb') as f:
        pickle.dump(test_data, f)
    with open(TRAIN_DATA_SAVE_PATH, 'wb') as f:
        pickle.dump(train_data, f)

    # 保存测试集CSV备份
    save_test_data_to_csv(test_data, TEST_CSV_SAVE_PATH)

    # 打印拆分统计
    print(f"\n📊 数据拆分完成:")
    print(f"   测试集总样本数: {len(test_data['labels'])}")
    print(f"   测试集类别分布: {Counter(test_data['class_names'])}")
    print(f"   训练集样本数:")
    for cls in CLASSES:
        print(f"     {cls}: {len(train_data[cls])}")
    print(f"✅ 训练数据已保存至: {TRAIN_DATA_SAVE_PATH}")
    print(f"✅ 测试数据已保存至: {TEST_DATA_SAVE_PATH}")

    return train_data, test_data


def save_test_data_to_csv(test_data, csv_path):
    """保存测试集为CSV格式（便于验证）"""
    # 拼接特征
    all_features = np.hstack([
        test_data["features_flow"],
        test_data["features_seq"]
    ])

    # 构造DataFrame
    feature_cols = [f"flow_{i}" for i in range(FLOW_DIM)] + [f"seq_{i}" for i in range(SEQ_DIM)]
    df = pd.DataFrame(all_features, columns=feature_cols)
    df["label"] = test_data["labels"]
    df["class_name"] = test_data["class_names"]

    # 保存
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"📄 测试集CSV备份已保存至: {csv_path}")


def load_test_data(load_path=TEST_DATA_SAVE_PATH):
    """加载独立的测试集"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"测试集不存在，请先运行数据拆分: {load_path}")

    with open(load_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"✅ 成功加载独立测试集: {load_path}")
    print(f"   测试集样本数: {len(test_data['labels'])}")
    return test_data


def load_train_data(load_path=TRAIN_DATA_SAVE_PATH):
    """加载隔离后的训练集"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"训练集不存在，请先运行数据拆分: {load_path}")

    with open(load_path, 'rb') as f:
        train_data = pickle.load(f)
    print(f"✅ 成功加载隔离训练集: {load_path}")
    return train_data


# =====================================================
# 数据加载函数
# =====================================================
def load_raw_data(stable_csv, web_gan_csv=None, use_gan=False):
    """
    仅加载all.csv原始数据，不处理network.csv
    参数:
        stable_csv: all.csv路径
        web_gan_csv: GAN增强数据路径
        use_gan: 是否加载GAN数据
    返回:
        raw_data: 字典 {类别: [(flow特征, seq特征)]}
        gan_data: GAN增强数据（仅web类别）
    """
    print("📥 加载all.csv原始数据并进行预处理...")

    raw_data = {c: [] for c in CLASSES}
    gan_data = []

    # 1. 加载all.csv数据
    df_stable = pd.read_csv(stable_csv)
    X_flow = df_stable.iloc[:, :FLOW_DIM].values.astype(np.float32)
    X_seq = df_stable.iloc[:, FLOW_DIM:FLOW_DIM + SEQ_DIM].values.astype(np.float32)
    y = df_stable.iloc[:, FLOW_DIM + SEQ_DIM].values.astype(int)

    # 2. 特征归一化
    # flow特征归一化
    flow_scaler = StandardScaler()
    X_flow = flow_scaler.fit_transform(X_flow)

    # seq特征归一化
    seq_scaler = StandardScaler()
    X_seq = seq_scaler.fit_transform(X_seq)

    # 3. 填充原始数据
    for f, s, label in zip(X_flow, X_seq, y):
        cls = INV_LABEL_MAP.get(label)
        if cls in CLASSES:
            raw_data[cls].append((f, s))

    # 4. 加载GAN数据（如果使用）
    if use_gan and web_gan_csv:
        print("   加载GAN增强数据（仅web类别）...")
        try:
            df_gan = pd.read_csv(web_gan_csv)
            X_seq_gan = df_gan.iloc[:, :SEQ_DIM].values.astype(np.float32)
            X_seq_gan = seq_scaler.transform(X_seq_gan)

            # GAN数据仅补充web类别，flow特征补零
            zero_flow_gan = np.zeros((len(X_seq_gan), FLOW_DIM), dtype=np.float32)
            for f, s in zip(zero_flow_gan, X_seq_gan):
                gan_data.append((f, s))

            print(f"   GAN增强样本数: {len(gan_data)}")
        except Exception as e:
            print(f"⚠️ 加载GAN数据失败: {e}")
            gan_data = []

    # 打印原始数据统计
    print(f"\n原始数据统计（仅all.csv）:")
    print(f"  总样本数: {sum(len(raw_data[c]) for c in CLASSES)}")
    print(f"  类别分布: {Counter([cls for cls in CLASSES for _ in raw_data[cls]])}")

    return raw_data, gan_data


# =====================================================
# 模型评估函数（仅使用独立测试集）
# =====================================================
def evaluate_model_on_isolated_test(model, test_data):
    """
    使用完全隔离的测试集评估模型（确保无数据泄露）
    参数:
        model: 待评估模型
        test_data: 独立测试集
    返回:
        accuracy: 整体准确率
    """
    model.eval()
    all_preds = []
    all_labels = test_data["labels"]
    num_samples = len(all_labels)

    # 批量处理（避免显存溢出）
    batch_size = 64

    with torch.no_grad():
        # 1. 计算原型（仅使用训练集数据）
        print(f"\n📝 使用训练集计算类别原型（无测试数据参与）...")
        train_data = load_train_data()
        all_support_features = []
        all_support_labels = []

        for cls in CLASSES:
            # 仅从训练集取样本计算原型
            train_samples = train_data[cls]
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
                all_support_labels.append(LABEL_MAP[cls])

        if not all_support_features:
            print("❌ 无法计算原型，评估失败")
            return 0.0

        support_features = torch.cat(all_support_features, dim=0)
        support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
        prototypes = model.compute_prototypes(support_features, support_labels)

        # 2. 评估独立测试集
        print(f"📊 在独立测试集评估模型...")
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)

            # 测试集数据（完全隔离）
            batch_flow = torch.tensor(test_data["features_flow"][i:end_idx]).to(DEVICE)
            batch_seq = torch.tensor(test_data["features_seq"][i:end_idx]).to(DEVICE)

            # 模型预测
            batch_features = model(batch_flow, batch_seq)
            batch_preds, _ = model.predict(batch_features, prototypes)

            all_preds.extend(batch_preds.cpu().numpy())

    # 计算评估指标
    all_preds = np.array(all_preds)
    accuracy = np.mean(all_preds == all_labels)

    print(f"\n🎯 独立测试集评估结果:")
    print(f"   测试样本数: {num_samples}")
    print(f"   整体准确率: {accuracy:.4f}")
    print("\n📋 详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))

    return accuracy


# =====================================================
# 模型架构（移除域对抗相关逻辑，简化模型）
# =====================================================
class FeatureEncoder(nn.Module):
    """简化版特征提取器（无域对抗）"""

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
    """原型网络模型（简化版）"""

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
# 任务采样器（仅使用训练集，无偏移域）
# =====================================================
class BaseTaskSampler:
    """基础采样器（仅从训练集采样）"""

    def __init__(self, train_data):
        self.train_data = train_data  # 仅使用隔离后的训练集
        self.indices = {c: list(range(len(train_data[c]))) for c in CLASSES}

    def sample_task(self):
        n_way = min(5, len(CLASSES))
        selected_classes = random.sample(CLASSES, n_way)

        sx_f, sx_s, sy = [], [], []
        qx_f, qx_s, qy = [], [], []

        for cls in selected_classes:
            # 支持集采样
            train_samples = self.train_data[cls]
            if len(train_samples) >= K_SHOT + Q_QUERY:
                # 随机采样，避免重复
                sample_indices = random.sample(self.indices[cls], K_SHOT + Q_QUERY)
                support_indices = sample_indices[:K_SHOT]
                query_indices = sample_indices[K_SHOT:]

                # 支持集
                for idx in support_indices:
                    f, s = train_samples[idx]
                    sx_f.append(f)
                    sx_s.append(s)
                    sy.append(LABEL_MAP[cls])

                # 查询集
                for idx in query_indices:
                    f, s = train_samples[idx]
                    qx_f.append(f)
                    qx_s.append(s)
                    qy.append(LABEL_MAP[cls])

        # 转换为张量（空值保护）
        if not sx_f or not qx_f:
            return self.sample_task()

        sx_f = torch.tensor(np.array(sx_f, dtype=np.float32)).to(DEVICE)
        sx_s = torch.tensor(np.array(sx_s, dtype=np.float32)).to(DEVICE)
        sy = torch.tensor(sy, dtype=torch.long).to(DEVICE)
        qx_f = torch.tensor(np.array(qx_f, dtype=np.float32)).to(DEVICE)
        qx_s = torch.tensor(np.array(qx_s, dtype=np.float32)).to(DEVICE)
        qy = torch.tensor(qy, dtype=torch.long).to(DEVICE)

        return sx_f, sx_s, sy, qx_f, qx_s, qy


class EnhancedTaskSampler(BaseTaskSampler):
    """增强采样器（包含GAN数据）"""

    def __init__(self, train_data, gan_data):
        super().__init__(train_data)
        self.gan_data = gan_data  # GAN增强数据（仅web类别）
        self.gan_indices = list(range(len(gan_data))) if gan_data else []

    def sample_task(self):
        n_way = min(5, len(CLASSES))
        selected_classes = random.sample(CLASSES, n_way)

        sx_f, sx_s, sy = [], [], []
        qx_f, qx_s, qy = [], [], []

        for cls in selected_classes:
            # web类别使用GAN增强数据
            if cls == "web" and self.gan_data and len(self.gan_data) > 0:
                train_samples = self.train_data[cls]
                # 支持集：一半真实数据 + 一半GAN数据
                gan_to_use = min(K_SHOT // 2, len(self.gan_data))
                real_to_use = K_SHOT - gan_to_use

                if len(train_samples) >= real_to_use and gan_to_use > 0:
                    # 真实数据
                    real_indices = random.sample(self.indices[cls], real_to_use)
                    for idx in real_indices:
                        f, s = train_samples[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])

                    # GAN数据
                    gan_indices = random.sample(self.gan_indices, gan_to_use)
                    for idx in gan_indices:
                        f, s = self.gan_data[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])
                else:
                    # 不足时使用纯真实数据
                    if len(train_samples) >= K_SHOT:
                        support_indices = random.sample(self.indices[cls], K_SHOT)
                        for idx in support_indices:
                            f, s = train_samples[idx]
                            sx_f.append(f)
                            sx_s.append(s)
                            sy.append(LABEL_MAP[cls])
            else:
                # 其他类别使用纯真实数据
                train_samples = self.train_data[cls]
                if len(train_samples) >= K_SHOT:
                    support_indices = random.sample(self.indices[cls], K_SHOT)
                    for idx in support_indices:
                        f, s = train_samples[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])

            # 查询集：仅使用真实训练数据
            train_samples = self.train_data[cls]
            if len(train_samples) >= Q_QUERY:
                query_indices = random.sample(self.indices[cls], Q_QUERY)
                for idx in query_indices:
                    f, s = train_samples[idx]
                    qx_f.append(f)
                    qx_s.append(s)
                    qy.append(LABEL_MAP[cls])

        # 转换为张量（空值保护）
        if not sx_f or not qx_f:
            return self.sample_task()

        sx_f = torch.tensor(np.array(sx_f, dtype=np.float32)).to(DEVICE)
        sx_s = torch.tensor(np.array(sx_s, dtype=np.float32)).to(DEVICE)
        sy = torch.tensor(sy, dtype=torch.long).to(DEVICE)
        qx_f = torch.tensor(np.array(qx_f, dtype=np.float32)).to(DEVICE)
        qx_s = torch.tensor(np.array(qx_s, dtype=np.float32)).to(DEVICE)
        qy = torch.tensor(qy, dtype=torch.long).to(DEVICE)

        return sx_f, sx_s, sy, qx_f, qx_s, qy


# =====================================================
# 训练器（简化版，无域对抗）
# =====================================================
class PrototypicalTrainer:
    """原型网络训练器（简化版）"""

    def __init__(self, model, is_enhanced=False):
        self.model = model
        self.is_enhanced = is_enhanced
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=META_LR * (0.5 if is_enhanced else 1.0),
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=META_EPOCHS
        )

    def train_episode(self, sampler):
        self.model.train()
        # 仅从训练集采样任务
        if self.is_enhanced:
            sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()
        else:
            sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()

        # 提取特征
        support_features = self.model(sx_f, sx_s)
        query_features = self.model(qx_f, qx_s)
        prototypes = self.model.compute_prototypes(support_features, sy)
        predictions, distances = self.model.predict(query_features, prototypes)

        # 计算损失
        all_distances = []
        for query_feature in query_features:
            query_distances = []
            for label in prototypes.keys():
                distance = torch.norm(query_feature - prototypes[label], p=2)
                query_distances.append(-distance)
            all_distances.append(torch.stack(query_distances))
        logits = torch.stack(all_distances)
        loss = self.loss_fn(logits, qy)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 计算准确率
        accuracy = (predictions == qy).float().mean().item()
        return loss.item(), accuracy

    def train(self, sampler, target_acc=None):
        if self.is_enhanced:
            print("\n=== 增强模型训练开始（使用GAN数据） ===")
        else:
            print(f"\n=== 基础模型训练开始 ===")
            if target_acc:
                print(f"   目标准确率: {target_acc}")

        best_val_acc = 0.0
        target_reached = False
        for epoch in range(META_EPOCHS):
            if target_acc and target_reached:
                print(f"   在第{epoch}轮达到目标准确率，停止训练")
                break

            epoch_loss = 0.0
            epoch_acc = 0.0
            num_episodes = 0

            for _ in range(TASKS_PER_BATCH):
                try:
                    loss, acc = self.train_episode(sampler)
                    epoch_loss += loss
                    epoch_acc += acc
                    num_episodes += 1
                except Exception as e:
                    continue

            if num_episodes > 0:
                avg_loss = epoch_loss / num_episodes
                avg_acc = epoch_acc / num_episodes
                self.scheduler.step()
                mode = "增强" if self.is_enhanced else "基础"
                print(f"   [{mode} Epoch {epoch + 1:03d}/{META_EPOCHS}] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

            # 验证（仅使用训练集内部验证）
            if (epoch + 1) % 5 == 0:
                val_acc = self.validate(sampler)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_path = 'best_enhanced_model.pth' if self.is_enhanced else 'best_baseline_model.pth'
                    torch.save(self.model.state_dict(), save_path)
                # 检查目标准确率
                if not self.is_enhanced and target_acc and val_acc >= target_acc:
                    target_reached = True
                    torch.save(self.model.state_dict(), 'target_reached_model.pth')
                    print(f"   🎉 达到目标准确率 {target_acc}!")

        if not self.is_enhanced and not target_reached and target_acc:
            print(f"   ⚠️ 未达到目标准确率 {target_acc}")
        return best_val_acc

    def validate(self, sampler, num_tasks=8):
        """验证（仅使用训练集）"""
        self.model.eval()
        accuracies = []
        with torch.no_grad():
            for _ in range(num_tasks):
                try:
                    if self.is_enhanced:
                        sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()
                    else:
                        sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()

                    support_features = self.model(sx_f, sx_s)
                    query_features = self.model(qx_f, qx_s)
                    prototypes = self.model.compute_prototypes(support_features, sy)
                    predictions, _ = self.model.predict(query_features, prototypes)

                    accuracy = (predictions == qy).float().mean().item()
                    accuracies.append(accuracy)
                except Exception as e:
                    continue

        avg_acc = np.mean(accuracies) if accuracies else 0.0
        mode = "增强" if self.is_enhanced else "基础"
        print(f"   [{mode} Validation] 训练集内部验证准确率: {avg_acc:.4f}")
        self.model.train()
        return avg_acc


# =====================================================
# 主程序（仅基于all.csv）
# =====================================================
def main():
    print("=" * 80)
    print("模型训练与评估系统（仅基于all.csv，移除network.csv）")
    print("=" * 80)

    # 数据路径（仅all.csv和GAN数据）
    STABLE_CSV = "data/all.csv"
    WEB_GAN_CSV = "data/web_gan.csv"

    # =====================================================
    # 阶段0：核心步骤 - 先拆分训练/测试数据（完全隔离）
    # =====================================================
    print("\n" + "=" * 80)
    print("阶段0：拆分训练/测试数据（仅基于all.csv）")
    print("=" * 80)

    # 1. 加载原始数据（仅all.csv）
    raw_data, gan_data = load_raw_data(STABLE_CSV, WEB_GAN_CSV, use_gan=True)

    # 2. 询问是否强制重新拆分
    force_rebuild = input("\n是否强制重新拆分训练/测试数据？(y/n，默认n): ")
    force_rebuild = True if force_rebuild.lower() == 'y' else False

    # 3. 拆分数据（测试集完全不参与训练）
    train_data, test_data = split_train_test_data(
        raw_data,
        test_ratio=TEST_SPLIT_RATIO,
        test_size=TEST_SAMPLE_SIZE,
        force_rebuild=force_rebuild
    )

    # =====================================================
    # 阶段1：训练基础模型（仅使用训练集）
    # =====================================================
    print("\n" + "=" * 80)
    print("阶段1：训练基础模型（仅使用all.csv训练集）")
    print("=" * 80)

    # 1. 创建基础模型
    print("\n📌 创建基础模型...")
    baseline_model = PrototypicalNetwork().to(DEVICE)

    # 2. 创建训练集采样器（仅训练集）
    base_sampler = BaseTaskSampler(train_data)

    # 3. 训练基础模型（仅训练集）
    baseline_trainer = PrototypicalTrainer(baseline_model, is_enhanced=False)
    baseline_best_val_acc = baseline_trainer.train(base_sampler, target_acc=TARGET_ACC)

    # 4. 加载最佳模型
    try:
        baseline_model.load_state_dict(torch.load('target_reached_model.pth'))
        print("\n✅ 加载达到目标准确率的基础模型")
        target_reached = True
    except:
        print("\n⚠️ 未达到目标准确率，加载训练集最佳模型")
        baseline_model.load_state_dict(torch.load('saved_models/best_baseline_model.pth'))
        target_reached = False

    # 5. 评估基础模型（仅使用独立测试集）
    print("\n📊 评估基础模型（使用完全隔离的测试集）...")
    baseline_test_acc = evaluate_model_on_isolated_test(baseline_model, test_data)

    # 保存基础模型
    torch.save(baseline_model.state_dict(), 'saved_models/baseline_model_final.pth')
    print(f"\n📝 基础模型最终结果:")
    print(f"   训练集内部最佳验证准确率: {baseline_best_val_acc:.4f}")
    print(f"   独立测试集准确率: {baseline_test_acc:.4f}")

    # =====================================================
    # 阶段2：训练/评估增强模型（使用GAN数据）
    # =====================================================
    print("\n" + "=" * 80)
    print("阶段2：训练/评估增强模型（训练集+GAN）")
    print("=" * 80)

    # 1. 创建增强模型并加载基础模型参数
    print("\n📌 创建增强模型...")
    enhanced_model = PrototypicalNetwork().to(DEVICE)
    try:
        if target_reached:
            enhanced_model.load_state_dict(torch.load('target_reached_model.pth'))
            print("   从达标基础模型加载参数")
        else:
            enhanced_model.load_state_dict(torch.load('saved_models/best_baseline_model.pth'))
            print("   从最佳基础模型加载参数")
    except:
        print("   ⚠️ 无法加载基础模型参数，使用随机初始化")

    # 2. 创建增强采样器（训练集+GAN）
    enhanced_sampler = EnhancedTaskSampler(train_data, gan_data)

    # 3. 训练增强模型
    enhanced_trainer = PrototypicalTrainer(enhanced_model, is_enhanced=True)
    enhanced_best_val_acc = enhanced_trainer.train(enhanced_sampler)

    # 4. 加载增强模型最佳参数
    enhanced_model.load_state_dict(torch.load('saved_models/best_enhanced_model.pth'))

    # 5. 评估增强模型（仅使用独立测试集）
    print("\n📊 评估增强模型（使用完全隔离的测试集）...")
    enhanced_test_acc = evaluate_model_on_isolated_test(enhanced_model, test_data)

    # 保存增强模型
    os.makedirs("saved_models", exist_ok=True)
    torch.save(enhanced_model.state_dict(), "saved_models/enhanced_model_final.pth")
    print(f"\n📝 增强模型最终结果:")
    print(f"   训练集内部最佳验证准确率: {enhanced_best_val_acc:.4f}")
    print(f"   独立测试集准确率: {enhanced_test_acc:.4f}")

    # =====================================================
    # 阶段3：最终对比总结
    # =====================================================
    print("\n" + "=" * 80)
    print("最终性能总结（仅基于all.csv）")
    print("=" * 80)

    print(f"\n📈 模型性能对比:")
    print(f"   基础模型（无GAN）独立测试集准确率: {baseline_test_acc:.4f}")
    print(f"   增强模型（有GAN）独立测试集准确率: {enhanced_test_acc:.4f}")
    print(f"   准确率提升: {enhanced_test_acc - baseline_test_acc:+.4f}")

    if target_reached:
        print(f"\n✅ 基础模型成功达到目标准确率 {TARGET_ACC}")
    else:
        print(f"\n⚠️ 基础模型未达到目标准确率 {TARGET_ACC}")
        print(f"   训练集内部最佳验证准确率: {baseline_best_val_acc:.4f}")

    print(f"\n📋 关键信息:")
    print(f"   ✅ 仅使用all.csv数据（已移除network.csv依赖）")
    print(f"   ✅ 测试集占比: {TEST_SPLIT_RATIO * 100}%")
    print(f"   ✅ 测试集样本数: {len(test_data['labels'])}")
    print(f"   ✅ 测试数据完全不参与训练/原型计算")
    print(f"   ✅ 所有模型评估使用同一批独立测试集")


if __name__ == "__main__":
    # 设置随机种子（保证拆分结果可复现）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()