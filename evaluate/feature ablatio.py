#text+value
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import matplotlib.pyplot as plt

# =====================================================
# 全局配置（根据你的实际路径调整）
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 特征维度配置
FLOW_DIM = 60  # 待消融的前60个特征
SEQ_DIM = 32
NUM_CLASSES = 5

# 少样本训练配置（与原模型保持一致）
K_SHOT = 10
Q_QUERY = 10
TASKS_PER_BATCH = 12
META_LR = 3e-4
META_EPOCHS = 5  # 消融实验训练轮数可适当减少
TARGET_ACC = 0.68

# 路径配置
MODEL_DIR = "saved_models"
TEST_DATA_DIR = "test_data"
TRAIN_DATA_DIR = "train_data"
RESULT_DIR = "ablation_results2"
WEB_GAN_CSV_PATH = "data/web_gan.csv"  # GAN数据CSV文件路径
ALL_CSV_PATH = "data/all.csv"  # 用于获取归一化器的all.csv路径

# 关键文件路径
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "enhanced_model_final.pth")  # 原始GAN增强模型
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "fixed_test_data.pkl")
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "train_data.pkl")

# 消融实验配置
ABLATION_FEATURES = list(range(FLOW_DIM))  # 0-59号flow特征
ABLATION_VALUE = 0.0  # 消融值（置零）
SAVE_PLOTS = True  # 是否保存可视化结果
VERBOSE = True  # 是否打印详细日志
CONFUSION_MATRIX_SAVE = True  # 是否保存混淆矩阵

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

# 设置随机种子（保证实验可复现）
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 创建必要目录
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# 1. 数据加载与预处理
# =====================================================
def load_gan_data_from_csv(gan_csv_path=WEB_GAN_CSV_PATH, all_csv_path=ALL_CSV_PATH):
    """直接从CSV文件加载GAN数据，并使用all.csv的归一化器进行标准化"""
    gan_data = []

    if not os.path.exists(gan_csv_path):
        print(f"⚠️ GAN数据CSV文件不存在: {gan_csv_path}，增强模型将退化为基础模型")
        return gan_data

    if not os.path.exists(all_csv_path):
        print(f"⚠️ all.csv不存在: {all_csv_path}，GAN数据将使用自身归一化")
        use_all_scaler = False
    else:
        use_all_scaler = True

    try:
        print(f"\n📥 从CSV加载GAN数据: {gan_csv_path}")

        df_gan = pd.read_csv(gan_csv_path)
        X_seq_gan = df_gan.iloc[:, :SEQ_DIM].values.astype(np.float32)

        if use_all_scaler:
            df_all = pd.read_csv(all_csv_path)
            X_seq_all = df_all.iloc[:, FLOW_DIM:FLOW_DIM + SEQ_DIM].values.astype(np.float32)
            seq_scaler = StandardScaler()
            seq_scaler.fit(X_seq_all)
            X_seq_gan = seq_scaler.transform(X_seq_gan)
        else:
            seq_scaler = StandardScaler()
            X_seq_gan = seq_scaler.fit_transform(X_seq_gan)

        zero_flow_gan = np.zeros((len(X_seq_gan), FLOW_DIM), dtype=np.float32)
        for f, s in zip(zero_flow_gan, X_seq_gan):
            gan_data.append((f, s))

        print(f"✅ GAN数据加载完成，共{len(gan_data)}个样本")

    except Exception as e:
        print(f"⚠️ 加载GAN数据失败: {str(e)}")
        gan_data = []

    return gan_data


def load_saved_data():
    """加载已保存的训练集、测试集，直接从CSV加载GAN数据"""
    print("\n📥 加载已保存的实验数据...")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"测试数据不存在: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_DATA_PATH}")
    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)

    gan_data = load_gan_data_from_csv()

    print(f"✅ 数据加载完成:")
    print(f"   测试集样本数: {len(test_data['labels'])}")
    print(f"   训练集类别分布: {[f'{cls}: {len(train_data[cls])}' for cls in CLASSES]}")
    print(f"   GAN数据样本数: {len(gan_data)}")

    return train_data, test_data, gan_data


def ablation_feature(data, feature_idx, ablation_value=ABLATION_VALUE, is_test_data=True):
    """消融指定索引的flow特征（置零）"""
    if feature_idx < 0 or feature_idx >= FLOW_DIM:
        raise ValueError(f"特征索引超出范围（0-{FLOW_DIM - 1}）: {feature_idx}")

    if is_test_data:
        ablated_data = data.copy()
        ablated_flow = ablated_data["features_flow"].copy()
        ablated_flow[:, feature_idx] = ablation_value
        ablated_data["features_flow"] = ablated_flow
    else:
        ablated_data = {cls: [] for cls in CLASSES}
        for cls in CLASSES:
            for flow, seq in data[cls]:
                ablated_flow = flow.copy()
                ablated_flow[feature_idx] = ablation_value
                ablated_data[cls].append((ablated_flow, seq))

    return ablated_data


# =====================================================
# 2. 混淆矩阵计算工具（支持不同评估场景）
# =====================================================
def compute_confusion_matrix(model, data_source, sampler=None, test_data=None, feature_idx=-1, eval_type="val"):
    """
    计算混淆矩阵（支持验证集/测试集）
    eval_type: "val" (训练集验证) / "test" (独立测试集)
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        if eval_type == "val":
            # 训练集验证任务
            if sampler is None:
                raise ValueError("eval_type=val时必须传入sampler")
            for _ in range(16):  # 16个验证任务
                try:
                    sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()
                    support_features = model(sx_f, sx_s)
                    prototypes = model.compute_prototypes(support_features, sy)
                    query_features = model(qx_f, qx_s)
                    predictions, _ = model.predict(query_features, prototypes)
                    all_true.extend(qy.cpu().numpy())
                    all_pred.extend(predictions.cpu().numpy())
                except Exception as e:
                    continue
        elif eval_type == "test":
            # 独立测试集
            if test_data is None:
                raise ValueError("eval_type=test时必须传入test_data")
            # 计算原型
            with open(TRAIN_DATA_PATH, 'rb') as f:
                train_data = pickle.load(f)
            all_support_features = []
            all_support_labels = []
            for cls in CLASSES:
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
            support_features = torch.cat(all_support_features, dim=0)
            support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
            prototypes = model.compute_prototypes(support_features, support_labels)

            # 测试集预测
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

    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 保存混淆矩阵
    if CONFUSION_MATRIX_SAVE:
        # 保存数值矩阵
        cm_df = pd.DataFrame(cm_normalized,
                             index=[f'真实_{cls}' for cls in CLASSES],
                             columns=[f'预测_{cls}' for cls in CLASSES])
        cm_filename = f"confusion_matrix_{feature_idx}_{eval_type}.csv"
        cm_path = os.path.join(RESULT_DIR, cm_filename)
        cm_df.to_csv(cm_path, encoding='utf-8')

        # 绘制并保存混淆矩阵图
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(8, 6))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        eval_name = "验证集" if eval_type == "val" else "测试集"
        plt.title(f'混淆矩阵（特征{feature_idx} - {eval_name}）')
        plt.colorbar()
        tick_marks = np.arange(len(CLASSES))
        plt.xticks(tick_marks, CLASSES, rotation=45)
        plt.yticks(tick_marks, CLASSES)

        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")

        plt.ylabel('真实类别')
        plt.xlabel('预测类别')
        plt.tight_layout()
        fig_filename = f"confusion_matrix_{feature_idx}_{eval_type}.png"
        plt.savefig(os.path.join(RESULT_DIR, fig_filename), dpi=300)
        plt.close()

    model.train()
    return cm_normalized, cm, accuracy_score(all_true, all_pred)


# =====================================================
# 3. 模型定义
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
# 4. 训练器与采样器
# =====================================================
class EnhancedTaskSampler:
    """增强模型采样器"""

    def __init__(self, train_data, gan_data):
        self.train_data = train_data
        self.gan_data = gan_data
        self.indices = {c: list(range(len(train_data[c]))) for c in CLASSES}
        self.gan_indices = list(range(len(gan_data))) if gan_data else []

    def sample_task(self):
        n_way = min(5, len(CLASSES))
        selected_classes = random.sample(CLASSES, n_way)

        sx_f, sx_s, sy = [], [], []
        qx_f, qx_s, qy = [], [], []

        for cls in selected_classes:
            if cls == "web" and self.gan_data and len(self.gan_data) > 0:
                train_samples = self.train_data[cls]
                gan_to_use = min(K_SHOT // 2, len(self.gan_data))
                real_to_use = K_SHOT - gan_to_use

                if len(train_samples) >= real_to_use and gan_to_use > 0:
                    real_indices = random.sample(self.indices[cls], real_to_use)
                    for idx in real_indices:
                        f, s = train_samples[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])

                    gan_indices = random.sample(self.gan_indices, gan_to_use)
                    for idx in gan_indices:
                        f, s = self.gan_data[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])
                else:
                    if len(train_samples) >= K_SHOT:
                        support_indices = random.sample(self.indices[cls], K_SHOT)
                        for idx in support_indices:
                            f, s = train_samples[idx]
                            sx_f.append(f)
                            sx_s.append(s)
                            sy.append(LABEL_MAP[cls])
            else:
                train_samples = self.train_data[cls]
                if len(train_samples) >= K_SHOT:
                    support_indices = random.sample(self.indices[cls], K_SHOT)
                    for idx in support_indices:
                        f, s = train_samples[idx]
                        sx_f.append(f)
                        sx_s.append(s)
                        sy.append(LABEL_MAP[cls])

            train_samples = self.train_data[cls]
            if len(train_samples) >= Q_QUERY:
                query_indices = random.sample(self.indices[cls], Q_QUERY)
                for idx in query_indices:
                    f, s = train_samples[idx]
                    qx_f.append(f)
                    qx_s.append(s)
                    qy.append(LABEL_MAP[cls])

        if not sx_f or not qx_f:
            return self.sample_task()

        sx_f = torch.tensor(np.array(sx_f, dtype=np.float32)).to(DEVICE)
        sx_s = torch.tensor(np.array(sx_s, dtype=np.float32)).to(DEVICE)
        sy = torch.tensor(sy, dtype=torch.long).to(DEVICE)
        qx_f = torch.tensor(np.array(qx_f, dtype=np.float32)).to(DEVICE)
        qx_s = torch.tensor(np.array(qx_s, dtype=np.float32)).to(DEVICE)
        qy = torch.tensor(qy, dtype=torch.long).to(DEVICE)

        return sx_f, sx_s, sy, qx_f, qx_s, qy


class AblationTrainer:
    """消融实验训练器（记录最佳验证指标）"""

    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=META_LR,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=META_EPOCHS
        )
        self.best_val_acc = 0.0
        self.best_val_epoch = 0

    def train_episode(self, sampler):
        self.model.train()
        sx_f, sx_s, sy, qx_f, qx_s, qy = sampler.sample_task()

        support_features = self.model(sx_f, sx_s)
        query_features = self.model(qx_f, qx_s)
        prototypes = self.model.compute_prototypes(support_features, sy)
        predictions, distances = self.model.predict(query_features, prototypes)

        all_distances = []
        for query_feature in query_features:
            query_distances = []
            for label in prototypes.keys():
                distance = torch.norm(query_feature - prototypes[label], p=2)
                query_distances.append(-distance)
            all_distances.append(torch.stack(query_distances))
        logits = torch.stack(all_distances)
        loss = self.loss_fn(logits, qy)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        accuracy = (predictions == qy).float().mean().item()
        return loss.item(), accuracy

    def train(self, sampler, feature_idx):
        """训练消融模型，返回最佳验证准确率"""
        print(f"\n🚀 开始训练消融特征{feature_idx}后的模型...")

        for epoch in range(META_EPOCHS):
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
                    if VERBOSE:
                        print(f"⚠️ 训练episode失败: {e}")
                    continue

            if num_episodes > 0:
                avg_loss = epoch_loss / num_episodes
                avg_acc = epoch_acc / num_episodes
                self.scheduler.step()

                if VERBOSE and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch + 1:03d}/{META_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

            if (epoch + 1) % 5 == 0:
                val_acc = self.validate(sampler)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_epoch = epoch + 1
                    # 保存最佳验证模型
                    model_path = os.path.join(RESULT_DIR, f"ablated_model_{feature_idx}.pth")
                    torch.save(self.model.state_dict(), model_path)

        print(
            f"✅ 消融特征{feature_idx}模型训练完成 | 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_val_epoch})")
        return self.best_val_acc

    def validate(self, sampler, num_tasks=8):
        """验证函数（训练集内部）"""
        self.model.eval()
        accuracies = []
        with torch.no_grad():
            for _ in range(num_tasks):
                try:
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
        if VERBOSE:
            print(f"   Validation Acc: {avg_acc:.4f}")
        self.model.train()
        return avg_acc


# =====================================================
# 5. 双评估核心函数（
# =====================================================
def evaluate_ablation_model(model, feature_idx, ablated_train_data, ablated_test_data, gan_data):
   
    # 1. 加载最佳验证模型
    model_path = os.path.join(RESULT_DIR, f"ablated_model_{feature_idx}.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 2. 评估1：最佳验证准确率（重新计算，确保准确）
    val_sampler = EnhancedTaskSampler(ablated_train_data, gan_data)
    val_cm, _, val_acc = compute_confusion_matrix(
        model, None, sampler=val_sampler, feature_idx=feature_idx, eval_type="val"
    )

    # 3. 评估2：测试集最终准确率
    test_cm, _, test_acc = compute_confusion_matrix(
        model, None, test_data=ablated_test_data, feature_idx=feature_idx, eval_type="test"
    )

    # 4. 取较低值，确定最终指标
    if val_acc < test_acc:
        final_acc = val_acc
        final_cm = val_cm
        final_type = "val"
        #print(f"⚠️ 特征{feature_idx}：验证准确率({val_acc:.4f}) < 测试准确率({test_acc:.4f})，最终取验证集指标")
    else:
        final_acc = test_acc
        final_cm = test_cm
        final_type = "test"
        #print(f"⚠️ 特征{feature_idx}：测试准确率({test_acc:.4f}) ≤ 验证准确率({val_acc:.4f})，最终取测试集指标")

    # 保存最终混淆矩阵
    if CONFUSION_MATRIX_SAVE:
        cm_df = pd.DataFrame(final_cm,
                             index=[f'真实_{cls}' for cls in CLASSES],
                             columns=[f'预测_{cls}' for cls in CLASSES])
        cm_path = os.path.join(RESULT_DIR, f"confusion_matrix_{feature_idx}_final.csv")
        cm_df.to_csv(cm_path, encoding='utf-8')

        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(8, 6))
        plt.imshow(final_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'混淆矩阵（特征{feature_idx} - 最终指标/{final_type}）')
        plt.colorbar()
        tick_marks = np.arange(len(CLASSES))
        plt.xticks(tick_marks, CLASSES, rotation=45)
        plt.yticks(tick_marks, CLASSES)

        thresh = final_cm.max() / 2.
        for i in range(final_cm.shape[0]):
            for j in range(final_cm.shape[1]):
                plt.text(j, i, f'{final_cm[i, j]:.2f}',
                         horizontalalignment="center",
                         color="white" if final_cm[i, j] > thresh else "black")

        plt.ylabel('真实类别')
        plt.xlabel('预测类别')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"confusion_matrix_{feature_idx}_final.png"), dpi=300)
        plt.close()

    # 返回双评估结果和最终结果
    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "final_acc": final_acc,
        "final_type": final_type,
        "final_cm": final_cm
    }


def evaluate_baseline_model(model, test_data):
    """评估基准模型（仅测试集）"""
    print(f"\n📊 评估基准模型性能（测试集）...")

    # 计算测试集混淆矩阵和准确率
    test_cm, _, test_acc = compute_confusion_matrix(
        model, None, test_data=test_data, feature_idx=-1, eval_type="test"
    )

    # 计算类别准确率
    all_true = test_data["labels"]
    all_pred = []
    batch_size = 64
    num_samples = len(all_true)

    with torch.no_grad():
        # 计算原型
        with open(TRAIN_DATA_PATH, 'rb') as f:
            train_data = pickle.load(f)
        all_support_features = []
        all_support_labels = []
        for cls in CLASSES:
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

        support_features = torch.cat(all_support_features, dim=0)
        support_labels = torch.tensor(all_support_labels, dtype=torch.long).to(DEVICE)
        prototypes = model.compute_prototypes(support_features, support_labels)

        # 测试集预测
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_flow = torch.tensor(test_data["features_flow"][i:end_idx]).to(DEVICE)
            batch_seq = torch.tensor(test_data["features_seq"][i:end_idx]).to(DEVICE)
            batch_features = model(batch_flow, batch_seq)
            batch_preds, _ = model.predict(batch_features, prototypes)
            all_pred.extend(batch_preds.cpu().numpy())

    class_acc = {}
    for cls in CLASSES:
        cls_label = LABEL_MAP[cls]
        mask = np.array(all_true) == cls_label
        if np.sum(mask) > 0:
            class_acc[cls] = accuracy_score(np.array(all_true)[mask], np.array(all_pred)[mask])
        else:
            class_acc[cls] = 0.0

    print(f"   基准模型测试集准确率: {test_acc:.4f}")
    print(f"   类别准确率:")
    for cls in CLASSES:
        print(f"     {cls}: {class_acc[cls]:.4f}")

    return test_acc, class_acc, test_cm


# =====================================================
# 6. 主实验流程
# =====================================================
def main():

    print("=" * 80)

    # Step 1: 加载数据
    train_data, test_data, gan_data = load_saved_data()

    # Step 2: 基准模型评估（仅测试集）
    print("\n" + "=" * 80)
    print("Step 1: 基准模型性能测试（独立测试集）")
    print("=" * 80)

    baseline_model = PrototypicalNetwork().to(DEVICE)
    if os.path.exists(BASE_MODEL_PATH):
        baseline_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
        print(f"✅ 加载预训练GAN增强模型: {BASE_MODEL_PATH}")
    else:
        print(f"⚠️ 预训练模型不存在，使用随机初始化模型作为基准")

    baseline_acc, baseline_class_acc, baseline_cm = evaluate_baseline_model(baseline_model, test_data)

    # 保存基准结果
    baseline_results = {
        "test_acc": baseline_acc,
        "class_acc": baseline_class_acc,
        "cm": baseline_cm,
        "feature_idx": -1
    }
    with open(os.path.join(RESULT_DIR, "baseline_results.pkl"), 'wb') as f:
        pickle.dump(baseline_results, f)

    # Step 3: 消融实验（双评估+取低值）
    print("\n" + "=" * 80)
    print("Step 2: 消融模型评估实验")
    print("=" * 80)

    ablation_results = {
        "feature_idx": [],
        "val_acc": [],  # 验证集准确率
        "test_acc": [],  # 测试集准确率
        "final_acc": [],  # 最终取的低值
        "final_type": [],  # 最终指标类型（val/test）
        "acc_drop": []  # 相对于基准的下降幅度
    }

    for feat_idx in ABLATION_FEATURES:
        print(f"\n🔬 开始消融特征{feat_idx}实验...")

        # a. 消融训练集和测试集
        ablated_train_data = ablation_feature(train_data, feat_idx, is_test_data=False)
        ablated_test_data = ablation_feature(test_data, feat_idx, is_test_data=True)

        # b. 初始化并训练消融模型
        ablated_model = PrototypicalNetwork().to(DEVICE)
        ablated_sampler = EnhancedTaskSampler(ablated_train_data, gan_data)
        ablation_trainer = AblationTrainer(ablated_model)
        best_val_acc = ablation_trainer.train(ablated_sampler, feat_idx)

        # c. 双评估+取低值
        eval_results = evaluate_ablation_model(
            ablated_model, feat_idx, ablated_train_data, ablated_test_data, gan_data
        )

        # d. 计算相对于基准的下降幅度
        acc_drop = baseline_acc - eval_results["final_acc"]

        # e. 保存结果
        ablation_results["feature_idx"].append(feat_idx)
        ablation_results["val_acc"].append(eval_results["val_acc"])
        ablation_results["test_acc"].append(eval_results["test_acc"])
        ablation_results["final_acc"].append(eval_results["final_acc"])
        ablation_results["final_type"].append(eval_results["final_type"])
        ablation_results["acc_drop"].append(acc_drop)

        # 实时保存
        with open(os.path.join(RESULT_DIR, "ablation_results.pkl"), 'wb') as f:
            pickle.dump(ablation_results, f)

        print(f"📝 消融特征{feat_idx}最终结果:")
        print(
            f"   最终取: {eval_results['final_type']} ({eval_results['final_acc']:.4f}) | 相对于基准下降: {acc_drop:.4f}")

    # Step 4: 结果分析与可视化
    print("\n" + "=" * 80)
    print("Step 3: 实验结果分析与可视化")
    print("=" * 80)

    # 转换为DataFrame
    results_df = pd.DataFrame(ablation_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "ablation_results.csv"), index=False, encoding='utf-8')

    # 统计分析
    print(f"\n📈 消融实验统计（最终指标）:")
    print(f"   平均最终准确率: {np.mean(ablation_results['final_acc']):.4f}")
    print(f"   平均相对于基准下降: {np.mean(ablation_results['acc_drop']):.4f}")

    # 影响最大/最小的特征
    top_10_impactful = results_df.nlargest(10, "acc_drop")
    top_10_least = results_df.nsmallest(10, "acc_drop")

    print(f"\n📊 影响最大的10个特征（下降最多）:")
    print(top_10_impactful[["feature_idx", "final_acc", "acc_drop"]])

    print(f"\n📊 影响最小的10个特征（下降最少）:")
    print(top_10_least[["feature_idx", "final_acc", "acc_drop"]])

    # 可视化
    if SAVE_PLOTS:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 图1：验证集vs测试集vs最终准确率
        plt.figure(figsize=(15, 7))
        plt.plot(ablation_results["feature_idx"], ablation_results["val_acc"], 'o-', label='验证集准确率', color='blue',
                 alpha=0.7)
        plt.plot(ablation_results["feature_idx"], ablation_results["test_acc"], 's-', label='测试集准确率',
                 color='green', alpha=0.7)
        plt.plot(ablation_results["feature_idx"], ablation_results["final_acc"], '^-', label='最终取低值', color='red',
                 linewidth=2)
        plt.axhline(y=baseline_acc, color='orange', linestyle='--', label=f'基准测试集准确率: {baseline_acc:.4f}')
        plt.xlabel("Flow特征索引（0-59）")
        plt.ylabel("准确率")
        plt.title("消融模型验证集/测试集/最终准确率对比")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(RESULT_DIR, "ablation_acc_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2：最终准确率下降幅度
        plt.figure(figsize=(15, 6))
        plt.bar(ablation_results["feature_idx"], ablation_results["acc_drop"], color='orangered')
        plt.axhline(y=np.mean(ablation_results["acc_drop"]), color='blue', linestyle='--',
                    label=f'平均下降: {np.mean(ablation_results["acc_drop"]):.4f}')
        plt.xlabel("Flow特征索引（0-59）")
        plt.ylabel("相对于基准的准确率下降幅度")
        plt.title("消融模型最终准确率下降幅度")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(RESULT_DIR, "ablation_acc_drop.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Step 5: 最终总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    print(f"\n📊 基准性能:")
    print(f"   基准模型测试集准确率: {baseline_acc:.4f}")

    print(f"\n🔍 消融实验最终统计:")
    print(f"   消融模型平均最终准确率: {np.mean(ablation_results['final_acc']):.4f}")
    print(f"   平均下降幅度: {np.mean(ablation_results['acc_drop']):.4f}")
    print(
        f"   最大下降幅度: {np.max(ablation_results['acc_drop']):.4f} (特征{results_df['feature_idx'].iloc[results_df['acc_drop'].idxmax()]})")
    print(
        f"   最小下降幅度: {np.min(ablation_results['acc_drop']):.4f} (特征{results_df['feature_idx'].iloc[results_df['acc_drop'].idxmin()]})")

    print(f"\n💡 关键结论:")
    print(f"   1. 特征{results_df['feature_idx'].iloc[results_df['acc_drop'].idxmax()]}是对模型性能影响最大的Flow特征")
    print(f"   2. 特征{results_df['feature_idx'].iloc[results_df['acc_drop'].idxmin()]}是对模型性能影响最小的Flow特征")

    print(f"\n📁 实验结果已保存至: {RESULT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()