# 模拟网络波动对数据包大小序列的影响，结果存入output文件夹，后续需要进一步修改，用于AE降维
import csv
import random
import re
import os
from typing import List, Tuple
import pandas as pd


def read_packet_sequences_as_columns(csv_path: str, max_rows: int = 2000, num_columns: int = 50) -> List[List[int]]:
    """
    读取CSV文件中的包大小序列，每行转换为50列的数据

    参数:
        csv_path: CSV文件路径
        max_rows: 最大读取行数（流数量）
        num_columns: 列数（固定为50）

    返回:
        包大小序列列表，每个元素是长度为50的包大小列表
    """
    sequences = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            row_count = 0
            for row in reader:
                if row_count >= max_rows:
                    break

                if not row:  # 跳过空行
                    continue

                # 处理每一行的数据（可能用制表符或逗号分隔）
                if len(row) == 1:
                    # 可能是制表符分隔的数据，需要分割
                    line_content = row[0].strip()
                    # 使用正则表达式分割数字（支持制表符、空格、逗号分隔）
                    packet_strs = re.split(r'[\t, ]+', line_content)
                else:
                    # 已经是分割好的列表
                    packet_strs = row

                # 转换为整数列表，过滤空字符串
                packet_sequence = []
                for s in packet_strs:
                    s = s.strip()
                    if s:  # 只跳过空字符串，保留0（因为0是填充值）
                        try:
                            packet_sequence.append(int(s))
                        except ValueError:
                            # 如果转换失败，尝试处理小数
                            try:
                                packet_sequence.append(int(float(s)))
                            except:
                                print(f"警告: 跳过无效的值: '{s}'")
                                packet_sequence.append(0)  # 用0填充无效值

                # 固定长度为num_columns（50）
                if len(packet_sequence) < num_columns:
                    # 不足50个，用0填充
                    packet_sequence.extend([0] * (num_columns - len(packet_sequence)))
                elif len(packet_sequence) > num_columns:
                    # 超过50个，截断
                    packet_sequence = packet_sequence[:num_columns]

                sequences.append(packet_sequence)
                row_count += 1

                # 显示进度
                if row_count % 100 == 0:
                    print(f"  已读取 {row_count} 个流...")

        print(f"✅ 成功读取 {len(sequences)} 个流的包序列")
        print(f"   每个流固定长度: {num_columns}")

        # 验证长度
        lengths = [len(seq) for seq in sequences]
        if all(l == num_columns for l in lengths):
            print(f"   所有流长度均为 {num_columns}")
        else:
            print(f"⚠️  流长度不一致: 最小{min(lengths)}, 最大{max(lengths)}")

        if len(sequences) < max_rows:
            print(f"⚠️  文件只有 {len(sequences)} 个流，少于请求的 {max_rows} 个")
        return sequences

    except FileNotFoundError:
        print(f"❌ 文件不存在: {csv_path}")
        return []
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return []


def write_packet_sequences_as_columns(sequences: List[List[int]], out_path: str):
    """
    将包序列列表写入CSV文件，每行50列（无表头）

    参数:
        sequences: 包序列列表
        out_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        # 使用pandas创建DataFrame，不设置列名
        df = pd.DataFrame(sequences)

        # 保存为CSV，不使用索引，不写入表头
        df.to_csv(out_path, index=False, header=False, encoding='utf-8')

        print(f"✅ 成功写入 {len(sequences)} 个流到 {out_path}")
        print(f"   输出格式: 每行50列，无表头")
        print(f"   总行数: {len(sequences)}, 总列数: 50")

        # 显示前几行作为示例
        print(f"\n📋 输出文件前3行示例:")
        print(df.head(3).to_string(index=False, header=False))

    except Exception as e:
        print(f"❌ 写入文件失败: {e}")


def write_packet_sequences_as_columns_csv_module(sequences: List[List[int]], out_path: str):
    """
    使用csv模块将包序列列表写入CSV文件，每行50列（无表头）

    参数:
        sequences: 包序列列表
        out_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 直接写入数据，不写入列名（去掉表头）
            for seq in sequences:
                # 确保每个序列正好50个值
                if len(seq) != 50:
                    print(f"⚠️  警告: 流长度不是50，而是{len(seq)}，已自动调整")
                    if len(seq) < 50:
                        seq.extend([0] * (50 - len(seq)))
                    else:
                        seq = seq[:50]
                writer.writerow(seq)

        print(f"✅ 成功写入 {len(sequences)} 个流到 {out_path}")
        print(f"   输出格式: 每行50列，无表头")
        print(f"   总行数: {len(sequences)}, 总列数: 50")

        # 显示文件内容示例
        print(f"\n📋 输出文件前3行示例:")
        if sequences:
            print(f"第1行: {sequences[0][:5]}...")
            if len(sequences) > 1:
                print(f"第2行: {sequences[1][:5]}...")
            if len(sequences) > 2:
                print(f"第3行: {sequences[2][:5]}...")

    except Exception as e:
        print(f"❌ 写入文件失败: {e}")


def jitter_size(size: int, ratio: float = 0.1, min_size: int = 40) -> int:
    """
    对包大小引入随机抖动

    参数:
        size: 原始包大小（正负表示方向）
        ratio: 抖动比例（0-1）
        min_size: 最小包大小

    返回:
        抖动后的包大小
    """
    if size == 0:
        return 0

    direction = 1 if size > 0 else -1
    base_size = abs(size)

    # 计算抖动范围
    max_jitter = int(base_size * ratio)

    # 应用抖动
    jittered_size = base_size + random.randint(-max_jitter, max_jitter)

    # 确保不小于最小包大小
    jittered_size = max(min_size, jittered_size)

    # 确保不超过最大包大小（MTU限制）
    jittered_size = min(jittered_size, 1500)  # 以太网MTU

    return direction * jittered_size


def analyze_packet_sequence(sequence: List[int]) -> dict:
    """
    分析包序列的统计信息

    参数:
        sequence: 包大小列表

    返回:
        统计信息字典
    """
    if not sequence:
        return {}

    # 过滤掉0（填充值）
    valid_packets = [p for p in sequence if p != 0]

    if not valid_packets:
        return {
            'total_packets': len(sequence),
            'valid_packets': 0,
            'valid_ratio': 0.0,
            'forward_packets': 0,
            'backward_packets': 0,
            'forward_ratio': 0.0,
            'avg_size': 0.0,
            'min_size': 0,
            'max_size': 0
        }

    # 计算基本统计
    forward_packets = [p for p in valid_packets if p > 0]
    backward_packets = [p for p in valid_packets if p < 0]

    stats = {
        'total_packets': len(sequence),
        'valid_packets': len(valid_packets),
        'valid_ratio': len(valid_packets) / len(sequence),
        'forward_packets': len(forward_packets),
        'backward_packets': len(backward_packets),
        'forward_ratio': len(forward_packets) / len(valid_packets) if valid_packets else 0,
        'avg_size': sum(abs(p) for p in valid_packets) / len(valid_packets),
        'min_size': min(abs(p) for p in valid_packets),
        'max_size': max(abs(p) for p in valid_packets),
    }

    return stats


def simulate_quic_network_fluctuation_fixed_length(
        sequence: List[int],
        loss_rate: float = 0.05,
        retransmit_prob: float = 0.6,
        size_jitter_ratio: float = 0.15,
        mtu_drop_prob: float = 0.1,
        small_pkt_inject_prob: float = 0.1,
        mtu_limit: int = 1200,
        target_length: int = 50
) -> List[int]:
    """
    为单个包序列模拟QUIC网络波动，保持固定长度target_length

    参数:
        sequence: 原始包大小列表（长度应为target_length）
        target_length: 输出序列的目标长度（固定为50）

    返回:
        波动后的包大小列表（长度固定为target_length）
    """
    if not sequence:
        return [0] * target_length

    # 过滤掉0（填充值），只处理有效的包
    valid_sequence = [p for p in sequence if p != 0]

    if not valid_sequence:
        # 如果没有有效包，返回全0序列
        return [0] * target_length

    simulated_sequence = []
    pending_retrans = []

    for pkt in valid_sequence:
        direction = 1 if pkt > 0 else -1
        size = abs(pkt)

        # ---------- 1. 模拟丢包 ----------
        if random.random() < loss_rate:
            if random.random() < retransmit_prob:
                pending_retrans.append(pkt)
            continue

        # ---------- 2. 模拟 MTU 波动 ----------
        if size > mtu_limit and random.random() < mtu_drop_prob:
            size = random.randint(mtu_limit - 200, mtu_limit)

        # ---------- 3. 包大小抖动 ----------
        pkt = direction * size
        pkt = jitter_size(pkt, size_jitter_ratio)

        simulated_sequence.append(pkt)

        # ---------- 4. 插入 ACK / 控制小包 ----------
        if random.random() < small_pkt_inject_prob:
            small_size = random.randint(60, 300)
            simulated_sequence.append(-direction * small_size)

        # ---------- 5. 延迟插入"帧级重传" ----------
        if pending_retrans and random.random() < 0.3:
            rpkt = pending_retrans.pop(0)
            rpkt = jitter_size(rpkt, 0.08)
            simulated_sequence.append(rpkt)

    # 处理剩余重传
    for rpkt in pending_retrans:
        simulated_sequence.append(jitter_size(rpkt, 0.1))

    # ---------- 6. 调整到固定长度 ----------
    current_length = len(simulated_sequence)

    if current_length < target_length:
        # 如果模拟后的包少于50个，用0填充
        simulated_sequence.extend([0] * (target_length - current_length))
    elif current_length > target_length:
        # 如果模拟后的包多于50个，截断
        simulated_sequence = simulated_sequence[:target_length]

    return simulated_sequence


def simulate_all_sequences_fixed_length(
        sequences: List[List[int]],
        target_length: int = 50,
        **kwargs
) -> Tuple[List[List[int]], dict]:
    """
    为所有包序列模拟QUIC网络波动，保持固定长度

    参数:
        sequences: 原始包序列列表（每个长度应为target_length）
        target_length: 固定长度（默认50）
        **kwargs: 其他模拟参数

    返回:
        (波动后的序列列表, 总体统计信息)
    """
    simulated_sequences = []
    overall_stats = {
        'total_flows': len(sequences),
        'total_original_valid_packets': 0,
        'total_simulated_valid_packets': 0,
        'total_lost_packets': 0,
        'total_injected_packets': 0,
        'flows_processed': 0
    }

    print(f"\n开始模拟网络波动（保持固定长度 {target_length}）...")
    for i, seq in enumerate(sequences):
        # 分析原始序列
        original_stats = analyze_packet_sequence(seq)
        overall_stats['total_original_valid_packets'] += original_stats.get('valid_packets', 0)

        # 模拟网络波动
        simulated_seq = simulate_quic_network_fluctuation_fixed_length(
            seq,
            target_length=target_length,
            **kwargs
        )
        simulated_sequences.append(simulated_seq)

        # 分析模拟后序列
        simulated_stats = analyze_packet_sequence(simulated_seq)
        overall_stats['total_simulated_valid_packets'] += simulated_stats.get('valid_packets', 0)
        overall_stats['flows_processed'] += 1

        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(sequences)} 个流...")

    # 计算总体统计
    if overall_stats['total_original_valid_packets'] > 0:
        overall_stats['valid_packet_loss_rate'] = (
                                                          overall_stats['total_original_valid_packets'] - overall_stats[
                                                      'total_simulated_valid_packets']
                                                  ) / overall_stats['total_original_valid_packets']

    return simulated_sequences, overall_stats


def main():
    """主函数"""
    input_csv = "../../output/lori.csv"  # 修改为你的文件名
    output_simulated_csv = "./output/lori_simulated.csv"  # 模拟后的数据
    output_original_csv = "./output/lori_original.csv"  # 原始预处理后的数据

    print("=" * 60)
    print("QUIC网络波动模拟器 (输出50列CSV，无表头)")
    print("=" * 60)

    # 参数设置
    MAX_ROWS = 500  # 最大流数
    NUM_COLUMNS = 50  # 每个流的固定列数

    print(f"只处理前 {MAX_ROWS} 个流...")
    print(f"输出格式: 每行 {NUM_COLUMNS} 列，无表头")

    # 读取包序列（确保固定长度）
    sequences = read_packet_sequences_as_columns(input_csv, max_rows=MAX_ROWS, num_columns=NUM_COLUMNS)

    if not sequences:
        print("❌ 没有读取到数据，程序退出")
        return

    # 保存原始预处理后的数据到CSV（无表头）
    print(f"\n📤 保存原始预处理数据...")
    write_packet_sequences_as_columns(sequences, output_original_csv)

    print(f"\n📊 原始数据统计:")
    print(f"  总流数: {len(sequences)}")

    # 分析总体统计
    total_packets = len(sequences) * NUM_COLUMNS
    total_valid = sum(len([p for p in seq if p != 0]) for seq in sequences)
    valid_ratio = total_valid / total_packets

    print(f"  总包数: {total_packets} ({len(sequences)} × {NUM_COLUMNS})")
    print(f"  有效包数: {total_valid} ({valid_ratio:.1%})")
    print(f"  填充包数: {total_packets - total_valid}")

    # 显示前几个流的统计信息
    print(f"\n前3个流的详细信息:")
    for i in range(min(3, len(sequences))):
        stats = analyze_packet_sequence(sequences[i])
        print(f"  流 {i + 1}: 有效包数={stats['valid_packets']}/{NUM_COLUMNS} "
              f"({stats['valid_ratio']:.1%}), "
              f"前向比例: {stats['forward_ratio']:.2%}, "
              f"包大小范围: [{stats['min_size']}, {stats['max_size']}]")

    # 模拟网络波动参数
    simulation_params = {
        'loss_rate': 0.08,
        'retransmit_prob': 0.7,
        'size_jitter_ratio': 0.2,
        'mtu_drop_prob': 0.15,
        'small_pkt_inject_prob': 0.15,
        'mtu_limit': 1200
    }

    # 模拟网络波动（保持固定长度）
    simulated_sequences, overall_stats = simulate_all_sequences_fixed_length(
        sequences,
        target_length=NUM_COLUMNS,
        **simulation_params
    )

    # 输出统计信息
    print(f"\n📊 模拟结果统计:")
    print(f"  总流数: {overall_stats['total_flows']}")
    print(f"  原始有效包数: {overall_stats['total_original_valid_packets']}")
    print(f"  模拟后有效包数: {overall_stats['total_simulated_valid_packets']}")

    if overall_stats['total_original_valid_packets'] > 0:
        change = overall_stats['total_simulated_valid_packets'] - overall_stats['total_original_valid_packets']
        change_rate = change / overall_stats['total_original_valid_packets']
        print(f"  有效包数量变化: {change:+d} ({change_rate:+.1%})")
        print(f"  有效包丢失率: {overall_stats.get('valid_packet_loss_rate', 0) * 100:.1f}%")

    # 验证输出长度
    output_lengths = [len(seq) for seq in simulated_sequences]
    if all(l == NUM_COLUMNS for l in output_lengths):
        print(f"✅ 所有输出流长度均为 {NUM_COLUMNS}")
    else:
        print(f"⚠️  输出流长度不一致: 最小{min(output_lengths)}, 最大{max(output_lengths)}")

    # 写入模拟后的数据（50列格式，无表头）
    print(f"\n📤 保存模拟后的数据...")
    write_packet_sequences_as_columns(simulated_sequences, output_simulated_csv)

    print(f"\n✅ 处理完成！")
    print(f"   原始预处理数据保存到: {output_original_csv}")
    print(f"   模拟波动后数据保存到: {output_simulated_csv}")
    print(f"   输出格式: CSV文件，每行50列，无表头")


if __name__ == "__main__":
    main()