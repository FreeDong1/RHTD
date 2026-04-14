#!/bin/bash

# XSS测试脚本
# 使用xss_alert.txt中的载荷测试跨站脚本漏洞

# 配置参数
TARGET_URL="https://192.168.139.141/search?q="  # 目标URL（包含查询参数）
DURATION=300                               # 测试持续时间(秒)
BASE_RATE=3                                # 基础请求速率(请求/秒)
LOG_FILE="xss_test_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="xss_alert.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
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
echo "XSS测试报告 - $(date)" > $LOG_FILE
echo "目标URL: $TARGET_URL" >> $LOG_FILE
echo "载荷文件: $PAYLOAD_FILE" >> $LOG_FILE
echo "开始时间: $(date +%T)" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# URL编码函数
urlencode() {
  local string="${1}"
  local strlen=${#string}
  local encoded=""
  local pos c o

  for (( pos=0 ; pos<strlen ; pos++ )); do
     c=${string:$pos:1}
     case "$c" in
        [-_.~a-zA-Z0-9] ) o="${c}" ;;
        * )               printf -v o '%%%02x' "'$c"
     esac
     encoded+="${o}"
  done
  echo "${encoded}"
}

# 主扫描函数
start_scan() {
  local start_time=$(date +%s)
  local end_time=$((start_time + DURATION))
  local current_time=$(date +%s)
  local request_count=0
  local potential_xss=0
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择载荷和用户代理
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±2)
    local current_rate=$((BASE_RATE + (RANDOM % 5) - 2))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # URL编码载荷
    local encoded_payload=$(urlencode "$random_payload")
    
    # 构建测试URL
    local test_url="${TARGET_URL}${encoded_payload}"
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): ${test_url:0:80}..." | tee -a $LOG_FILE
    
    # 使用curl发送请求并检查响应
    local response=$(curl -s -k -v --http3 \
      -A "$user_agent" \
      "$test_url" 2>&1)
    
    # 检查响应中是否包含未转义的payload
    echo "$response" | grep -E '^< HTTP|Content-Type:' >> $LOG_FILE
    
    if echo "$response" | grep -q -F "$random_payload"; then
      echo "[!] 可能的XSS漏洞: 载荷未转义返回" >> $LOG_FILE
      echo "原始载荷: $random_payload" >> $LOG_FILE
      ((potential_xss++))
    fi
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/3000)}")
    
    current_time=$(date +%s)
  done
  
  echo "----------------------------------------" >> $LOG_FILE
  echo "扫描完成" >> $LOG_FILE
  echo "总请求数: $request_count" >> $LOG_FILE
  echo "潜在XSS漏洞发现: $potential_xss" >> $LOG_FILE
  echo "平均速率: $((request_count / DURATION)) 请求/秒" >> $LOG_FILE
}

# 执行扫描
start_scan

# 结果摘要
echo -e "\n潜在XSS漏洞摘要:"
grep -B3 -A1 -i "可能的XSS漏洞" $LOG_FILE | sort | uniq -c | sort -nr

echo -e "\n响应状态统计:"
grep "^< HTTP" $LOG_FILE | awk '{print $2}' | sort | uniq -c | sort -nr
