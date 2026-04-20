#!/bin/bash

# Open Redirect 测试脚本
# 使用open_redirect_wordlist.txt中的载荷测试开放重定向漏洞

# 配置参数
TARGET_DOMAIN="192.168.139.141"  # 目标域名
BASE_PATH="/"                         # 基础路径
PROTOCOL="https"                      # 使用http或https
DURATION=300                          # 测试持续时间(秒)
BASE_RATE=3                           # 基础请求速率(请求/秒)
LOG_FILE="open_redirect_test_$(date +%Y%m%d_%H%M%S).log"
PAYLOAD_FILE="open_redirect_wordlist.txt"

# 用户代理列表
USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
  "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
  "curl/7.68.0"
)

# 常见重定向参数
REDIRECT_PARAMS=(
  "url" "next" "redirect" "redir" "rurl" "destination" "return" "returnTo" "return_to"
  "callback" "forward" "forwardTo" "jump" "target" "view" "page" "file" "load"
)

# 检查载荷文件是否存在
if [ ! -f "$PAYLOAD_FILE" ]; then
  echo "错误: 载荷文件 $PAYLOAD_FILE 不存在!"
  exit 1
fi

# 读取载荷文件
mapfile -t PAYLOADS < "$PAYLOAD_FILE"

# 初始化日志文件
echo "Open Redirect 测试报告 - $(date)" > $LOG_FILE
echo "目标域名: $TARGET_DOMAIN" >> $LOG_FILE
echo "协议: $PROTOCOL" >> $LOG_FILE
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
    # 随机选择载荷和用户代理
    local random_payload=${PAYLOADS[$RANDOM % ${#PAYLOADS[@]}]}
    local user_agent=${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}
    
    # 随机请求速率(基础速率±2)
    local current_rate=$((BASE_RATE + (RANDOM % 5) - 2))
    [ $current_rate -lt 1 ] && current_rate=1
    
    # 构建测试URL
    local test_url=""
    local test_case=$((RANDOM % 3))
    
    case $test_case in
      0) # 直接作为路径
        test_url="$PROTOCOL://$TARGET_DOMAIN$random_payload"
        ;;
      1) # 作为单个参数值
        local param=${REDIRECT_PARAMS[$RANDOM % ${#REDIRECT_PARAMS[@]}]}
        test_url="$PROTOCOL://$TARGET_DOMAIN$BASE_PATH?$param=$random_payload"
        ;;
      2) # 作为多个参数值
        local param1=${REDIRECT_PARAMS[$RANDOM % ${#REDIRECT_PARAMS[@]}]}
        local param2=${REDIRECT_PARAMS[$RANDOM % ${#REDIRECT_PARAMS[@]}]}
        test_url="$PROTOCOL://$TARGET_DOMAIN$BASE_PATH?$param1=$random_payload&$param2=$random_payload"
        ;;
    esac
    
    # 发送请求并检查重定向
    echo "[$(date +%T)] 测试 #$((++request_count)): ${test_url:0:80}..." | tee -a $LOG_FILE
    
    curl -k -L -v -s --http3\
      -A "$user_agent" \
      -H "X-Request-ID: $(date +%s%N)-$RANDOM" \
      --connect-timeout 5 \
      --max-time 10 \
      "$test_url" 2>&1 | grep -E '^< HTTP|^< Location:|Host: example.com|301 Moved Permanently|302 Found|303 See Other|307 Temporary Redirect|308 Permanent Redirect' >> $LOG_FILE
    
    # 随机延迟控制速率
    sleep $(awk "BEGIN {print (1/$current_rate) + (($RANDOM%1000)/3000}")
    
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
echo -e "\n潜在开放重定向漏洞摘要:"
grep -B2 -i "Location:.*example.com" $LOG_FILE | grep -E '测试 #|Location:' | sort | uniq -c | sort -nr
