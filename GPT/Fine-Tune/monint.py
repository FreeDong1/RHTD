from openai import OpenAI
import time

API_SECRET_KEY = "xxx";
BASE_URL = "xx"; 

client = OpenAI(api_key=API_SECRET_KEY,base_url=BASE_URL)
id='xxx'
def monitor_training_job(job_id):
    """
    监控微调任务进度直到完成
    """
    print("\n开始监控微调任务进度...")
    print("-" * 50)

    while True:
        # 获取任务状态
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        # 打印任务状态
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"任务状态: {status}")

        if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
            print(f"微调模型名称: {job.fine_tuned_model}")

        # 如果任务完成或失败，退出循环
        if status in ["succeeded", "failed", "cancelled"]:
            print("-" * 50)
            break

        # 等待30秒后再次查询
        print("等待30秒后继续监控...")
        print("-" * 50)
        time.sleep(30)

    if status == "succeeded":
        print(f"✅ 微调成功！")
        print(f"微调后的模型名称: {job.fine_tuned_model}")
        return job.fine_tuned_model
    else:
        print(f"❌ 微调失败，状态为: {status}")
        if job.error:
            print(f"错误信息: {job.error.message}")
        return None

monitor_training_job(id)
