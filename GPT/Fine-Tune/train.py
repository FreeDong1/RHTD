from openai import OpenAI
import json

API_SECRET_KEY = "xxx";
BASE_URL = "xxx"; 
id="xxx"
client = OpenAI(api_key=API_SECRET_KEY,base_url=BASE_URL)


def create_fine_tuning_job(training_file_id):
    """
    创建微调任务，带详细调试信息
    """
    print("正在创建微调任务...")
    print(f"使用的文件ID: {training_file_id}")

    try:
        # 创建微调任务
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model="gpt-4.1-mini-2025-04-14",
            # 如果需要可以添加超参数
            # hyperparameters={
            #     "n_epochs": 3,
            # }
        )

        # 打印完整的job对象
        print("\n=== 完整的job对象 ===")
        print(f"类型: {type(job)}")
        print(f"所有属性: {dir(job)}")
        print("\n=== job的字典表示 ===")
        print(job.model_dump() if hasattr(job, 'model_dump') else str(job))

        # 尝试获取各种可能的ID字段
        job_id = job.id if hasattr(job, 'id') else None
        if not job_id and hasattr(job, 'job_id'):
            job_id = job.job_id
        if not job_id and hasattr(job, 'task_id'):
            job_id = job.task_id

        job_status = job.status if hasattr(job, 'status') else None

        print(f"\n解析后的任务ID: {job_id}")
        print(f"解析后的状态: {job_status}")

        return job_id

    except Exception as e:
        print(f"创建微调任务时出错: {e}")
        # 打印详细的错误信息
        if hasattr(e, 'response'):
            print(f"错误响应: {e.response.text if hasattr(e.response, 'text') else e.response}")
        return None


def list_fine_tuning_jobs():
    """
    列出所有微调任务，查看API支持哪些字段
    """
    print("\n正在获取微调任务列表...")
    try:
        jobs = client.fine_tuning.jobs.list(limit=10)

        print("=== 任务列表 ===")
        for job in jobs:
            print(f"\n任务对象类型: {type(job)}")
            print(f"任务属性: {dir(job)}")
            print(f"任务ID: {getattr(job, 'id', 'N/A')}")
            print(f"任务状态: {getattr(job, 'status', 'N/A')}")
            print(f"任务模型: {getattr(job, 'model', 'N/A')}")

    except Exception as e:
        print(f"获取任务列表时出错: {e}")


def check_file_status(file_id):
    """
    检查文件状态
    """
    print("\n正在检查文件状态...")
    try:
        file_info = client.files.retrieve(file_id)
        print(f"文件ID: {file_info.id}")
        print(f"文件名: {getattr(file_info, 'filename', 'N/A')}")
        print(f"文件状态: {getattr(file_info, 'status', 'N/A')}")
        print(f"文件用途: {getattr(file_info, 'purpose', 'N/A')}")
        return file_info
    except Exception as e:
        print(f"检查文件状态时出错: {e}")
        return None


# 首先检查文件状态
print("=" * 50)
print("步骤1: 检查上传的文件状态")
print("=" * 50)
file_info = check_file_status(id)

if file_info:
    print("\n" + "=" * 50)
    print("步骤2: 尝试获取现有的微调任务列表")
    print("=" * 50)
    list_fine_tuning_jobs()

    print("\n" + "=" * 50)
    print("步骤3: 创建新的微调任务")
    print("=" * 50)
    job_id = create_fine_tuning_job(id)

    if job_id:
        print(f"\n✅ 成功创建微调任务！任务ID: {job_id}")
    else:
        print("\n❌ 创建微调任务失败，请检查API响应")
else:
    print("文件不存在或无法访问，请检查文件ID")



