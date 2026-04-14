import json
from openai import OpenAI
API_SECRET_KEY = "xxx";
BASE_URL = "xxx"; 
client = OpenAI(api_key=API_SECRET_KEY,base_url=BASE_URL)
file="all_http3_training.jsonl"

def upload_training_file(file_path):
    with open(file_path, "rb") as file:
        training_file = client.files.create(
            file=file,
            purpose="fine-tune"
        )

    training_file_id = training_file.id
    print(f"文件上传成功！文件ID: {training_file_id}")

upload_training_file(file)

