import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


def upload_file_to_s3(local_file: Path, s3_key: str, bucket: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    s3.upload_file(str(local_file), bucket, s3_key)
    print(f"☁️ Uploaded file: {s3_key}")


def download_file_from_s3(s3_key: str, local_path: Path, bucket: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))
    print(f"⬇️ Downloaded file: {s3_key}")


def upload_dir_to_s3(local_dir: Path, s3_prefix: str, bucket):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    for file in local_dir.rglob("*"):
        if file.is_file():
            s3_key = f"{s3_prefix}/{file.relative_to(local_dir)}"
            s3.upload_file(str(file), bucket, s3_key)
            print(f"✅ Uploaded: {s3_key}")


def download_dir_from_s3(s3_prefix: str, local_dir: Path, bucket: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

    downloaded = 0
    for page in pages:
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = Path(s3_key).relative_to(s3_prefix)
            dest_path = local_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, s3_key, str(dest_path))
            print(f"⬇️ Downloaded: {s3_key}")
            downloaded += 1

    if downloaded == 0:
        print("⚠️ No files found to download.")
    else:
        print(f"✅ Downloaded {downloaded} files to {local_dir}")
