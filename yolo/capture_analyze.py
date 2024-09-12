import os
import requests
import cv2
import numpy as np
import time
import json
from influxdb_client import InfluxDBClient, Point
from requests.exceptions import Timeout, RequestException
from ultralytics import YOLO
import threading

# 環境変数から設定を読み込む
influx_url = os.getenv("INFLUXDB_URL")
bucket = os.getenv("INFLUXDB_BUCKET")
org = os.getenv("INFLUXDB_ORG")
token = os.getenv("INFLUXDB_TOKEN")
#webhook_url = os.getenv("SLACK_WEBHOOK_URL")

client = InfluxDBClient(url=influx_url, token=token, org=org)
write_api = client.write_api()

# Load the YOLO model
model = YOLO("yolov8n.pt")

# カメラ情報を含むJSONファイルのパス
CAMERA_LIST_FILE = "cameras.json"

# タイムアウトとリトライ設定
TIMEOUT_SECONDS = 5
MAX_RETRIES = 3
INTERVAL_SECONDS = 5

# グローバルにエラーカウントを保持
failure_counts = {}

def load_camera_list(file_path):
    """JSONファイルからカメラリストを読み込む"""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading camera list file: {str(e)}")
        return []

def count_person(image):
    """人数をカウントするための関数"""
    result = model(image)
    boxes = result[0].boxes  # Bounding box outputs
    labels = boxes.cls  # Class values of the boxes
    n_person = (labels == 0).sum().item()  # 0は人のクラスID
    return n_person

def process_camera(room_name, camera_url):
    """指定されたカメラの画像を取得して分析し、InfluxDBに結果を保存する"""
    global failure_counts
    
    retries = 0
    if camera_url not in failure_counts:
        failure_counts[camera_url] = 0

    while retries < MAX_RETRIES:
        try:
            response = requests.get(camera_url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()

            image_data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"Failed to decode image from {camera_url}")

            # 人数をカウント
            num_people = count_person(image)

            # InfluxDBにデータを書き込む
            point = Point("people_count") \
                        .tag("room", room_name) \
                        .field("count", num_people)
            write_api.write(bucket=bucket, record=point)

            print(f"Room: {room_name}, People count: {num_people}")
            failure_counts[camera_url] = 0
            break

        except (Timeout, RequestException) as e:
            retries += 1
            print(f"Error accessing {camera_url}: {str(e)}. Retrying ({retries}/{MAX_RETRIES})...")
            time.sleep(1)

            failure_counts[camera_url] += 1
            #if failure_counts[camera_url] >= 20:
            #    send_slack_notification(camera_url)

        except ValueError as e:
            print(f"Error processing image from {camera_url}: {str(e)}")
            break

    if retries == MAX_RETRIES:
        print(f"Max retries reached for {camera_url}. Skipping this camera.")

#def send_slack_notification(camera_url):
#    """Slackに通知を送信する"""
#    if webhook_url:
#        message = {
#            "text": f"Access to the camera at {camera_url} has failed more than 20 times."
#        }
#        try:
#            requests.post(webhook_url, json=message)
#        except RequestException as e:
#            print(f"Error sending Slack notification: {str(e)}")

def main():
    """メイン処理: カメラリストを読み込み、各カメラを5秒ごとに分析"""
    while True:
        camera_list = load_camera_list(CAMERA_LIST_FILE)
        if not camera_list:
            print("No cameras to process. Retrying in 60 seconds.")
            time.sleep(60)
            continue

        start_time = time.time()
        threads = []

        for camera in camera_list:
            room_name = camera['room']
            camera_url = camera['url']
            thread = threading.Thread(target=process_camera, args=(room_name, camera_url))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        elapsed_time = time.time() - start_time
        sleep_time = max(0, INTERVAL_SECONDS - elapsed_time)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
