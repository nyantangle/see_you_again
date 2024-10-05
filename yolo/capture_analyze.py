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
from datetime import datetime, timedelta

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

# カメラ再起動後のクールダウン時間（5分）
REBOOT_COOLDOWN_SECONDS = 300

# グローバルにエラーカウントと最終再起動時刻を保持
failure_counts = {}
last_reboot_times = {}

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

def reboot_camera(camera_url):
    """指定されたカメラを再起動する"""
    global last_reboot_times
    try:
        reboot_url = f"{camera_url}/control?var=reboot&val=0"
        response = requests.get(reboot_url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        print(f"Reboot request sent to {camera_url}")
        # 再起動時刻を記録
        last_reboot_times[camera_url] = datetime.now()
    except requests.RequestException as e:
        print(f"Failed to reboot camera {camera_url}: {e}")

def process_camera(room_name, camera_url):
    """指定されたカメラの画像を取得して分析し、InfluxDBに結果を保存する"""
    global failure_counts, last_reboot_times
    
    retries = 0
    if camera_url not in failure_counts:
        failure_counts[camera_url] = 0

    # 再起動後5分以内であれば、リクエストをスキップ
    if camera_url in last_reboot_times:
        elapsed_time = (datetime.now() - last_reboot_times[camera_url]).total_seconds()
        if elapsed_time < REBOOT_COOLDOWN_SECONDS:
            print(f"Skipping image request for {camera_url} due to recent reboot (wait {REBOOT_COOLDOWN_SECONDS - elapsed_time:.0f} seconds)")
            return

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

def reboot_all_cameras():
    """すべてのカメラを再起動する"""
    camera_list = load_camera_list(CAMERA_LIST_FILE)
    for camera in camera_list:
        reboot_camera(camera['url'])

def main():
    """メイン処理: カメラリストを読み込み、各カメラを5秒ごとに分析"""
    last_reboot_day = None  # 最後に再起動を行った日付を記録

    while True:
        current_time = datetime.now()

        # 毎日3時にすべてのカメラを再起動
        if current_time.hour == 3 and (last_reboot_day is None or last_reboot_day != current_time.date()):
            print("Rebooting all cameras at 03:00 AM")
            reboot_all_cameras()
            last_reboot_day = current_time.date()  # 再起動を行った日付を記録

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
