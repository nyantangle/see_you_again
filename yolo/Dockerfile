# 使用するベースイメージ
FROM ultralytics/ultralytics:latest

# 作業ディレクトリを設定
WORKDIR /usr/src/app

# 必要なパッケージをインストール
RUN pip install --no-cache-dir requests opencv-python-headless influxdb-client

# ローカルのスクリプトをコンテナにコピー
COPY capture_analyze.py ./
COPY cameras.json ./

# スクリプトを実行するコマンドを設定
CMD ["python", "capture_analyze.py"]
