#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(
        description="指定時間の1時間前のデータにラベルを付与するプログラム"
    )
    parser.add_argument("input_csv_file", help="入力CSVファイルのパス")
    parser.add_argument("output_csv_file", help="出力CSVファイルのパス")
    parser.add_argument("target_time", help="基準時間（hhmm形式、例: 0700 または 9999（全て0にする場合））")
    args = parser.parse_args()

    # CSVファイルの読み込み（インデックスは1列目とする）
    df = pd.read_csv(args.input_csv_file, index_col=0)

    # インデックスがファイルパスの場合、最後の部分を取り出して".jpg"を除去
    # 例: "bigred-c100/2024-03-16/03_16_31.451172.jpg" -> "03_16_31.451172"
    df.index = df.index.to_series().apply(lambda x: x.split('/')[-1].replace('.jpg', ''))

    # 取り出した時刻文字列を、"HH_MM_SS.%f"形式としてdatetimeに変換（基準日は1900-01-01となる）
    df.index = pd.to_datetime(df.index, format="%H_%M_%S.%f")

    if args.target_time == "9999":
        # target_timeが9999の場合はすべてのlabelを0に設定
        df['label'] = 0
    else:
        # 入力のhhmm形式から基準時刻を作成（基準日も1900-01-01とする）
        target_hour = int(args.target_time[:2])
        target_minute = int(args.target_time[2:])
        base_date = datetime(1900, 1, 1)
        target_datetime = base_date.replace(hour=target_hour, minute=target_minute)

        # 基準時刻の1時間前の時間範囲を設定（[target_datetime - 1時間, target_datetime)）
        start_time = target_datetime - timedelta(hours=1)
        end_time = target_datetime

        # 指定した時間範囲内の行は1、それ以外は0
        df['label'] = ((df.index >= start_time) & (df.index < end_time)).astype(int)

    # インデックスから日付部分を除き、時間のみを保持
    df.index = df.index.time

    # インデックス（時間）を昇順にソート
    df = df.sort_index()

    # 結果をCSVファイルとして出力
    os.makedirs(os.path.dirname(args.output_csv_file), exist_ok=True)
    output_file = args.output_csv_file
    df.to_csv(output_file)
    print(f"出力CSVファイル: {output_file}")

if __name__ == "__main__":
    main()
