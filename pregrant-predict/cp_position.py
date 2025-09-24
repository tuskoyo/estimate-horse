#!/usr/bin/env python3
import os
import shutil
import argparse
import sys

def main(input_dir, output_dir):
    # 入力ディレクトリが存在するか確認
    if not os.path.isdir(input_dir):
        print(f"Error: 入力ディレクトリ '{input_dir}' が存在しません。")
        sys.exit(1)

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    exclude_folder = 'pose-estimatoin'

    # 入力ディレクトリ以下を再帰的に探索
    for root, dirs, files in os.walk(input_dir):
        if 'position.csv' in files:
            source_file = os.path.join(root, 'position.csv')

            # 入力ディレクトリからの相対パスを取得
            rel_dir = os.path.relpath(root, input_dir)

            # 相対パスが '.'の場合は入力ディレクトリ名を使用
            if rel_dir == '.':
                base_name = os.path.basename(os.path.normpath(input_dir))
            else:
                # 相対パスを分割し、指定のフォルダ名を除外して再結合
                path_parts = [part for part in rel_dir.split(os.sep) if part != exclude_folder]
                base_name = '-'.join(path_parts)

            destination_file = os.path.join(output_dir, base_name + ".csv")

            # コピー先に同名のファイルが存在する場合はコピーをスキップ
            if os.path.exists(destination_file):
                print(f"Duplicate file name detected: '{destination_file}' already exists. "
                      f"Skipping copy for '{source_file}'.")
                continue

            print(f"Copying '{source_file}' to '{destination_file}'")
            shutil.copy2(source_file, destination_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input_dir 内の position.csv を探索し、"
                                                 "出力ディレクトリにリネームしてコピーします。")
    parser.add_argument('input_dir', help="入力ディレクトリのパスを指定します")
    parser.add_argument('output_dir', help="出力ディレクトリのパスを指定します")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
