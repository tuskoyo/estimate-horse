#!/bin/bash
# シェルスクリプト：position 内の全ての該当CSVファイルに対して add_label.py を実行し、
# 出力ファイルを train/記録場所-ID.csv の形式で保存する

# --- hdd1 ---
# c100: 出産時刻 22:44 → target_time 2244
python add_label.py position/hdd1-bigred-c100-2024-05-10.csv train/hdd1-c100-2024-05-10.csv 2244 &
# c101: 出産時刻 04:45 → target_time 0445
python add_label.py position/hdd1-bigred-c101-2024-05-10.csv train/hdd1-c101-2024-05-10.csv 0445 &
# c102: 出産時刻 21:08 → target_time 2108
python add_label.py position/hdd1-bigred-c102-2024-05-10.csv train/hdd1-c102-2024-05-10.csv 2108 &

# --- hdd5 (bigred-c100) ---
# 2024-05-25: 出産時刻 22:47 → target_time 2247
python add_label.py position/hdd5-bigred-c100-2024-05-25.csv train/hdd5-c100-2024-05-25.csv 2247 &
# 2024-05-27: 出産時刻 05:16 → target_time 0516
python add_label.py position/hdd5-bigred-c100-2024-05-27.csv train/hdd5-c100-2024-05-27.csv 0516 &
# 2024-06-02: 出産時刻 23:22 → target_time 2322
python add_label.py position/hdd5-bigred-c100-2024-06-02.csv train/hdd5-c100-2024-06-02.csv 2322 &

# --- hdd5 (bigred-c101) ---
# 2024-05-30: 出産時刻 02:21 → target_time 0221
python add_label.py position/hdd5-bigred-c101-2024-05-30.csv train/hdd5-c101-2024-05-30.csv 0221 &
# 2024-06-02: 出産時刻 03:31 → target_time 0331
python add_label.py position/hdd5-bigred-c101-2024-06-02.csv train/hdd5-c101-2024-06-02.csv 0331 &
# 2024-06-03: 出産時刻 07:48 → target_time 0748
python add_label.py position/hdd5-bigred-c101-2024-06-03.csv train/hdd5-c101-2024-06-03.csv 0748 &

# --- hdd6 (bigred-c100) ---
# 2024-02-28: 出産時刻 23:55 → target_time 2355
python add_label.py position/hdd6-bigred-c100-2024-02-28.csv train/hdd6-c100-2024-02-28.csv 2355 &
# 2024-03-08: 出産時刻 20:16 → target_time 2016
python add_label.py position/hdd6-bigred-c100-2024-03-08.csv train/hdd6-c100-2024-03-08.csv 2016 &
# 2024-03-11: 出産時刻 23:08 → target_time 2308
python add_label.py position/hdd6-bigred-c100-2024-03-11.csv train/hdd6-c100-2024-03-11.csv 2308 &
# 2024-03-29: 出産時刻 20:37 → target_time 2037
# ※ hdd6 系統に該当ファイルが無いため、hdd6_v2 系統のファイルを使用
python add_label.py position/hdd6_v2-bigred-c100-2024-03-29.csv train/hdd6-c100-2024-03-29.csv 2037 &

# --- その他 ---
# hdd2、hdd4、hdd7 のファイルは「出産無し」または対象外のため、処理しません

# 全てのバックグラウンドプロセスの終了を待機
wait
echo "全ての処理が完了しました。"
