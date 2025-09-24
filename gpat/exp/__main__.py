import argparse
import os
import sys

from gpat.exp.exp import run
from gpat.utils.config import read_config
from gpat.utils.extensions import MEDIA_EXTENSIONS
from gpat.utils.gpat2fpat import gpat2fpat


def main():
    # 設定ファイルの読み込み
    config = read_config()
    pose_model = os.path.expanduser(config["exp-setting"]["pose_model"])
    pose_checkpoint = os.path.expanduser(config["exp-setting"]["pose_checkpoint"])
    det_model = os.path.expanduser(config["exp-setting"]["det_model"])
    det_checkpoint = os.path.expanduser(config["exp-setting"]["det_checkpoint"])
    
    # プログラム全体の説明を設定
    description = "GPAT: Golf Player Analysis Tool"
    args_parser = argparse.ArgumentParser(description=description)
    
    args_parser.add_argument("-i", "--input", type=str, required=True, help="input video file path")
    args_parser.add_argument("-o", "--output", type=str, default=os.getcwd(), help="output directory path")
    args_parser.add_argument("-pm", "--pose-model", type=str, default=pose_model, help="pose model path")
    args_parser.add_argument("-pc", "--pose-config", type=str, default=pose_checkpoint, help="pose config path")
    args_parser.add_argument("-dm", "--det-model", type=str, default=det_model, help="detection model path")
    args_parser.add_argument("-dc", "--det-config", type=str, default=det_checkpoint, help="detection config path")
    args_parser.add_argument("-f", "--fpat", action="store_true", default=False, help="convert GPAT data to FPAT data")
    args_parser.add_argument("-s", "--save_img", action="store_true", default=False, help="save images")
    
    # 引数を解析
    args = args_parser.parse_args()
    if os.path.isfile(args.input):
        run(
            video_path=args.input,
            output_path=args.output,
            pose_model=args.pose_model,
            pose_checkpoint=args.pose_config,
            det_model=args.det_model,
            det_checkpoint=args.det_config,
            save_img=args.save_img
        )
        video_name = os.path.basename(args.input).split(".")[0]
        if args.fpat:
            gpat2fpat(os.path.join(args.output, "data", video_name))
    elif os.path.isdir(args.input):
        for file_ in os.listdir(args.input):
            ex = file_.split(".")[-1]
            if os.path.isdir(os.path.join(args.input, file_)):
                continue  # ignore directory
            if ex in MEDIA_EXTENSIONS:
                base_name = os.path.basename(file_).split(".")[0]
                if os.path.exists(os.path.join(args.output, base_name)):
                    print(f"{base_name} already exists in {args.output}")
                    continue
                run(
                    video_path=os.path.join(args.input, file_),
                    output_path=args.output,
                    pose_model=args.pose_model,
                    pose_checkpoint=args.pose_config,
                    det_model=args.det_model,
                    det_checkpoint=args.det_config,
                    save_img=args.save_img
                )
                video_name = os.path.basename(file_).split(".")[0]
                if args.fpat:
                        gpat2fpat(os.path.join(args.output, "data", video_name))

if __name__ == "__main__":
    main()