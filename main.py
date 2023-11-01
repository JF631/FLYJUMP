import sys

from ljanalyzer import video as vd
import argparse
from PyQt5.QtWidgets import QApplication

from ui.MainWindow import MainWindow

def main(path:str):
    print("video analyzer started...")
    video = vd.Video(path)
    video.analyze()
    print(f"... video analyzed, output written to {video.get_output_path()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Analyzer")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    # main(args.video_path)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())