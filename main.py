from src import video as vd

def main():
    print("video analyzer started...")
    video = vd.Video('C:/Users/Jakob/OneDrive/Documents/FLYJUMP_videos/vid5.mp4')
    video.analyze()
    print(f"... video analyzed, output written to {video.get_output_path()}")

if __name__ == "__main__":
    main()