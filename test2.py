from src import video as vd


def main():
    print("hi")
    video = vd.Video('C:/Users/Jakob/OneDrive/Documents/FLYJUMP_videos/vid5.mp4')
    video.analyze()

    
    # video.load()
    # video.play()
    print("done")

if __name__ == "__main__":
    main()