from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initializing a tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stup=True, stub_path="stubs/track_stubs.pkl"
    )

    # Draw object tracks
    output_video_frames = tracker.annotate_frames(
        video_frames=video_frames, tracks=tracks
    )

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
