from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner


def main():
    video_frames = read_video("input_videos/short.mp4")

    # Initializing a tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=False, stub_path="stubs/short.pkl"
    )

    # Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    tracks = team_assigner.assign_teams_to_players(video_frames, tracks)

    # Draw object tracks
    output_video_frames = tracker.annotate_frames(
        video_frames=video_frames, tracks=tracks
    )

    save_video(output_video_frames, "output_videos/short.avi")


if __name__ == "__main__":
    main()
