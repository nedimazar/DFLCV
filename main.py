from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner


def main():
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initializing a tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stup=True, stub_path="stubs/track_stubs.pkl"
    )

    # Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # Draw object tracks
    output_video_frames = tracker.annotate_frames(
        video_frames=video_frames, tracks=tracks
    )

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
