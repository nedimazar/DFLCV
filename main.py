from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from team_possession import TeamPossession


def main():
    input_video_name = "DFL.mp4"
    base_name = input_video_name.split(".")[0]
    input_video_path = f"input_videos/{input_video_name}"
    stub_path = f"stubs/{base_name}.pkl"
    output_video_path = f"output_videos/{base_name}.avi"

    video_frames = read_video(input_video_path)

    # Initializing a tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, stub_path=stub_path)

    # Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    tracks = team_assigner.assign_teams_to_players(video_frames, tracks)

    # Assign ball possession
    player_ball_assigner = PlayerBallAssigner()
    tracks = player_ball_assigner.assign_ball_to_players(tracks)

    # Get team possession
    team_possesion_assigner = TeamPossession()
    team_possession = team_possesion_assigner.get_team_possession(tracks)

    # Draw object tracks
    output_video_frames = tracker.annotate_frames(
        video_frames=video_frames, tracks=tracks, team_possession=team_possession
    )

    save_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()
