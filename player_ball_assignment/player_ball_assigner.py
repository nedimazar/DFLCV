import copy
from utils import get_bbox_center, measure_distance


class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_bbox_center(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = None

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_to_players(self, tracks):
        updated_tracks = copy.deepcopy(tracks)
        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player:
                updated_tracks["players"][frame_num][assigned_player]["has_ball"] = True
        return updated_tracks
