from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# sys.path.append("../")
from utils import get_bbox_center, get_bbox_width


class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolating missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames, batch_size=1):
        detections = []
        for i in tqdm(
            range(0, len(frames), batch_size), desc="Object Detection Inference"
        ):
            batch_detections = self.model.predict(
                frames[i : i + batch_size], conf=0.1, verbose=False
            )
            detections += batch_detections

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # We want to get rid of the goakleeper class and treat it as a normal player
            for i, cls_index in enumerate(detection_supervision.class_id):
                if cls_names[cls_index] == "goalkeeper":
                    detection_supervision.class_id[cls_index] = cls_names_inv["player"]

            # Tracking objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            img=frame,
            center=(x_center, y2),
            # axes=(int(width), int(0.35 * width)),
            axes=(int(50), int(0.35 * 50)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width // 2)
        x2_rect = int(x_center + rectangle_width // 2)
        y1_rect = int((y2 - rectangle_height // 2) + 15)
        y2_rect = int((y2 + rectangle_height // 2) + 15)

        if track_id:
            cv2.rectangle(
                img=frame,
                pt1=(x1_rect, y1_rect),
                pt2=(x2_rect, y2_rect),
                color=color,
                thickness=cv2.FILLED,
            )

            x1_text = int(x1_rect + 12)
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                img=frame,
                text=f"{track_id}",
                org=(x1_text, y1_rect + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=2,
            )

        return frame

    def draw_triangle(self, frame, bbox):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 15],
                [x + 10, y - 15],
            ]
        )

        # Drawing triangle fill
        cv2.drawContours(
            image=frame,
            contours=[triangle_points],
            contourIdx=0,
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )

        # Drawing triangle border
        cv2.drawContours(
            image=frame,
            contours=[triangle_points],
            contourIdx=0,
            color=(0, 0, 0),
            thickness=2,
        )

        return frame

    def annotate_frames(self, video_frames, tracks):
        output_video_frames = []

        for i, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][i]
            referee_dict = tracks["referees"][i]
            ball_dict = tracks["ball"][i]

            # Draw circles for players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw circles for referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 0, 0))

            # Draw symbol
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"])

            output_video_frames.append(frame)
        return output_video_frames
