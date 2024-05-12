import cv2
import os
from pathlib import Path


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_frames, output_video_path):
    # making sure the output directory exists
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        25,
        (output_frames[0].shape[1], output_frames[0].shape[0]),
    )

    for frame in output_frames:
        out.write(frame)

    out.release()
