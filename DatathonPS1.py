SOURCE_VIDEO_PATH = "subclips/clip3.mp4"
from IPython import display
import yolox

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass



import ultralytics
ultralytics.checks()


from IPython import display
import yolox
import supervision as sv
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
import supervision as sv
# from supervision.geometry.dataclasses import Point
# from supervision.video.dataclasses import VideoInfo
# from supervision.video.source import get_video_frames_generator
from supervision import VideoSink
# from supervision.notebook.utils import show_frame_in_notebook
from supervision import Detections, BoxAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator



MODEL = "yolov8x.pt"
from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()
CLASS_NAMES_DICT = model.model.names
CLASS_NAMES_DICT
from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
import os
HOME = os.getcwd()
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
TARGET_VIDEO_PATH =  f"{HOME}/analyzed.mp4"
LINE_START = sv.Point(820,180)
LINE_END = sv.Point(1600,180)
LINE_START2 = sv.Point(1150,180)
LINE_END2 = sv.Point(1150,710)

from tqdm.notebook import tqdm
bytetracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
iterator = iter(generator)
#frame = next(iterator)
box_annotator = sv.BoxAnnotator(thickness=2,text_thickness= 1,text_scale=0.5)
box_annotator2 = sv.BoxAnnotator(thickness=6,text_thickness= 6,text_scale=1)

linecounter = sv.LineZone(start= LINE_START,end= LINE_END)
linecounter2 = sv.LineZone(start= LINE_START2,end= LINE_END2)
lineAnnotator = sv.LineZoneAnnotator(thickness=2,text_thickness=1,text_scale=0.5)
# start, end = Point(x=0, y=1080), Point(x=3840, y=1080)
# line_zone = LineZone(start=start, end=end)
frameCount = 0
prev_count = 0
threshold = 4
with VideoSink(TARGET_VIDEO_PATH,video_info) as sink:
    for frame in tqdm(generator,total=video_info.total_frames):
        frameCount+=1
        results = model(frame)[0]
        indices = []
        for k,i in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
            if (i>0 and i<11):
                indices.append(k)
            else:
                pass
        indices = np.array(indices)
        detections = Detections(
            xyxy = results.boxes.xyxy.cpu().numpy()[indices],
            confidence= results.boxes.conf.cpu().numpy()[indices],
            class_id = results.boxes.cls.cpu().numpy().astype(int)[indices]
        )

        tracks = bytetracker.update(
            output_results = detections2boxes(detections=detections),
            img_info = frame.shape,
            img_size = frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections,tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {condfidence:0.2f}"
            for _,condfidence,class_id,tracker_id
            in detections
        ]
        if (frameCount%120 == 0):
            count_increase = max(linecounter2.in_count,linecounter2.out_count)-prev_count
            if (count_increase >= threshold):
                category = [1]
                labels2 = ["Good"]
            else:
                category = [0]
                labels2 = ["Bad"]
            prev_count = max(linecounter2.in_count,linecounter2.out_count)
        
        detections2= Detections(
            xyxy= np.array([[100,100,1750,1000]]),
            confidence= np.array([1.0]),
            class_id=np.array(category)
        )
        frame = box_annotator.annotate(frame=frame,detections=detections,labels = labels)
        frame = box_annotator2.annotate(frame=frame,detections=detections2,labels = labels2)

        # show_frame_in_notebook(frame,(16,16))
        linecounter.update(detections=detections)
        lineAnnotator.annotate(frame=frame,line_counter=linecounter)
        # crossed_in, crossed_out = line_zone.trigger(detections)

        linecounter2.update(detections=detections)
        lineAnnotator.annotate(frame=frame,line_counter=linecounter2)

        sink.write_frame(frame)

