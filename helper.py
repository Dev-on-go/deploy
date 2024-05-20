from ultralytics import YOLO
import time
import streamlit as st
import cv2
import os
from pytube import YouTube
<<<<<<< Updated upstream
import supervision as sv
import ffmpeg
=======
from pathlib import Path
from datetime import datetime, timedelta
from moviepy.video.io import ffmpeg_tools
>>>>>>> Stashed changes

import settings
import argparse
import json
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from inference import get_roboflow_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from utils.timers import ClockBasedTimer

import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]
violations = []
displayed={}
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
current_mouse_position: Optional[Tuple[int, int]] = None

class CustomSink:
    def __init__(self, zone_configuration_path: str, classes, violation_time: int):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]
        self.violation_time = violation_time

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = sv.Detections.from_inference(result)
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                if tracker_ID not in displayed:
                    if(time%60 >= int(self.violation_time)):
                        violations.append(tracker_ID)
                        str = tracker_ID + " " + cl + " Location: CrossingX "
                        st.warning(str, icon= "⚠️")
                        displayed[tracker_ID] = 1 
                
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)
        
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def enchroachment():
    source_vid = st.sidebar.selectbox(
    "Choose a video...", settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))
    time = st.sidebar.text_input("Violation Time:")
    source_url = st.sidebar.text_input("Source Url:")
    
    if st.sidebar.button("Generate Bottleneck Alerts"):
        if(source_url):
            livedetection(source_url=source_url, violation_time=int(time), zone_configuration_path="configure/config.json")
        else:
            drawzones(source_path = source_path, zone_configuration_path = "configure/config.json")
            timedetect(source_path = source_path, zone_configuration_path = "configure/config.json", violation_time=time)
        

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

<<<<<<< Updated upstream
def drawzones(source_path, zone_configuration_path):
    
    def resolve_source(source_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(source_path):
            return None

        image = cv2.imread(source_path)
        if image is not None:
            return image

        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        frame = next(frame_generator)
        return frame
    
    def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
        global current_mouse_position
        if event == cv2.EVENT_MOUSEMOVE:
            current_mouse_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            POLYGONS[-1].append((x, y))
    
    def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
        global POLYGONS, current_mouse_position
        image[:] = original_image.copy()
        for idx, polygon in enumerate(POLYGONS):
            color = (
                COLORS.by_idx(idx).as_bgr()
                if idx < len(POLYGONS) - 1
                else sv.Color.WHITE.as_bgr()
            )

            if len(polygon) > 1:
                for i in range(1, len(polygon)):
                    cv2.line(
                        img=image,
                        pt1=polygon[i - 1],
                        pt2=polygon[i],
                        color=color,
                        thickness=THICKNESS,
                    )
                if idx < len(POLYGONS) - 1:
                    cv2.line(
                        img=image,
                        pt1=polygon[-1],
                        pt2=polygon[0],
                        color=color,
                        thickness=THICKNESS,
                    )
            if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=current_mouse_position,
                    color=color,
                    thickness=THICKNESS,
                )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, image)

    def redraw_polygons(image: np.ndarray) -> None:
        for idx, polygon in enumerate(POLYGONS[:-1]):
            if len(polygon) > 1:
                color = COLORS.by_idx(idx).as_bgr()
                for i in range(len(polygon) - 1):
                    cv2.line(
                        img=image,
                        pt1=polygon[i],
                        pt2=polygon[i + 1],
                        color=color,
                        thickness=THICKNESS,
                    )
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )

    def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
        if len(POLYGONS[-1]) > 2:
            cv2.line(
                img=image,
                pt1=POLYGONS[-1][-1],
                pt2=POLYGONS[-1][0],
                color=COLORS.by_idx(0).as_bgr(),
                thickness=THICKNESS,
            )
        POLYGONS.append([])
        image[:] = original_image.copy()
        redraw_polygons(image)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, image)
    
    def save_polygons_to_json(polygons, target_path):
        data_to_save = polygons if polygons[-1] else polygons[:-1]
        with open(target_path, "w") as f:
            json.dump(data_to_save, f)
    
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, zone_configuration_path)
            print(f"Polygons saved to {zone_configuration_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()

def timedetect(source_path, zone_configuration_path, violation_time):
    COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
    COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
    )
    model_id = "yolov8x-640"
    classes = [2,5,6,7]
    confidence = 0.3
    iou = 0.7
    model = get_roboflow_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    frames_generator = sv.get_video_frames_generator(source_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    vid_cap = cv2.VideoCapture(source_path)
    st_frame = st.empty()
    while(vid_cap.isOpened()):
            success = vid_cap.read()
            st.subheader("ALERTS: ")
            if success:
                    for frame in frames_generator:
                        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
                        detections = sv.Detections.from_inference(results)
                        detections = detections[find_in_list(detections.class_id, classes)]
                        detections = tracker.update_with_detections(detections)

                        annotated_frame = frame.copy()

                        for idx, zone in enumerate(zones):
                            annotated_frame = sv.draw_polygon(
                                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                            )

                            detections_in_zone = detections[zone.trigger(detections)]
                            time_in_zone = timers[idx].tick(detections_in_zone)
                            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                            annotated_frame = COLOR_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                custom_color_lookup=custom_color_lookup,
                            )
                            labels = [
                                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                            ]

                            annotated_frame = LABEL_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                labels=labels,
                                custom_color_lookup=custom_color_lookup,
                            )
                            
                            for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                                if tracker_ID not in displayed:
                                    if(time%60 >= int(violation_time)):
                                        violations.append(tracker_ID)
                                        cla = settings.CLASSES[cl]
                                        s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: CrossingX "
                                        st.warning(s, icon= "⚠️")
                                        displayed[tracker_ID] = 1
                        
                        st_frame.image(annotated_frame,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
                    vid_cap.release()
                    cv2.destroyAllWindows()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
                    
            
    
def livedetection(source_url: str, violation_time: int, zone_configuration_path: str):
    model_id = 'yolov8x-640'
    classes = [2,5,6,7]
    confidence = 0.3
    iou = 0.7
    model = YOLO('weights\yolov8n.pt')
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes, violation_time = violation_time)

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=source_url,
        on_prediction=sink.on_prediction,
        confidence=confidence,
        iou_threshold=iou,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()
=======
def input_video(conf, model):
    #User selects a video file using Streamlit's file_uploader

    video_source = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov"))
    
    # If a video file is selected
    if video_source is not None:
        # Get the path of the selected video file
        print(video_source.name)
        video_path = os.path.abspath(video_source.name)
        print(video_path)
        
        # Process the video here
        st.success(f"Selected video: {video_path}")
        
        # Datasetcreation.py
        
        # Function to subtract timestamps and calculate time difference in seconds
        def subtract_timestamps(timestamp1, timestamp2):
            time_format = "%H:%M:%S"
            dt1 = datetime.strptime(timestamp1, time_format)
            dt2 = datetime.strptime(timestamp2, time_format)
            time_difference = dt1 - dt2
            total_seconds = time_difference.total_seconds()
            return total_seconds
        
        # Define video length in minutes and seconds
        m_VideoLength = 21
        s_VideoLength = 16
        
        # Define start and end timestamps for the desired clip
        clip_start_time = "10:00:12"
        clip_end_time = "10:19:59"
        
        # Calculate the duration of the clip in seconds
        clipDuration = subtract_timestamps(clip_end_time, clip_start_time)
        
        # Calculate the total duration of the video in seconds
        videoDuration = (m_VideoLength * 60) + s_VideoLength
        
        # Calculate the offset constant for slicing the video
        offSetConstant = clipDuration / videoDuration
        
        # Define the starting timestamp for slicing the video
        start_time_stamp = 11
        
        # Define a cycle of durations for each clip
        cycle = [50, 35, 40]
        
        # Define the total duration of all clips
        total_duration = 19 + (21 * 60)
        
        # Initialize a counter variable
        i = 0
        
        # Loop to slice the video into multiple clips
        while True:
            # Get the index of the current cycle duration
            index = i % 3
            
            # Calculate the end timestamp for the current clip
            slice_end = start_time_stamp + (cycle[index] / offSetConstant)
            
            # If the end timestamp exceeds the total duration, break the loop
            if slice_end > total_duration:
                break
            
            # Define the name of the current clip
            clip_name = f"clip{i}.mp4"
            
            # Define the input video path
            input_video = f"{video_path}"
            
            # Create a folder named subclips if it doesn't exist
            subclips_folder = "subclips"
            if not os.path.exists(subclips_folder):
                os.makedirs(subclips_folder)
            
            # Use ffmpeg to extract the subclip from the input video
            output_path = os.path.join(subclips_folder, clip_name)
            ffmpeg_tools.ffmpeg_extract_subclip(input_video, start_time_stamp, slice_end, output_path)
            
            # Increment the counter variable and update the start timestamp
            i += 1
            start_time_stamp = slice_end
    else:
        st.warning("Please choose a video file.")
>>>>>>> Stashed changes
