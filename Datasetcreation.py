from datetime import datetime, timedelta
from moviepy.video.io import ffmpeg_tools
def subtract_timestamps(timestamp1, timestamp2):
    time_format = "%H:%M:%S"
    dt1 = datetime.strptime(timestamp1, time_format)
    dt2 = datetime.strptime(timestamp2, time_format)
    time_difference = dt1 - dt2
    total_seconds = time_difference.total_seconds()
    return total_seconds
m_VideoLength = 21
s_VideoLength = 16
clip_start_time = "10:00:12"
clip_end_time = "10:19:59"
clipDuration = subtract_timestamps(clip_end_time, clip_start_time)
videoDuration = (m_VideoLength*60)+s_VideoLength
offSetConstant = clipDuration/videoDuration
start_time_stamp = 11
cycle = [50,35,40]
total_duration = 19+(21*60)
i = 0
while (True):
    index = i%3
    slice_end = start_time_stamp+(cycle[index]/offSetConstant)
    if (slice_end>total_duration):
        break
    clip_name = "clip"+str(i)+".mp4"
    ffmpeg_tools.ffmpeg_extract_subclip("ChikpeteTPSJunction.AVI",start_time_stamp,slice_end,clip_name)
    i+=1
    start_time_stamp = slice_end
