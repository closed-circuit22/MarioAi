from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import numpy as np
from config import *


def process_frame(frame):
    gray_frame = rgb2gray(frame)
    cropped_frame = gray_frame[40:, :]
    frame_norm = cropped_frame / 255.0
    proc_frame = transform.resize(frame_norm, [100, 128])
    return proc_frame


def stack_frame(stacked_frames, state, is_new_episode):
    frame = process_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((100, 128,), dtype=np.int) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames
