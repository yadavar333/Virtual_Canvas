import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os


# Define color queues for drawing
color_queues = {
    "blue": [deque(maxlen=1024)],
    "green": [deque(maxlen=1024)],
    "red": [deque(maxlen=1024)],
    "yellow": [deque(maxlen=1024)],
}

color_indices = {"blue": 0, "green": 0, "red": 0, "yellow": 0}

save_folder = "saved_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Predefined color values and active color
colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}

# Initialize canvas
canvas_width, canvas_height = 1280, 720
paint_window = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
button_regions = [
    {"label": "Clear", "coords": (40, 1, 140, 65), "color": (255, 255, 255)},
    {"label": "Blue", "coords": (160, 1, 255, 65), "color": (255, 0, 0)},
    {"label": "Green", "coords": (275, 1, 370, 65), "color": (0, 255, 0)},
    {"label": "Red", "coords": (390, 1, 485, 65), "color": (0, 0, 255)},
    {"label": "Yellow", "coords": (505, 1, 600, 65), "color": (0, 255, 255)},
    {"label": "Rectangle", "coords": (610, 1, 700, 65), "color": (255, 255, 255)},
    {"label": "Line", "coords": (710, 1, 805, 65), "color": (255, 255, 255)},
]
