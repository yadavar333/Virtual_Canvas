from preReq import *

active_color = "blue"
cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

# Initialize Mediapipe and Webcam
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
c = 0
rectangles, lines, circles = [], [], []


def draw_buttons(frame, button_regions):
    """Draw buttons on the frame."""
    for button in button_regions:
        x1, y1, x2, y2 = button["coords"]
        frame = cv2.rectangle(
            frame, (x1, y1), (x2, y2), button["color"], thickness=cv2.FILLED
        )
        cv2.putText(
            frame,
            button["label"],
            (x1 + 10, y1 + 35),
            cv2.FONT_ITALIC,
            0.5,
            (0, 0, 0),
            1,
        )
    x1, y1, x2, y2 = button_regions[3]["coords"]
    cv2.putText(
        frame,
        'Press "s" to save !',
        (x1, y2 + 20),
        cv2.FONT_ITALIC,
        0.5,
        (0, 0, 0),
        1,
    )


def process_rectangle_creation(color_queues, active_color, rectangles, colors):
    """Create a rectangle from the first and last non-empty points."""
    current_color_queue = color_queues[active_color]
    non_empty_queues = [q for q in current_color_queue if q]
    if len(non_empty_queues) >= 2:
        first_queue, last_queue = non_empty_queues[-2], non_empty_queues[-1]
        x1, y1 = first_queue[0]
        x2, y2 = last_queue[-1]
        rectangles.append(
            {"coords": (x1, y1, x2, y2), "color": colors[active_color], "thickness": 2}
        )
        color_queues[active_color] = [deque(maxlen=1024)]  # Clear used queues
        color_indices[active_color] = 0


def process_line_creation(color_queues, active_color, lines, colors):
    """Create a line from the first and last non-empty points."""
    current_color_queue = color_queues[active_color]
    non_empty_queues = [q for q in current_color_queue if q]
    if len(non_empty_queues) >= 2:
        first_queue, last_queue = non_empty_queues[-2], non_empty_queues[-1]
        x1, y1 = first_queue[0]
        x2, y2 = last_queue[-1]
        lines.append(
            {
                "coords": (x1, y1, x2, y2),
                "color": colors[active_color],
                "type": cv2.LINE_AA,
                "thickness": 2,
            }
        )
        color_queues[active_color] = [deque(maxlen=1024)]  # Clear used queues
        color_indices[active_color] = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and preprocess the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons
    draw_buttons(frame, button_regions)

    # Process hands using Mediapipe
    hand_results = mp_hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = [
                (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                for lm in hand_landmarks.landmark
            ]
            fore_finger = landmarks[8]
            thumb = landmarks[4]

            # Detect thumb pinch gestures
            if (
                abs(thumb[1] - fore_finger[1]) >= 40
                or abs(fore_finger[0] - thumb[0]) >= 40
            ):
                for key in color_queues:
                    color_queues[key].append(deque(maxlen=1024))
                    color_indices[key] += 1
            elif fore_finger[1] <= 65:
                for button in button_regions:
                    x1, y1, x2, y2 = button["coords"]
                    if x1 <= fore_finger[0] <= x2:
                        if button["label"] == "Clear":
                            for key in color_queues:
                                color_queues[key] = [deque(maxlen=1024)]
                                color_indices[key] = 0
                            rectangles.clear()
                            lines.clear()
                            paint_window[:, :, :] = 255
                        elif button["label"] == "Rectangle":
                            process_rectangle_creation(
                                color_queues, active_color, rectangles, colors
                            )
                        elif button["label"] == "Line":
                            process_line_creation(
                                color_queues, active_color, lines, colors
                            )
                        else:
                            active_color = button["label"].lower()
                        break
            else:
                midpoint = (
                    (fore_finger[0] + thumb[0]) // 2,
                    (fore_finger[1] + thumb[1]) // 2,
                )
                color_queues[active_color][color_indices[active_color]].appendleft(
                    midpoint
                )

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

    # Draw lines and rectangles
    for color, points in color_queues.items():
        for queue in points:
            if len(queue) > 1:
                pts = np.array(queue, dtype=np.int32)
                cv2.polylines(
                    frame, [pts], isClosed=False, color=colors[color], thickness=2
                )
                cv2.polylines(
                    paint_window,
                    [pts],
                    isClosed=False,
                    color=colors[color],
                    thickness=2,
                )

    for rect in rectangles:
        cv2.rectangle(
            paint_window,
            rect["coords"][:2],
            rect["coords"][2:],
            rect["color"],
            rect["thickness"],
        )
        cv2.rectangle(
            frame,
            rect["coords"][:2],
            rect["coords"][2:],
            rect["color"],
            rect["thickness"],
        )

    for line in lines:
        cv2.line(
            frame,
            line["coords"][:2],
            line["coords"][2:],
            line["color"],
            line["thickness"],
            line["type"],
        )
        cv2.line(
            paint_window,
            line["coords"][:2],
            line["coords"][2:],
            line["color"],
            line["thickness"],
            line["type"],
        )

    # Display windows
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paint_window)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        file_path = os.path.join(save_folder, f"image{c}.jpg")
        success = cv2.imwrite(file_path, paint_window)
        print(f"Image saved at {file_path}" if success else "Failed to save the image.")
        c += 1
    elif key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
