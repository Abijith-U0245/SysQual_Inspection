import cv2

def open_camera():
    backends = [
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_ANY
    ]

    for backend in backends:
        for idx in range(5):
            cap = cv2.VideoCapture(idx, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            ret, frame = cap.read()

            if ret:
                print(f"[camera_utils] Camera found: index={idx}, backend={backend}")
                return cap

            cap.release()

    raise RuntimeError(
        "No working camera found. Check USB connection and camera permissions."
    )

def read_frame(cap):
    ret, frame = cap.read()
    return ret, frame

def release_camera(cap):
    if cap is not None:
        cap.release()

def draw_overlay(frame, text):
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame