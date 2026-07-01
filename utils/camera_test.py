from camera_utils import open_camera, read_frame, release_camera
import cv2

cap = open_camera()

while True:
    ret, frame = read_frame(cap)

    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

release_camera(cap)
cv2.destroyAllWindows()