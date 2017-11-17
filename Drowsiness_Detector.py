import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import playsound
from threading import Thread
import os

folder_path = "/Users/PNCHEE/Virtualenvs/tensorflow/dlib"
predictor_path = "shape_predictor_68_face_landmarks.dat"
alarm_path = "Wake-up-sounds.mp3"
image_path = "photo2.jpg"
# global variable for determine of drowsiness
ear_base = 0.28
ear_threshold_consec_frames = 15

try:
    os.chdir(folder_path)
except FileNotFoundError:
    print("\nNo such directory!\n")

# helper function
def rect_to_boundingbox(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x,y,w,h)

def shape_to_np(shape, dtype="int"):
    """
    create tuple of (x, y) coordinates.
    """
    coords = np.zeros((12, 2), dtype=dtype)
    index = 0
    for i in range(36, 48):
        coords[index] = (shape.part(i).x, shape.part(i).y)
        index += 1
    return coords

def eye_aspect_ratio(eye):
    """
    compute eye aspect ratio.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

def warning(alarm):
    playsound.playsound(alarm)

face_detector = dlib.get_frontal_face_detector()
try:
    eye_predictor = dlib.shape_predictor(predictor_path)
except RuntimeError:
    print("\nPlease double check file name of the predictor module!\n")

try:
    # local variable
    counter = 0
    cap = cv2.VideoCapture(0)
    while True: 
        # getting video stream
        ret, video = cap.read()
        video_grey = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        video_grey = cv2.resize(video_grey, (640,480))
        # detector start
        dets = face_detector(video_grey,0) 
        if len(dets) > 1:
            cv2.putText(video, "Multiple face detected, adjust camera.", (160, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif len(dets) == 1:
            preds = eye_predictor(video_grey, dets[0])
            preds = shape_to_np(preds)
            left_eye = preds[0:6]
            right_eye = preds[6:]
            for x,y in left_eye:
                cv2.circle(video, (x,y), 1, (0, 0, 255), -1)
            for x,y in right_eye:
                cv2.circle(video, (x,y), 1, (0, 0, 255), -1)
            # compute average eye aspect ratio
            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear_average = (ear_left + ear_right) / 2
            # show comparison of the ear
            cv2.putText(video, "EAR_Base: {:.2f}".format(ear_base), (400, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(video, "EAR: {:.2f}".format(ear_average), (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if ear_average < ear_base:
                counter += 1
                cv2.putText(video, "Counter: {}".format(counter), (400, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if counter >= ear_threshold_consec_frames:
                    counter = 0
                    cv2.putText(video, "Warning!!!", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    t = Thread(target=warning, args = (alarm_path,))
                    t.deamon = True
                    t.start()
            else:
                counter = 0 
        cv2.imshow("Detection", cv2.resize(video, (640,480)))
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break  
finally:
    cap.release()
    cv2.destroyAllWindows()

