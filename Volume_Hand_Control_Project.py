import cv2
import time
import numpy as np
from Hand_Tracking_Module import HandDetectorMP
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

h_cam, w_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, h_cam)
cap.set(4, w_cam)
p_time = 0

detector = HandDetectorMP(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
vol_bar = 450
vol_per = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, (160, 170, 100), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (160, 170, 100), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (160, 170, 100), 3)
        cv2.circle(img, (cx, cy), 7, (160, 170, 100), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand Range 50-300
        # Volume Range -65 - 0

        vol = np.interp(length, [50, 300], [min_vol, max_vol])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 7, (150, 70, 20), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.rectangle(img, (50, 150), (85, 400), (160, 200, 50), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (160, 200, 50), cv2.FILLED)
    cv2.putText(img, f'Vol: {int(vol_per)} %', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (150, 200, 110), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# https://www.youtube.com/watch?v=01sAkU_NvOY
