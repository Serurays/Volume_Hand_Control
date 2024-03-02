import cv2
import mediapipe as mp
import time

from google.protobuf.json_format import MessageToDict


class HandDetectorMP:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        type = ""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:

            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms,
                                                self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_no]
            for l_id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([l_id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (200, 100, 200), cv2.FILLED)

        return lm_list


def main():
    p_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetectorMP()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img, draw=False)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,
                   (100, 200, 200), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

# https://www.youtube.com/watch?v=01sAkU_NvOY
