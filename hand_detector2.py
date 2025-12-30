import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, max_hands=2,
                 detection_con=0.5, track_con=0.5):
        
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None


    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, handLms, self.mp_hands.HAND_CONNECTIONS
                    )
        return img


    def find_position(self, img, draw=True):
        all_landmarks = []

        if self.results and self.results.multi_hand_landmarks:
            height, width, _ = img.shape   # ✅ FIXED HERE

            for hand_landmarks in self.results.multi_hand_landmarks:
                landmark_list = []

                for id, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 4, (255, 255, 255), cv2.FILLED)

                all_landmarks.append(("Unknown", landmark_list))

        return all_landmarks
