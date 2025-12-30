import hand_detector2 as hdm
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
from gtts import gTTS
import io
import pygame
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('hand_signals.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

X = data.drop('letter', axis=1)
y = data['letter']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)


def speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def main():
    pygame.mixer.init()

    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()

    letters = [0]
    word = ''
    words = []
    signal_data = {}

    start = time.time()
    end = time.time()

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img = cv2.resize(img, (640, 480))

            key = cv2.waitKey(1) & 0xFF

            img = detector.find_hands(img, draw=False)
            landmarks = detector.find_position(img)

            confidence_threshold = 0.7

           
            if not landmarks:
                start = time.time()
                idle_timer = start - end

                if idle_timer >= 3 and word != '' and word[-1] != ' ':
                    speech(word)
                    words.append(word)
                    word += ' '

         
            if landmarks and len(landmarks) == 1:
                lmlist = landmarks[0][1]
                end = time.time()

                p1 = (
                    min(lm[1] for lm in lmlist) - 25,
                    min(lm[2] for lm in lmlist) - 25
                )
                p2 = (
                    max(lm[1] for lm in lmlist) + 25,
                    max(lm[2] for lm in lmlist) + 25
                )

                cv2.rectangle(img, p1, p2, (255, 255, 255), 2)

                location_vector = np.array(
                    [coord for lm in lmlist for coord in lm[1:3]]
                ).reshape(1, -1)

                probabilities = model.predict_proba(location_vector)
                max_prob = np.max(probabilities)

                if max_prob > confidence_threshold:
                    predicted_letter = model.predict(location_vector)[0]

                    if predicted_letter == letters[-1]:
                        letters.append(predicted_letter)
                    else:
                        letters = [predicted_letter]

                    cv2.putText(
                        img,
                        predicted_letter,
                        (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        2
                    )

                if len(letters) == 20:
                    word += letters[0]
                    letters = [0]
                    print("WORD:", word)

            cv2.imshow("Sign Language Interpreter", img)

            
            if key == ord('c') and landmarks:
                for item in lmlist:
                    signal_data.setdefault(f'{item[0]}x', []).append(item[1])
                    signal_data.setdefault(f'{item[0]}y', []).append(item[2])

           
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

        if signal_data:
            signal_data['letter'] = ['a'] * len(signal_data['0x'])
            new_signals = pd.DataFrame(signal_data)
            existing = pd.read_csv('hand_signals.csv')
            updated = pd.concat([existing, new_signals], ignore_index=True)
            updated.to_csv('hand_signals.csv', index=False)


if __name__ == '__main__':
    main()
