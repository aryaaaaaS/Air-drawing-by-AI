# --- Keep all imports unchanged ---
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime
import math

class AirDrawingSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.brush_thickness = 15
        self.eraser_thickness = 50
        self.drawColor = (0, 0, 0)
        self.eraser_color = (255, 255, 255)

        self.img_canvas = None
        self.xp, self.yp = 0, 0
        self.eraser_mode = False
        self.data_collection_mode = False
        self.current_shape_class = "circle"

        self.shape_classes = []
        self.model = None
        self.load_model()

        self.emoji_images = self.load_emojis()

        self.prediction_text = ""
        self.prediction_confidence = 0.0
        self.show_prediction = False
        self.prediction_timer = 0

        print("Air Drawing System Initialized!")
        print(f"Loaded shape classes: {self.shape_classes}")  # DEBUG

    def load_model(self):
        model_path = "models/drawing_model.h5"
        label_path = "models/drawing_model_label_encoder.npy"
        if os.path.exists(model_path) and os.path.exists(label_path):
            self.model = keras.models.load_model(model_path)
            self.shape_classes = list(np.load(label_path))
            print(f"Model & labels loaded: {self.shape_classes}")
        else:
            print("Model or label file not found. Please run model_trainer.py first.")

    def load_emojis(self):
        emoji_dict = {}
        if not os.path.exists("emojis"):
            os.makedirs("emojis")
            return emoji_dict
        for shape in self.shape_classes:
            path = os.path.join("emojis", f"{shape}.png")
            if os.path.exists(path):
                emoji_dict[shape] = cv2.imread(path)
        return emoji_dict

    def fingers_up(self, lm_list):
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        fingers.append(int(lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]))
        for id in range(1, 5):
            fingers.append(int(lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]))
        return fingers

    def get_hand_landmarks(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        lm_list = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([cx, cy])
                self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return lm_list

    def preprocess_canvas_for_prediction(self):
        if self.img_canvas is None:
            return None
        gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(thresh.shape[1], x_max + padding)
        y_max = min(thresh.shape[0], y_max + padding)
        cropped = thresh[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None
        resized = cv2.resize(cropped, (64, 64))
        norm = resized.astype('float32') / 255.0
        return norm.reshape(1, 64, 64, 1)

    def predict_drawing(self):
        if self.model is None or not self.shape_classes:
            print("Model not loaded. Cannot make predictions.")
            return
        print(f"Predicting with shape classes: {self.shape_classes}")  # DEBUG
        processed_img = self.preprocess_canvas_for_prediction()
        if processed_img is None:
            print("No drawing found to predict.")
            return
        preds = self.model.predict(processed_img, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        if confidence > 0.5:
            label = self.shape_classes[idx]
            self.prediction_text = f"{label} ({confidence:.2f})"
            self.prediction_confidence = confidence
            self.show_prediction = True
            self.prediction_timer = 120
            print(f"Prediction: {label} | Confidence: {confidence:.2f}")
        else:
            self.prediction_text = "Uncertain"
            self.show_prediction = True
            self.prediction_timer = 60
            print("Prediction uncertain")

    def save_drawing(self):
        if self.img_canvas is not None:
            os.makedirs("saved_drawings", exist_ok=True)
            fname = f"saved_drawings/drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, self.img_canvas)
            print(f"Drawing saved: {fname}")

    def save_for_dataset(self, shape_class):
        os.makedirs(f"dataset/{shape_class}", exist_ok=True)
        count = len(os.listdir(f"dataset/{shape_class}"))
        processed = self.preprocess_canvas_for_prediction()
        if processed is not None:
            img = (processed[0] * 255).astype(np.uint8)
            cv2.imwrite(f"dataset/{shape_class}/{shape_class}_{count+1:03d}.png", img)
            print(f"Saved to dataset: {shape_class}_{count+1:03d}.png")

    def clear_canvas(self):
        self.img_canvas = None
        self.xp, self.yp = 0, 0
        self.show_prediction = False
        print("Canvas cleared")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            if self.img_canvas is None:
                self.img_canvas = np.ones((h, w, 3), np.uint8) * 255

            lm_list = self.get_hand_landmarks(img)
            if lm_list:
                x1, y1 = lm_list[8][:2]
                fingers = self.fingers_up(lm_list)

                if fingers[1] and fingers[2]:  # Two fingers = eraser
                    cv2.circle(img, (x1, y1), self.eraser_thickness, (0, 255, 0), cv2.FILLED)
                    cv2.circle(self.img_canvas, (x1, y1), self.eraser_thickness, self.eraser_color, cv2.FILLED)
                elif fingers[1]:  # Drawing mode
                    cv2.circle(img, (x1, y1), 15, self.drawColor, cv2.FILLED)
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1
                    cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1),
                             self.eraser_color if self.eraser_mode else self.drawColor,
                             self.eraser_thickness if self.eraser_mode else self.brush_thickness)
                    self.xp, self.yp = x1, y1
                else:
                    self.xp, self.yp = 0, 0
            else:
                self.xp, self.yp = 0, 0

            alpha = 0.2  # Lower opacity for clearer camera
            img = cv2.addWeighted(img, 1 - alpha, self.img_canvas, alpha, 0)

            mode = "ERASER" if self.eraser_mode else "DRAW"
            cv2.putText(img, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if self.data_collection_mode:
                cv2.putText(img, f"Data Collection: {self.current_shape_class}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.show_prediction and self.prediction_timer > 0:
                cv2.putText(img, f"Prediction: {self.prediction_text}", (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                shape = self.prediction_text.split(' ')[0].lower()
                if shape in self.emoji_images:
                    emoji = cv2.resize(self.emoji_images[shape], (100, 100))
                    img[h-150:h-50, w-120:w-20] = emoji
                self.prediction_timer -= 1

            cv2.putText(img, "P:Predict C:Clear S:Save E:Eraser D:Dataset Q:Quit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Air Drawing System", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): self.clear_canvas()
            elif key == ord('s'):
                if self.data_collection_mode:
                    self.save_for_dataset(self.current_shape_class)
                else:
                    self.save_drawing()
            elif key == ord('p'): self.predict_drawing()
            elif key == ord('e'):
                self.eraser_mode = not self.eraser_mode
                print(f"Eraser mode: {'ON' if self.eraser_mode else 'OFF'}")
            elif key == ord('d'):
                self.data_collection_mode = not self.data_collection_mode
                print(f"Data Collection: {'ON' if self.data_collection_mode else 'OFF'}")
            elif self.data_collection_mode and key in range(ord('1'), ord('9')):
                idx = key - ord('1')
                if idx < len(self.shape_classes):
                    self.current_shape_class = self.shape_classes[idx]
                    print(f"Class selected: {self.current_shape_class}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirDrawingSystem()
    app.run()
