# -*- coding: utf-8 -*-
import os
import time
import tkinter as tk
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from tkinter import filedialog, messagebox
from threading import Thread
from sklearn.preprocessing import MinMaxScaler
from filterpy.kalman import KalmanFilter
from tensorflow_model_optimization.sparsity import keras as sparsity
import pyttsx3
import pyautogui

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'


class OptimizedGazeModel:
    def __init__(self, model_path='lstm_gaze_model.h5'):
        self.model_path = model_path
        self.model = self._load_custom_model()
        self.scaler_t = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_xy = MinMaxScaler(feature_range=(0, 1))
        self.last_prediction = None

    def _load_custom_model(self):
        """Load model with architecture that exactly matches the saved weights"""
        try:
            # Updated model architecture to match expected weights
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(10, 3)),

                # First Bidirectional LSTM - 128 units per direction (256 total)
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, return_sequences=True),
                    name='bidirectional_2'
                ),
                tf.keras.layers.Dropout(0.2, name='dropout_3'),
                tf.keras.layers.LayerNormalization(name='layer_normalization_2'),

                # Second Bidirectional LSTM - 64 units per direction (128 total)
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(64),
                    name='bidirectional_3'
                ),
                tf.keras.layers.Dropout(0.2, name='dropout_4'),
                tf.keras.layers.LayerNormalization(name='layer_normalization_3'),

                # Dense layers
                tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
                tf.keras.layers.Dropout(0.2, name='dropout_5'),
                tf.keras.layers.Dense(2, activation='sigmoid', name='dense_3')
            ])

            # Load weights with exact name matching
            model.load_weights(self.model_path, by_name=True)

            # Verify all weights loaded correctly
            for layer in model.layers:
                if not layer.weights:
                    continue
                for weight in layer.weights:
                    if not weight.shape.is_fully_defined():
                        raise ValueError(f"Layer {layer.name} has undefined weights")

            return self._prepare_model_for_quantization(model)

        except Exception as e:
            print(f"Custom model loading failed: {e}")
            print("Falling back to simple model...")
            return self._build_fallback_model()

    def _prepare_model_for_quantization(self, model):
        """Prepares the loaded model for quantization"""
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.30,
                final_sparsity=0.70,
                begin_step=0,
                end_step=1000
            )
        }
        pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        return converter.convert()

    def _build_fallback_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(10, 3)),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        return model

    def preprocess(self, sequence):
        timestamps = np.array([s[0] for s in sequence]).reshape(-1, 1)
        x_coords = np.array([s[1] for s in sequence]).reshape(-1, 1)
        y_coords = np.array([s[2] for s in sequence]).reshape(-1, 1)

        if not hasattr(self.scaler_t, 'n_samples_seen_'):
            self.scaler_t.fit(timestamps)
        if not hasattr(self.scaler_xy, 'n_samples_seen_'):
            self.scaler_xy.fit(x_coords)
            self.scaler_xy.fit(y_coords)

        timestamps = self.scaler_t.transform(timestamps)
        x_coords = self.scaler_xy.transform(x_coords)
        y_coords = self.scaler_xy.transform(y_coords)

        return np.hstack([timestamps, x_coords, y_coords]).astype(np.float32)

    def predict(self, sequence):
        sequence = self.preprocess(sequence)
        if isinstance(self.model, bytes):  # TFLite model
            interpreter = tf.lite.Interpreter(model_content=self.model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], sequence[np.newaxis, ...])
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            prediction = self.model.predict(sequence[np.newaxis, ...])[0]

        if self.last_prediction is not None:
            prediction = 0.7 * prediction + 0.3 * self.last_prediction
        self.last_prediction = prediction
        return prediction


class GazeFusion:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self._configure()

    def _configure(self):
        self.kf.F = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.P *= 10
        self.kf.R *= 0.01
        self.kf.Q *= 0.1

    def update(self, prediction, measurement):
        self.kf.predict()
        self.kf.update(np.array([measurement[0], measurement[1]]))
        return self.kf.x[:2]


class OnlineCalibrator:
    def __init__(self):
        self.matrix = np.eye(3)
        self.learning_rate = 0.1
        self.error_history = []
        self.calibration_data = []

    def update(self, raw_point, target_point):
        raw_vec = np.array([raw_point[0], raw_point[1], 1])
        target_vec = np.array([target_point[0], target_point[1], 1])
        prediction = self.matrix @ raw_vec
        error = target_vec - prediction

        self.calibration_data.append((raw_point, target_point))
        gain = self.learning_rate / (1 + np.linalg.norm(raw_vec) ** 2)
        update_term = gain * np.outer(error[:2], raw_vec)

        if update_term.shape == self.matrix.shape:
            self.matrix += update_term
        else:
            self.matrix[:2, :] += update_term

        current_error = np.linalg.norm(error[:2])
        self.error_history.append(current_error)

        if len(self.error_history) > 50:
            trend = np.polyfit(range(50), self.error_history[-50:], 1)[0]
            self.learning_rate = np.clip(self.learning_rate * (1 - trend / 10), 0.01, 0.5)

        return prediction[:2]

    def is_calibrated(self):
        if len(self.error_history) < 9:
            return False
        return np.mean(self.error_history[-9:]) < 0.05


class AACSystem:
    def __init__(self, root):
        self.root = root
        self.model = OptimizedGazeModel()
        self.fuser = GazeFusion()
        self.calibrator = OnlineCalibrator()
        self.tts_engine = pyttsx3.init()
        self.setup_ui()
        self.setup_eyetracking()

        self.dwell_start = None
        self.current_dwell_button = None
        self.dwell_threshold = 1.5
        self.gaze_history = []
        self.smoothed_gaze = None
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0

    def setup_ui(self):
        self.root.title("LSTM Gaze AAC System")
        self.root.geometry("1200x800")
        self.root.config(bg="#f0f0f0")

        # Status bar
        self.status_frame = tk.Frame(self.root, bg="#e0e0e0", height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(
            self.status_frame,
            text="Status: Ready",
            bg="#e0e0e0",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Gaze tracking canvas
        self.canvas = tk.Canvas(self.root, width=1000, height=500, bg='white', bd=2, relief='ridge')
        self.canvas.pack(pady=10)
        self.canvas_x_offset = self.canvas.winfo_x()
        self.canvas_y_offset = self.canvas.winfo_y()

        # Control buttons
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=5)

        self.start_btn = tk.Button(
            control_frame,
            text="Start Tracking",
            command=self.start_tracking,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.calibrate_btn = tk.Button(
            control_frame,
            text="Calibrate",
            command=self.start_calibration,
            width=15,
            height=2,
            bg="#2196F3",
            fg="white"
        )
        self.speak_btn = tk.Button(
            control_frame,
            text="Speak",
            command=self.speak_text,
            width=15,
            height=2,
            bg="#FF9800",
            fg="white"
        )
        self.clear_btn = tk.Button(
            control_frame,
            text="Clear",
            command=self.clear_text,
            width=15,
            height=2,
            bg="#F44336",
            fg="white"
        )

        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        self.speak_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Communication board
        self.comm_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.comm_frame.pack(pady=10)

        phrases = [
            "Hello", "Goodbye", "Yes", "No",
            "Hungry", "Thirsty", "Happy", "Sad",
            "Help", "Bathroom", "Pain", "Thank you",
            "More", "Done", "Home", "Medicine"
        ]

        for i, phrase in enumerate(phrases):
            btn = tk.Button(
                self.comm_frame,
                text=phrase,
                width=12,
                height=3,
                bg="#9E9E9E",
                fg="white",
                font=("Arial", 12),
                command=lambda p=phrase: self.select_phrase(p)
            )
            btn.grid(row=i // 4, column=i % 4, padx=5, pady=5)

        # Text display
        self.text_display = tk.Text(
            self.root,
            height=4,
            width=80,
            font=("Arial", 14),
            wrap=tk.WORD
        )
        self.text_display.pack(pady=10)

        # Gaze cursor
        self.gaze_viz = self.canvas.create_oval(0, 0, 20, 20, fill='red', state=tk.HIDDEN)

    def setup_eyetracking(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.calibration_mode = False
        self.current_cal_target = None
        self.gaze_history = []
        Thread(target=self.process_gaze, daemon=True).start()

    def start_tracking(self):
        if not self.running:
            if not self.calibrator.is_calibrated():
                messagebox.showwarning("Calibration Needed", "Please calibrate first")
                return
            self.running = True
            self.start_btn.config(text="Stop Tracking", bg="#F44336")
            self.canvas.itemconfig(self.gaze_viz, state=tk.NORMAL)
            self.status_label.config(text="Status: Tracking Active", fg="green")
        else:
            self.running = False
            self.start_btn.config(text="Start Tracking", bg="#4CAF50")
            self.canvas.itemconfig(self.gaze_viz, state=tk.HIDDEN)
            self.status_label.config(text="Status: Ready", fg="black")

    def start_calibration(self):
        if self.running:
            self.running = False
            self.start_btn.config(text="Start Tracking", bg="#4CAF50")
        self.calibration_mode = True
        self.calibrator.error_history = []
        self.calibrator.calibration_data = []
        self.calibration_points = []
        self.status_label.config(text="Status: Calibration Mode", fg="blue")
        self.show_calibration_target()

    def show_calibration_target(self):
        if not self.calibration_mode:
            return

        cal_positions = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]

        if len(self.calibration_points) < len(cal_positions):
            self.current_cal_target = cal_positions[len(self.calibration_points)]
            self.calibration_points.append(self.current_cal_target)

            self.canvas.delete("all")
            x = int(self.current_cal_target[0] * self.canvas.winfo_width())
            y = int(self.current_cal_target[1] * self.canvas.winfo_height())
            self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill='green', tags="target")

            self.root.after(2000, self.show_calibration_target)
        else:
            self.calibration_mode = False
            if self.calibrator.is_calibrated():
                self.status_label.config(text="Status: Calibration Complete", fg="green")
            else:
                self.status_label.config(text="Status: Calibration Failed", fg="red")

    def process_gaze(self):
        while True:
            if not self.running and not self.calibration_mode:
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                raw_gaze = self.extract_gaze_coords(landmarks)

                if self.calibration_mode and self.current_cal_target:
                    self.calibrator.update(raw_gaze, self.current_cal_target)
                elif self.running:
                    self.gaze_history.append((time.time(), raw_gaze[0], raw_gaze[1]))
                    if len(self.gaze_history) > 30:
                        self.gaze_history.pop(0)

                    if len(self.gaze_history) >= 10:
                        sequence = self.gaze_history[-10:]
                        prediction = self.model.predict(sequence)
                        fused = self.fuser.update(prediction, raw_gaze)

                        # Convert to screen coordinates
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()

                        # Get actual screen coordinates (not just canvas)
                        screen_x = int(fused[0] * screen_width)
                        screen_y = int(fused[1] * screen_height)

                        # Move system cursor
                        pyautogui.moveTo(screen_x, screen_y)

                        # Update visualization cursor (relative to canvas)
                        canvas_x = int(fused[0] * self.canvas.winfo_width())
                        canvas_y = int(fused[1] * self.canvas.winfo_height())
                        self.canvas.coords(
                            self.gaze_viz,
                            canvas_x - 10, canvas_y - 10,
                            canvas_x + 10, canvas_y + 10
                        )

                        # Check button selection
                        self.check_button_selection(canvas_x, canvas_y)

            time.sleep(0.02)

    def extract_gaze_coords(self, landmarks):
        left_iris = [landmarks.landmark[i] for i in [474, 475, 476, 477]]
        right_iris = [landmarks.landmark[i] for i in [469, 470, 471, 472]]
        x_coords = [lm.x for lm in left_iris + right_iris]
        y_coords = [lm.y for lm in left_iris + right_iris]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def check_button_selection(self, x, y):
        # Convert canvas coordinates to screen coordinates
        screen_x = self.canvas.winfo_rootx() + x
        screen_y = self.canvas.winfo_rooty() + y

        widget = self.root.winfo_containing(screen_x, screen_y)

        if isinstance(widget, tk.Button):
            if widget == self.current_dwell_button:
                if time.time() - self.dwell_start >= self.dwell_threshold:
                    widget.invoke()
                    self.dwell_start = None
                    self.current_dwell_button = None
            else:
                self.dwell_start = time.time()
                self.current_dwell_button = widget
        else:
            self.dwell_start = None
            self.current_dwell_button = None

    def select_phrase(self, phrase):
        current_text = self.text_display.get("1.0", tk.END).strip()
        new_text = f"{current_text} {phrase}".strip()
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", new_text)
        self.status_label.config(text=f"Selected: {phrase}", fg="blue")
        self.root.after(1000, lambda: self.status_label.config(
            text="Status: Tracking Active",
            fg="green") if self.running else None
                        )

    def speak_text(self):
        text = self.text_display.get("1.0", tk.END).strip()
        if text:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def clear_text(self):
        self.text_display.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = AACSystem(root)
    root.mainloop()