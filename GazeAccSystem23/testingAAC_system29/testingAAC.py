# -*- coding: utf-8 -*-
import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from threading import Thread
from sklearn.preprocessing import MinMaxScaler
from filterpy.kalman import KalmanFilter
import pyttsx3
import pyautogui
import json
import random
import winsound
from tkinter import Toplevel
from datetime import datetime

pyautogui.FAILSAFE = False

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'


class OptimizedGazeModel:
    def __init__(self, model_path='lstm_gaze_model.h5'):
        self.model_path = model_path
        self.model = self._load_simplified_model()
        self.scaler_t = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_xy = MinMaxScaler(feature_range=(0, 1))
        self.last_prediction = None

    def _load_simplified_model(self):
        """Handle layer mismatch by loading compatible architecture"""
        try:
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully with original architecture")
            return model
        except:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(10, 3)),
                tf.keras.layers.Dense(48, activation='relu'),
                tf.keras.layers.Dense(2, activation='sigmoid')
            ])
            try:
                model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
                print("Model weights loaded with layer skipping")
                return model
            except Exception as e:
                print(f"Failed to load weights: {e}")
                return model

    def predict(self, gaze_history):
        """Predict gaze coordinates from history"""
        if len(gaze_history) < 10:
            return np.array([0.5, 0.5])  # Return center if not enough data

        # Convert gaze history to numpy array
        data = np.array([[item[1], item[2], time.time() - item[0]] for item in gaze_history[-10:]])

        # Scale the data
        data_scaled = self.scaler_t.fit_transform(data)

        # Reshape for LSTM input (batch_size, timesteps, features)
        data_reshaped = data_scaled.reshape(1, 10, 3)

        # Make prediction
        prediction = self.model.predict(data_reshaped, verbose=0)[0]

        # Scale back to screen coordinates
        prediction = self.scaler_xy.fit_transform(prediction.reshape(-1, 2))[0]

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
        self.root.title("Enhanced AAC System v2.6")
        self.root.geometry("1200x800")

        # Initialize all state variables
        self.initialize_state_variables()

        # Initialize components
        self.model = OptimizedGazeModel()
        self.fuser = GazeFusion()
        self.calibrator = OnlineCalibrator()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # Setup UI
        self.setup_ui()
        self.setup_eyetracking()
        self.load_settings()

        # Communication board selection settings
        self.original_button_color = "#9E9E9E"  # Must match your button color
        self.dwell_threshold = 1.5  # seconds
        self.current_dwell_button = None
        self.dwell_start_time = None


    def initialize_state_variables(self):
        """Initialize all state variables in one place"""
        # Interaction modes
        self.dwell_time_enabled = True
        self.dwell_threshold = 1.5
        self.blink_detection_active = False
        self.snap_active = False

        # System states
        self.running = False
        self.game_active = False
        self.panic_alert_active = False
        self.calibration_mode = False
        self.keyboard_visible = False

        # Tracking variables
        self.current_cal_target = None
        self.calibration_points = []
        self.gaze_history = []
        self.current_dwell_button = None
        self.dwell_start = None
        self.current_selection = None
        self.selection_highlight = None

        # Game variables
        self.game_score = 0
        self.game_high_score = 0
        self.game_target = None
        self.game_targets = []
        self.game_target_duration = 3.0
        self.game_target_timer = None
        self.game_start_time = 0
        self.game_duration = 60  # 1 minute game duration
        self.game_target_timer = None  # Add this line

        # Blink detection
        self.EAR_THRESHOLD = 0.21
        self.BLINK_CONSEC_FRAMES = 2
        self.blink_counter = 0
        self.last_blink_time = 0
        self.double_blink_threshold = 0.4
        self.long_blink_threshold = 1.0
        self.blink_history = []

        # Initialize control buttons dictionary
        self.control_buttons = {}

    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top control panel
        control_frame = tk.Frame(self.main_frame, bg="#f0f0f0", padx=10, pady=5)
        control_frame.grid(row=0, column=1, sticky="ew", pady=10)

        # Communication board
        self.comm_frame = tk.Frame(self.main_frame, padx=20, pady=20)
        self.comm_frame.grid(row=1, column=1, sticky="nsew")

        # Gaze visualization
        self.canvas = tk.Canvas(self.main_frame, width=800, height=400, bg='white')
        self.canvas.grid(row=2, column=1, pady=20)
        self.gaze_viz = self.canvas.create_oval(0, 0, 20, 20, fill='red', state=tk.HIDDEN)

        # Text display
        self.text_display = tk.Text(self.main_frame, height=5, font=("Arial", 16), wrap=tk.WORD)
        self.text_display.grid(row=3, column=1, sticky="ew", padx=20)

        # Action buttons
        action_frame = tk.Frame(self.main_frame)
        action_frame.grid(row=4, column=1, pady=5)

        tk.Button(action_frame, text="Speak", command=self.speak_text,
                  bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Clear", command=self.clear_text,
                  bg="#F44336", fg="white").pack(side=tk.LEFT, padx=5)

        # QWERTY keyboard
        self.keyboard_frame = tk.Frame(self.main_frame)
        self.setup_qwerty_keyboard()

        # Status bar
        self.status_frame = tk.Frame(self.root, height=25, bg="#e0e0e0")
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(self.status_frame, text="Status: Ready", bg="#e0e0e0")
        self.status_label.pack(side=tk.LEFT)

        # Game score and timer
        self.score_frame = tk.Frame(self.status_frame, bg="#e0e0e0")
        self.score_frame.pack(side=tk.RIGHT, padx=10)
        self.score_label = tk.Label(self.score_frame, text="Score: 0", bg="#e0e0e0")
        self.score_label.pack(side=tk.LEFT)
        self.timer_label = tk.Label(self.score_frame, text="Time: 60s", bg="#e0e0e0")
        self.timer_label.pack(side=tk.LEFT, padx=10)

        # Blink mode indicators
        self.blink_mode_var = tk.StringVar(value="Single")
        for mode in ["Single", "Double", "Long"]:
            tk.Radiobutton(self.status_frame, text=mode, variable=self.blink_mode_var,
                           value=mode, bg="#e0e0e0").pack(side=tk.RIGHT, padx=5)

        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Control buttons with consistent naming
        controls = [
            ("start_tracking", "Start Tracking", "#4CAF50", self.start_tracking),
            ("calibrate", "Calibrate", "#2196F3", self.start_calibration),
            ("show_keyboard", "Show Keyboard", "#607D8B", self.toggle_keyboard),
            ("start_game", "Start Game", "#9C27B0", self.toggle_game_mode),
            ("panic", "Panic Button", "#D32F2F", self.trigger_panic),
            ("snap_mode", "Snap Mode: OFF", "#607D8B", self.toggle_snap_mode),
            ("blink_nav", "Blink Nav: OFF", "#009688", self.toggle_blink_navigation),
            ("settings", "Settings", "#795548", self.show_settings)
        ]

        for key, text, color, cmd in controls:
            btn = tk.Button(control_frame, text=text, bg=color, fg="white",
                            command=cmd, width=12, font=("Arial", 10))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self.control_buttons[key] = btn

        self.update_communication_board()

        # Shortcut keys
        self.root.bind('<F1>', lambda e: self.start_tracking())
        self.root.bind('<F2>', lambda e: self.toggle_keyboard())
        self.root.bind('<F3>', lambda e: self.toggle_game_mode())
        self.root.bind('<F4>', lambda e: self.trigger_panic())

    def setup_qwerty_keyboard(self):
        qwerty_layout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', ' ']
        ]

        for row in qwerty_layout:
            row_frame = tk.Frame(self.keyboard_frame)
            row_frame.pack()
            for key in row:
                btn = tk.Button(row_frame, text=key, width=3,
                                command=lambda k=key: self.select_phrase(k))
                btn.pack(side=tk.LEFT, padx=1, pady=1)

    def update_communication_board(self):
        for widget in self.comm_frame.winfo_children():
            widget.destroy()

        self.phrases = [
            "Hello", "Goodbye", "Yes", "No",
            "Hungry", "Thirsty", "Happy", "Sad",
            "Help", "Bathroom", "Pain", "Thank you",
            "More", "Done", "Home", "Medicine"
        ]

        for i, phrase in enumerate(self.phrases):
            btn = tk.Button(
                self.comm_frame, text=phrase,
                width=12, height=3, bg="#9E9E9E", fg="white",
                font=("Arial", 12, "bold"), command=lambda p=phrase: self.select_phrase(p)
            )
            btn.grid(row=i // 4, column=i % 4, padx=5, pady=5, sticky="nsew")

        for i in range(4):
            self.comm_frame.grid_columnconfigure(i, weight=1)
            self.comm_frame.grid_rowconfigure(i, weight=1)

    def setup_eyetracking(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        Thread(target=self.process_gaze, daemon=True).start()

    def process_gaze(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Enhanced blink detection
                left_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
                right_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                current_time = time.time()

                # Blink state detection
                if avg_ear < self.EAR_THRESHOLD:
                    self.blink_counter += 1
                    if self.blink_counter == self.BLINK_CONSEC_FRAMES:
                        self.last_blink_time = current_time
                        self.blink_history.append(current_time)
                else:
                    if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                        blink_duration = current_time - self.last_blink_time

                        # Determine blink type
                        if len(self.blink_history) > 1:
                            time_since_last_blink = current_time - self.blink_history[-2]
                            if time_since_last_blink < self.double_blink_threshold:
                                self.handle_blink("double")
                            elif blink_duration > self.long_blink_threshold:
                                self.handle_blink("long")
                            else:
                                self.handle_blink("single")
                        else:
                            if blink_duration > self.long_blink_threshold:
                                self.handle_blink("long")
                            else:
                                self.handle_blink("single")

                    self.blink_counter = 0

                # Gaze tracking
                raw_gaze = self.extract_gaze_coords(landmarks)

                if self.calibration_mode and self.current_cal_target:
                    self.calibrator.update(raw_gaze, self.current_cal_target)
                elif self.running or self.game_active:
                    self.gaze_history.append((time.time(), raw_gaze[0], raw_gaze[1]))
                    if len(self.gaze_history) > 30:
                        self.gaze_history.pop(0)

                    # In process_gaze() method:
                    if len(self.gaze_history) >= 10:
                        prediction = self.model.predict(self.gaze_history[-10:])
                        fused = self.fuser.update(prediction, raw_gaze)

                        # Convert to absolute screen coordinates
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()
                        screen_x = int(fused[0] * screen_width)
                        screen_y = int(fused[1] * screen_height)

                        # Move cursor (absolute screen coordinates)
                        pyautogui.moveTo(screen_x, screen_y)

                        # Convert to canvas coordinates (relative to canvas)
                        canvas_x = screen_x - self.canvas.winfo_rootx()
                        canvas_y = screen_y - self.canvas.winfo_rooty()

                        # Update visualization (ensure it stays within canvas bounds)
                        canvas_x = max(0, min(canvas_x, self.canvas.winfo_width()))
                        canvas_y = max(0, min(canvas_y, self.canvas.winfo_height()))

                        self.canvas.coords(self.gaze_viz,
                                           canvas_x - 10, canvas_y - 10,
                                           canvas_x + 10, canvas_y + 10)

                        # Handle selection based on current mode
                        if self.dwell_time_enabled:
                            self.handle_dwell_selection(canvas_x, canvas_y)
                        if self.game_active:
                            self.check_game_target(canvas_x, canvas_y)

            time.sleep(0.02)

    def eye_aspect_ratio(self, eye_landmarks):
        # Vertical distances
        v1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) -
                            np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
        v2 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) -
                            np.array([eye_landmarks[4].x, eye_landmarks[4].y]))

        # Horizontal distance
        h = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) -
                           np.array([eye_landmarks[3].x, eye_landmarks[3].y]))

        return (v1 + v2) / (2.0 * h)

    def handle_blink(self, blink_type):
        if not self.blink_detection_active:
            return

        self.blink_mode_var.set(blink_type.capitalize())

        if blink_type == "single":
            self.activate_selected_item()
        elif blink_type == "double":
            self.move_selection("next")
        elif blink_type == "long":
            self.move_selection("prev")

        winsound.Beep(800 if blink_type == "single" else 1000, 100)

    def handle_dwell_selection(self, x, y):
        if not self.dwell_time_enabled:
            return

        screen_x = self.canvas.winfo_rootx() + x
        screen_y = self.canvas.winfo_rooty() + y
        widget = self.root.winfo_containing(screen_x, screen_y)

        if isinstance(widget, tk.Button):
            if widget == self.current_dwell_button:
                dwell_time = time.time() - self.dwell_start
                if dwell_time >= self.dwell_threshold:
                    widget.invoke()
                    self.dwell_start = 0
                    self.current_dwell_button = None
            else:
                self.dwell_start = time.time()
                self.current_dwell_button = widget
        else:
            self.dwell_start = 0
            self.current_dwell_button = None

    def move_selection(self, direction):
        if not hasattr(self, 'phrases') or not self.phrases:
            return

        current_idx = self.phrases.index(self.current_selection) if self.current_selection else -1

        if direction == "next":
            new_idx = (current_idx + 1) % len(self.phrases)
        else:
            new_idx = (current_idx - 1) % len(self.phrases)

        self.current_selection = self.phrases[new_idx]
        self.highlight_selection()
        self.move_gaze_to_selection()

    def move_gaze_to_selection(self):
        if not self.current_selection:
            return

        for child in self.comm_frame.winfo_children():
            if isinstance(child, tk.Button) and child.cget("text") == self.current_selection:
                # Calculate center of the button
                x = child.winfo_rootx() + child.winfo_width() // 2
                y = child.winfo_rooty() + child.winfo_height() // 2

                # Move gaze visualization
                canvas_x = x - self.canvas.winfo_rootx()
                canvas_y = y - self.canvas.winfo_rooty()
                self.canvas.coords(self.gaze_viz, canvas_x - 10, canvas_y - 10, canvas_x + 10, canvas_y + 10)

                # Move actual cursor
                pyautogui.moveTo(x, y)
                break

    def activate_selected_item(self):
        if self.current_selection:
            self.select_phrase(self.current_selection)

    def extract_gaze_coords(self, landmarks):
        left_iris = [landmarks[i] for i in [474, 475, 476, 477]]
        right_iris = [landmarks[i] for i in [469, 470, 471, 472]]
        x_coords = [lm.x for lm in left_iris + right_iris]
        y_coords = [lm.y for lm in left_iris + right_iris]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def start_tracking(self):
        if not self.running:
            if not self.calibrator.is_calibrated() and not self.game_active:
                messagebox.showwarning("Calibration Needed", "Please calibrate first")
                return
            self.running = True
            self.control_buttons['start_tracking'].config(text="Stop Tracking", bg="#F44336")
            self.canvas.itemconfig(self.gaze_viz, state=tk.NORMAL)
            self.status_label.config(text="Status: Tracking Active", fg="green")
        else:
            self.running = False
            self.control_buttons['start_tracking'].config(text="Start Tracking", bg="#4CAF50")
            self.canvas.itemconfig(self.gaze_viz, state=tk.HIDDEN)
            self.status_label.config(text="Status: Ready", fg="black")

    def start_calibration(self):
        if self.running:
            self.running = False
            self.control_buttons['start_tracking'].config(text="Start Tracking", bg="#4CAF50")
        self.calibration_mode = True
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

    def toggle_keyboard(self):
        self.keyboard_visible = not self.keyboard_visible
        if self.keyboard_visible:
            self.keyboard_frame.grid(row=5, column=1, pady=5)
            self.control_buttons['show_keyboard'].config(text="Hide Keyboard")
        else:
            self.keyboard_frame.grid_forget()
            self.control_buttons['show_keyboard'].config(text="Show Keyboard")

    def toggle_game_mode(self):
        self.game_active = not self.game_active
        if self.game_active:
            self.start_game()
            self.control_buttons['start_game'].config(text="End Game", bg="#7B1FA2")
        else:
            self.end_game()
            self.control_buttons['start_game'].config(text="Start Game", bg="#9C27B0")

    def start_game(self):
        self.game_score = 0
        self.game_start_time = time.time()
        self.game_targets = []  # Reset targets
        self.game_target_timer = None  # Reset timer
        self.score_label.config(text=f"Score: {self.game_score}")
        self.create_game_targets(5)
        self.status_label.config(text="Game Mode: Active")
        self.update_game_timer()  # Start the timer

    def update_game_timer(self):
        if not self.game_active:
            return

        elapsed = time.time() - self.game_start_time
        remaining = max(0, self.game_duration - elapsed)
        self.timer_label.config(text=f"Time: {int(remaining)}s")

        if remaining <= 0:
            self.end_game()
        else:
            # Store the timer ID so we can cancel it later
            self.game_target_timer = self.root.after(1000, self.update_game_timer)

    def create_game_targets(self, count):
        self.canvas.delete("target")  # Clear only targets
        for _ in range(count):
            size = random.randint(30, 50)
            padding = size + 20  # Prevent overlapping targets
            x = random.randint(padding, self.canvas.winfo_width() - padding)
            y = random.randint(padding, self.canvas.winfo_height() - padding)

            # Ensure targets don't overlap
            attempts = 0
            while any((x - t['x']) ** 2 + (y - t['y']) ** 2 < (size + t['size']) ** 2
                      for t in self.game_targets) and attempts < 100:
                x = random.randint(padding, self.canvas.winfo_width() - padding)
                y = random.randint(padding, self.canvas.winfo_height() - padding)
                attempts += 1

            target_id = self.canvas.create_oval(
                x - size // 2, y - size // 2, x + size // 2, y + size // 2,
                fill='green', tags="target"
            )
            self.game_targets.append({
                'id': target_id,
                'x': x,
                'y': y,
                'size': size,
                'active': True
            })

    def check_game_target(self, x, y):
        """Check if gaze position hits any active target"""
        # Clear previous debug markers (remove in production)
        self.canvas.delete("debug")

        # Convert gaze canvas coords to screen coords
        gaze_screen_x = x + self.canvas.winfo_rootx()
        gaze_screen_y = y + self.canvas.winfo_rooty()

        # Debug visualization - red dot at gaze point
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                fill='red', tags="debug")

        # Debug visualization - blue dot at actual cursor
        cursor_x, cursor_y = pyautogui.position()
        canvas_cursor_x = cursor_x - self.canvas.winfo_rootx()
        canvas_cursor_y = cursor_y - self.canvas.winfo_rooty()
        self.canvas.create_oval(canvas_cursor_x - 3, canvas_cursor_y - 3,
                                canvas_cursor_x + 3, canvas_cursor_y + 3,
                                fill='blue', tags="debug")

        for target in self.game_targets:
            if not target['active']:
                continue

            # Convert target canvas coords to screen coords
            target_screen_x = target['x'] + self.canvas.winfo_rootx()
            target_screen_y = target['y'] + self.canvas.winfo_rooty()

            # Calculate distance between gaze and target center
            distance = ((gaze_screen_x - target_screen_x) ** 2 +
                        (gaze_screen_y - target_screen_y) ** 2) ** 0.5

            # Debug visualization - target hit radius (green circle)
            self.canvas.create_oval(
                target['x'] - target['size'] / 2, target['y'] - target['size'] / 2,
                target['x'] + target['size'] / 2, target['y'] + target['size'] / 2,
                outline='green', tags="debug")

            # Check if gaze is within target radius
            if distance < (target['size'] / 2):
                # Debug visualization - hit confirmation (yellow square)
                self.canvas.create_rectangle(
                    target['x'] - 10, target['y'] - 10,
                    target['x'] + 10, target['y'] + 10,
                    outline='yellow', width=3, tags="debug")

                # Mark target as hit
                target['active'] = False
                self.canvas.itemconfig(target['id'], fill='blue')

                # Update score
                self.game_score += 10
                self.score_label.config(text=f"Score: {self.game_score}")

                # Visual feedback
                self.show_hit_effect(target['x'], target['y'])

                # Audio feedback
                if self.game_score % 50 == 0:
                    winsound.Beep(1500, 300)  # Special sound for milestones
                else:
                    winsound.Beep(1000, 200)  # Normal hit sound

                # Remove target after delay
                self.root.after(200, lambda t=target: self.remove_target(t))
                break  # Only register one hit per frame

    def show_hit_effect(self, x, y):
        effect = self.canvas.create_oval(
            x - 25, y - 25, x + 25, y + 25,
            outline='yellow', width=3, tags="effect"
        )
        self.root.after(300, lambda: self.canvas.delete(effect))

    def remove_target(self, target):
        if target in self.game_targets:
            self.canvas.delete(target['id'])
            self.game_targets.remove(target)
            # Add new targets if we're below minimum
            if len([t for t in self.game_targets if t['active']]) < 3:
                self.create_game_targets(1)

    def show_congratulations(self):
        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-topmost', True)
        self.overlay.attributes('-transparentcolor', 'gray15')

        self.confetti_canvas = tk.Canvas(self.overlay, bg='gray15', highlightthickness=0)
        self.confetti_canvas.pack(fill=tk.BOTH, expand=True)

        self.confetti_canvas.create_rectangle(
            self.root.winfo_width() // 2 - 200, self.root.winfo_height() // 2 - 60,
            self.root.winfo_width() // 2 + 200, self.root.winfo_height() // 2 + 60,
            fill='gold', outline='black', width=3
        )
        self.confetti_canvas.create_text(
            self.root.winfo_width() // 2, self.root.winfo_height() // 2,
            text="NEW HIGH SCORE!", font=("Arial", 36, "bold"),
            fill="black"
        )

        self.create_confetti()
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        self.root.after(3000, self.cleanup_confetti)

    def create_confetti(self):
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink']
        for _ in range(100):
            x = random.randint(0, self.root.winfo_width())
            y = random.randint(0, self.root.winfo_height() // 2)
            size = random.randint(5, 15)
            color = random.choice(colors)

            if random.choice([True, False]):
                confetti = self.confetti_canvas.create_oval(x, y, x + size, y + size, fill=color,
                                                            outline='')
            else:
                confetti = self.confetti_canvas.create_rectangle(x, y, x + size, y + size, fill=color,
                                                                 outline='')

            self.animate_confetti(confetti)

    def animate_confetti(self, item):
        if not hasattr(self, 'confetti_canvas') or not self.confetti_canvas.winfo_exists():
            return

        self.confetti_canvas.move(item, random.randint(-5, 5), random.randint(1, 5))
        if self.confetti_canvas.coords(item)[1] < self.root.winfo_height():
            self.root.after(50, lambda: self.animate_confetti(item))
        else:
            self.confetti_canvas.delete(item)

    def cleanup_confetti(self):
        if hasattr(self, 'overlay') and self.overlay.winfo_exists():
            self.overlay.destroy()

    def end_game(self):
        self.game_active = False
        # Only cancel the timer if it exists
        if self.game_target_timer is not None:
            try:
                self.root.after_cancel(self.game_target_timer)
            except ValueError:
                pass  # Timer was already canceled or invalid
            self.game_target_timer = None

        # Calculate time bonus
        time_remaining = max(0, self.game_duration - (time.time() - self.game_start_time))
        time_bonus = int(time_remaining * 0.5)  # 0.5 points per second remaining
        self.game_score += time_bonus

        # Check high score
        if self.game_score > self.game_high_score:
            self.game_high_score = self.game_score
            self.show_congratulations()

        message = f"Final Score: {self.game_score}\n(Time Bonus: +{time_bonus})\nHigh Score: {self.game_high_score}"
        messagebox.showinfo("Game Over", message)
        self.status_label.config(text="Status: Ready")
        self.canvas.delete("all")
        self.control_buttons['start_game'].config(text="Start Game", bg="#9C27B0")

    def get_precise_gaze_position(self):
        """Returns (x,y) in screen coordinates or None"""
        if len(self.gaze_history) < 10:
            return None

        prediction = self.model.predict(self.gaze_history[-10:])
        if self.calibrator.is_calibrated():
            calibrated = self.calibrator.matrix @ np.array([prediction[0], prediction[1], 1])
            prediction = (calibrated[0] / calibrated[2], calibrated[1] / calibrated[2])

        return (
            int(prediction[0] * self.root.winfo_screenwidth()),
            int(prediction[1] * self.root.winfo_screenheight())
        )

    def handle_communication_selection(self):
        gaze_pos = self.get_precise_gaze_position()
        if not gaze_pos:
            return

        widget = self.root.winfo_containing(gaze_pos[0], gaze_pos[1])

        if widget and widget in self.comm_frame.winfo_children() and isinstance(widget, tk.Button):
            if widget != self.current_dwell_button:
                self.reset_dwell_timer()
                self.current_dwell_button = widget
                self.dwell_start_time = time.time()
                widget.config(bg='yellow')

            dwell_time = time.time() - self.dwell_start_time
            if dwell_time >= self.dwell_threshold:
                self.select_phrase(widget.cget("text"))
                widget.config(bg=self.original_button_color)
                self.reset_dwell_timer()
        else:
            self.reset_dwell_timer()

    def reset_dwell_timer(self):
        if hasattr(self, 'current_dwell_button') and self.current_dwell_button:
            self.current_dwell_button.config(bg=self.original_button_color)
        self.current_dwell_button = None
        self.dwell_start_time = None

    def select_phrase(self, phrase):
        current_text = self.text_display.get("1.0", tk.END).strip()
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", f"{current_text} {phrase}".strip())
        self.status_label.config(text=f"Selected: {phrase}", fg="blue")
        self.root.after(1000, lambda: self.status_label.config(
            text="Status: Tracking Active" if self.running else "Status: Ready",
            fg="green" if self.running else "black"))

    def speak_text(self):
        text = self.text_display.get("1.0", tk.END).strip()
        if text:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def clear_text(self):
        self.text_display.delete("1.0", tk.END)

    def toggle_snap_mode(self):
        self.snap_active = not self.snap_active
        if self.snap_active:
            self.control_buttons['snap_mode'].config(text="Snap Mode: ON", bg="#8BC34A")
            self.status_label.config(text="Snap Mode: Active", fg="purple")
        else:
            self.control_buttons['snap_mode'].config(text="Snap Mode: OFF", bg="#607D8B")
            self.status_label.config(text="Status: Ready", fg="black")

    def toggle_blink_navigation(self):
        self.blink_detection_active = not self.blink_detection_active
        if self.blink_detection_active:
            self.control_buttons['blink_nav'].config(text="Blink Nav: ON", bg="#4CAF50")
            self.status_label.config(text="Blink Navigation: Active", fg="blue")
        else:
            self.control_buttons['blink_nav'].config(text="Blink Nav: OFF", bg="#009688")
            self.status_label.config(text="Status: Ready", fg="black")

    def trigger_panic(self):
        if not self.panic_alert_active:
            self.panic_alert_active = True
            self.control_buttons['panic'].config(text="ALERTING!", bg="#FF5722")

            overlay = tk.Canvas(self.root, bg='red', highlightthickness=0)
            overlay.place(x=0, y=0, relwidth=1, relheight=1)
            overlay.create_text(self.root.winfo_width() // 2, self.root.winfo_height() // 2,
                                text="ALERT! HELP IS COMING!", font=("Arial", 48, "bold"),
                                fill="white")

            Thread(target=lambda: [winsound.Beep(1000, 500) for _ in range(3)]).start()

            with open("panic_log.txt", "a") as f:
                f.write(f"Panic alert at {datetime.now()}\n")

            messagebox.showwarning("ALERT", "Help is on the way!")
            self.root.after(5000, lambda: [overlay.destroy(), self.reset_panic_alert()])

    def reset_panic_alert(self):
        self.panic_alert_active = False
        self.control_buttons['panic'].config(text="Panic Button", bg="#D32F2F")

    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")

        # Dwell Time Setting
        tk.Label(settings_window, text="Dwell Time Selection:").pack(pady=5)
        dwell_var = tk.BooleanVar(value=self.dwell_time_enabled)
        tk.Checkbutton(settings_window, text="Enable Dwell Time Selection",
                       variable=dwell_var, command=lambda: self.toggle_dwell_time(dwell_var.get())).pack()

        tk.Label(settings_window, text="Dwell Time (seconds):").pack(pady=5)
        dwell_scale = tk.Scale(settings_window, from_=0.5, to=5, resolution=0.1,
                               orient=tk.HORIZONTAL, command=self.update_dwell_time)
        dwell_scale.set(self.dwell_threshold)
        dwell_scale.pack()

        # Blink Threshold Setting
        tk.Label(settings_window, text="Blink Sensitivity:").pack(pady=5)
        blink_scale = tk.Scale(settings_window, from_=0.1, to=0.3, resolution=0.01,
                               orient=tk.HORIZONTAL, command=self.update_blink_threshold)
        blink_scale.set(self.EAR_THRESHOLD)
        blink_scale.pack()

        # Game Duration Setting
        tk.Label(settings_window, text="Game Duration (seconds):").pack(pady=5)
        game_scale = tk.Scale(settings_window, from_=30, to=180, resolution=5,
                              orient=tk.HORIZONTAL, command=self.update_game_duration)
        game_scale.set(self.game_duration)
        game_scale.pack()

        # Save Button
        tk.Button(settings_window, text="Save Settings", command=lambda: [
            self.save_settings(),
            settings_window.destroy()
        ]).pack(pady=10)

    def toggle_dwell_time(self, enabled):
        self.dwell_time_enabled = enabled
        if not enabled:
            self.dwell_start = 0
            self.current_dwell_button = None

    def update_dwell_time(self, value):
        self.dwell_threshold = float(value)

    def update_blink_threshold(self, value):
        self.EAR_THRESHOLD = float(value)

    def update_game_duration(self, value):
        self.game_duration = int(value)

    def load_settings(self):
        try:
            with open('aac_settings.json', 'r') as f:
                settings = json.load(f)

            self.dwell_threshold = settings.get('dwell_threshold', 1.5)
            self.dwell_time_enabled = settings.get('dwell_time_enabled', True)
            self.snap_threshold = settings.get('snap_threshold', 2.0)
            self.snap_duration = settings.get('snap_duration', 3.0)
            self.EAR_THRESHOLD = settings.get('ear_threshold', 0.21)
            self.game_duration = settings.get('game_duration', 60)

            if settings.get('blink_detection_active', False):
                self.toggle_blink_navigation()

        except FileNotFoundError:
            pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")

    def save_settings(self):
        settings = {
            'dwell_threshold': self.dwell_threshold,
            'dwell_time_enabled': self.dwell_time_enabled,
            'snap_threshold': self.snap_threshold,
            'snap_duration': self.snap_duration,
            'ear_threshold': self.EAR_THRESHOLD,
            'blink_detection_active': self.blink_detection_active,
            'game_duration': self.game_duration
        }

        try:
            with open('aac_settings.json', 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AACSystem(root)
    root.mainloop()