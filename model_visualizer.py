
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np


def load_model_safely(model_path):
    """Special loader that handles batch_shape issues"""
    try:
        # Try standard load first
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Standard load failed: {e}")
        print("Attempting reconstruction...")

        # Reconstruct model architecture from inspection
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(10, 3)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

        # Load weights only
        model.load_weights(model_path, by_name=True)
        return model


def visualize_model(model_path):
    try:
        # Add Graphviz to PATH if needed
        if os.name == 'nt':  # Windows
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

        # Load model with workaround
        model = load_model_safely(model_path)

        # Print text summary
        print("\n=== Model Summary ===")
        model.summary(show_trainable=True, line_length=120)

        # Generate visualization
        img_path = os.path.splitext(model_path)[0] + "_architecture.png"
        print("\nGenerating visualization...")

        plot_model(
            model,
            to_file=img_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # Top-to-bottom layout
            expand_nested=True,
            dpi=96
        )

        # Display
        img = plt.imread(img_path)
        plt.figure(figsize=(12, 18))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Model Architecture', pad=20)
        plt.show()

        print(f"\nVisualization saved to: {img_path}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Install requirements: pip install tensorflow matplotlib pydot graphviz")
        print("2. Download and install Graphviz from https://graphviz.org/download/")
        print("3. Add Graphviz to your system PATH")
        print("4. If issues persist, try saving your model with:")
        print("   model.save('new_model.keras', save_format='keras')")


if __name__ == "__main__":
    model_path = r'C:\Users\edugh\PycharmProjects\GazeACCSystem\GazeAccSystem23\04\lstm_gaze_model.h5'
    visualize_model(model_path)