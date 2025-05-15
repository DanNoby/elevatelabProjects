import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Current information
CURRENT_USER = "DanNoby"
CURRENT_DATETIME = "2025-05-15 11:42:39"

# Display basic information
print(f"Sign Language Detection - Model Training and Prediction")
print(f"User: {CURRENT_USER}")
print(f"Date/Time: {CURRENT_DATETIME}")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Constants
IMG_SIZE = 64
LABELS = ['0', '1', '2', '3', '4', '5']
NUM_CLASSES = len(LABELS)
DATA_DIR = "sign_language_data"
MODEL_PATH = "sign_language_model.h5"

# Function to create a CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Function to load and preprocess data
def load_data():
    images = []
    labels = []
    
    print("Loading training data...")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please run collect_data.py first to collect training data.")
        return None, None
    
    for i, label_name in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(label_dir):
            print(f"Warning: Directory for label '{label_name}' not found. Skipping.")
            continue
            
        print(f"Loading images for digit {label_name}...")
        for img_name in os.listdir(label_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(label_dir, img_name)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                    
                # Resize to standard size
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalize pixel values
                img = img / 255.0
                
                # Add to dataset
                images.append(img)
                labels.append(i)  # Use integer index as label
    
    if not images:
        print("Error: No valid images found in the data directory.")
        return None, None
        
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {len(images)} images across {NUM_CLASSES} classes.")
    
    # One-hot encode the labels
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    
    return X, y

# Function to train the model
def train_model():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return None
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create and compile model
    model = create_model()
    print(model.summary())
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")
    
    return model

# Real-time prediction function
def predict_signs():
    # Check if model exists, if not, train it
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Training new model...")
        model = train_model()
        if model is None:
            print("Failed to train model. Exiting.")
            return
    else:
        # Load existing model
        print(f"Loading existing model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Variables for prediction smoothing
    prev_prediction = None
    pred_history = []
    max_history = 5
    
    print("\n=== REAL-TIME PREDICTION MODE ===")
    print("Show hand gestures for digits 0-5.")
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
            
        # Flip the frame for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = hands.process(rgb_frame)
        
        # Draw user info on frame
        cv2.putText(frame, f"User: {CURRENT_USER}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, CURRENT_DATETIME, (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get bounding box coordinates
                h, w, c = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add margin
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract hand region
                hand_img = frame[y_min:y_max, x_min:x_max]
                
                # Make sure we have a valid image
                if hand_img.size != 0 and hand_img.shape[0] > 10 and hand_img.shape[1] > 10:
                    # Resize to standard size
                    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize pixel values
                    hand_img = hand_img / 255.0
                    
                    # Add batch dimension
                    hand_img = np.expand_dims(hand_img, axis=0)
                    
                    # Predict
                    preds = model.predict(hand_img, verbose=0)
                    class_idx = np.argmax(preds[0])
                    confidence = preds[0][class_idx]
                    
                    # Apply smoothing using historical predictions
                    pred_history.append((class_idx, confidence))
                    if len(pred_history) > max_history:
                        pred_history.pop(0)
                    
                    # Count class occurrences in history
                    class_counts = {}
                    for idx, conf in pred_history:
                        if idx not in class_counts:
                            class_counts[idx] = 0
                        class_counts[idx] += 1
                    
                    # Get the most frequent class and its confidence
                    final_class_idx = max(class_counts, key=class_counts.get)
                    final_confidence = sum([conf for idx, conf in pred_history if idx == final_class_idx]) / class_counts[final_class_idx]
                    
                    # Determine color based on confidence
                    if final_confidence > 0.9:
                        color = (0, 255, 0)  # Green for high confidence
                    elif final_confidence > 0.7:
                        color = (0, 165, 255)  # Orange for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Display prediction with confidence
                    if final_confidence > 0.5:  # Only show if confidence is high enough
                        prediction_text = f"Digit: {LABELS[final_class_idx]}"
                        confidence_text = f"Confidence: {final_confidence*100:.1f}%"
                        
                        # Show prediction in larger font at the center of the screen
                        cv2.putText(frame, prediction_text, (w//2 - 100, h - 50), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                        cv2.putText(frame, confidence_text, (w//2 - 100, h - 20), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show the frame
        cv2.imshow("Sign Language Recognition", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    print("\n=== SIGN LANGUAGE DETECTION SYSTEM ===")
    print("1. Collect Data")
    print("2. Train Model")
    print("3. Real-time Prediction")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        print("\nStarting data collection...")
        from collect_data import collect_data
        collect_data()
    elif choice == '2':
        print("\nTraining model...")
        train_model()
    elif choice == '3':
        print("\nStarting real-time prediction...")
        predict_signs()
    elif choice == '4':
        print("\nExiting program. Goodbye!")
        return
    else:
        print("\nInvalid choice. Please try again.")
    
    main_menu()

if __name__ == "__main__":
    # If you want to train directly, uncomment this line
    # train_model()
    
    # If you want to predict directly, uncomment this line
    predict_signs()
    
    # If you want the menu interface, uncomment this line
    # main_menu()