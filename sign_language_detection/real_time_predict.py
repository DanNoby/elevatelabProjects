import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# Current information
current_user = "DanNoby"
current_datetime = "2025-05-15 11:39:14"  # UTC format

# Print startup message
print("Sign Language Recognition System Starting...")
print(f"Current Date and Time: {current_datetime}")
print(f"Current User: {current_user}")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
tf.get_logger().setLevel('ERROR')  # Suppress TF warnings

try:
    # Load your trained model
    print("Loading model...")
    model = tf.keras.models.load_model('sign_language_cnn_model.h5')
    print("Model loaded successfully!")
    
    # Get the actual number of output classes from the model
    num_classes = model.output_shape[1]
    print(f"Model has {num_classes} output classes")
    
    # Set image size same as training
    IMG_SIZE = 64
    
    # Your specific labels - only use the ones you care about
    used_labels = ['0', '1', '2', '3', '4', '5']
    print(f"Using labels: {used_labels}")
    
    # Check camera availability before starting
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Trying alternative camera index...")
        # Try alternative camera index
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise Exception("Could not access any camera. Please check your camera connection.")
    
    # Get actual frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        raise Exception("Could not read frame from camera. Please check your camera.")
    
    frame_height, frame_width, _ = test_frame.shape
    print(f"Camera initialized successfully. Frame size: {frame_width}x{frame_height}")
    
    # MediaPipe hands setup
    print("Initializing hand detection...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize variables for prediction smoothing
    prev_prediction = None
    frame_count = 0
    smooth_factor = 0.7
    
    print("Starting video capture. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Camera disconnected?")
            # Try to reinitialize camera
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Could not reconnect to camera. Exiting.")
                break
            continue
        
        # Increment frame count
        frame_count += 1
        
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame and find hands
        result = hands.process(rgb_frame)
        
        # Display user info on frame
        cv2.putText(frame, f"User: {current_user}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, current_datetime, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display instructions
        h, w, c = frame.shape
        cv2.putText(frame, "Show a number from 0-5", (w - 250, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (w - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box of hand
                h, w, c = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                
                # Add margin
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # Crop hand image
                hand_img = frame[y_min:y_max, x_min:x_max]
                
                # Error handling for small/invalid crops
                if hand_img.size == 0 or hand_img.shape[0] < 10 or hand_img.shape[1] < 10:
                    continue
                
                # Only run prediction every few frames for performance
                if frame_count % 3 == 0:  
                    # Preprocess for model
                    processed_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    processed_img = processed_img.astype('float32') / 255.0
                    processed_img = np.expand_dims(processed_img, axis=0)  # batch dimension
                    
                    # Predict
                    preds = model.predict(processed_img, verbose=0)  # Disable verbose output
                    
                    # Find the highest prediction among the labels we care about (0-5)
                    relevant_preds = preds[0][:len(used_labels)]
                    class_idx = np.argmax(relevant_preds)
                    confidence = relevant_preds[class_idx]
                    
                    # Smoothing predictions
                    if prev_prediction is not None and confidence < 0.95:
                        # Smooth predictions when confidence isn't very high
                        if prev_prediction == class_idx:
                            # Boost confidence if prediction is the same
                            confidence = min(confidence * 1.1, 1.0)
                        else:
                            # Reduce confidence if prediction changed
                            if confidence < smooth_factor:
                                class_idx = prev_prediction
                    
                    prev_prediction = class_idx
                    
                    # Store the latest prediction
                    latest_class_idx = class_idx
                    latest_confidence = confidence
                
                # Display the latest prediction (even on frames we don't calculate a new one)
                if 'latest_class_idx' in locals() and 'latest_confidence' in locals():
                    # Determine color based on confidence
                    if latest_confidence > 0.9:
                        color = (0, 255, 0)  # Green for high confidence
                    elif latest_confidence > 0.7:
                        color = (0, 165, 255)  # Orange for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Display prediction with confidence
                    if latest_confidence > 0.5:  # Lower the threshold a bit for testing
                        text = f"{used_labels[latest_class_idx]}: {latest_confidence*100:.1f}%"
                    else:
                        text = f"Uncertain: {latest_confidence*100:.1f}%"
                    
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Draw bounding box around hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Add a sign guide
                    cv2.putText(frame, f"Showing sign for: {used_labels[latest_class_idx]}", 
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show the processed frame
        cv2.imshow("Sign Language Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q'. Exiting...")
            break

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up
    print("Cleaning up resources...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Application terminated.")