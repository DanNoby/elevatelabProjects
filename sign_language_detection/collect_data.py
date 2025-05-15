import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Current information
CURRENT_USER = "DanNoby"
CURRENT_DATETIME = "2025-05-15 11:42:39"

# Display basic information
print(f"Sign Language Detection - Data Collection")
print(f"User: {CURRENT_USER}")
print(f"Date/Time: {CURRENT_DATETIME}")

# Constants
IMG_SIZE = 64  # Image size for the model
LABELS = ['0', '1', '2', '3', '4', '5']  # Labels we want to collect
DATA_DIR = "sign_language_data"  # Directory to save data

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def collect_data():
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
        
    print("\n=== DATA COLLECTION MODE ===")
    print("We will collect images for each digit (0-5).")
    print("For each digit, show your hand sign and press 'c' to capture images.")
    print("Press 'n' to move to the next digit.")
    print("Press 'q' to quit at any time.\n")
    
    samples_per_class = 100  # Number of samples to collect per class
    
    for label_idx, label in enumerate(LABELS):
        print(f"Preparing to collect data for digit: {label}")
        print(f"Show the sign for '{label}' and press 'c' to start capturing")
        print(f"Press 'n' when done to move to the next digit")
        
        sample_count = 0
        collecting = False
        
        while sample_count < samples_per_class:
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
                    
                    # If collecting, save the image
                    if collecting and sample_count < samples_per_class:
                        # Extract hand region
                        hand_img = frame[y_min:y_max, x_min:x_max]
                        
                        # Make sure we have a valid image
                        if hand_img.size != 0:
                            # Resize to standard size
                            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                            
                            # Save the image
                            save_path = os.path.join(DATA_DIR, label, f"{label}_{sample_count}.jpg")
                            cv2.imwrite(save_path, hand_img)
                            sample_count += 1
                            print(f"Captured sample {sample_count}/{samples_per_class} for digit {label}")
                            
                            # Short delay to avoid duplicate images
                            time.sleep(0.1)
            
            # Show tracking info on frame
            info_text = f"Digit: {label} | Samples: {sample_count}/{samples_per_class}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if collecting:
                status_text = "CAPTURING - Hold your gesture"
                color = (0, 0, 255)  # Red for capturing
            else:
                status_text = "Press 'c' to start capturing"
                color = (255, 255, 255)  # White for waiting
                
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Press 'n' for next digit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Data Collection", frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == ord('c'):
                collecting = not collecting
                if collecting:
                    print(f"Started capturing samples for digit {label}")
                else:
                    print(f"Paused capturing samples for digit {label}")
            elif key == ord('n'):
                print(f"Moving to the next digit. Collected {sample_count} samples for digit {label}")
                break
        
        # Check if we got enough samples or want to continue anyway
        if sample_count < samples_per_class:
            print(f"Warning: Only collected {sample_count}/{samples_per_class} samples for digit {label}")
            response = input("Continue to the next digit? (y/n): ").lower()
            if response != 'y':
                cap.release()
                cv2.destroyAllWindows()
                return False
    
    print("\nData collection completed for all digits!")
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    collect_data()