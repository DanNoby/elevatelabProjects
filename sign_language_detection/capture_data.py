import cv2
import os

# List of labels (you can customize this)
labels = ['0','1','2','3','4','5','6','7','8','9','p','n']  # Add more as needed

# Create folders if they don't exist
dataset_path = 'dataset'
for label in labels:
    os.makedirs(os.path.join(dataset_path, label), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press the corresponding key to capture image for that label:")
print(", ".join(f"{label}: '{label.lower()}' key" for label in labels))
print("Press 'q' to quit.")

# Counters to keep track of how many images saved per label
counters = {label: 0 for label in labels}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    cv2.putText(frame, "Press key for label to capture image. Q to quit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show counts on the frame
    y0 = 60
    for label in labels:
        text = f"{label}: {counters[label]} images"
        cv2.putText(frame, text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y0 += 30

    cv2.imshow('Capture Data', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting...")
        break

    # Check if pressed key corresponds to a label (case-insensitive)
    for label in labels:
        if key == ord(label.lower()):
            counters[label] += 1
            filename = os.path.join(dataset_path, label, f"{label}_{counters[label]}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

cap.release()
cv2.destroyAllWindows()
