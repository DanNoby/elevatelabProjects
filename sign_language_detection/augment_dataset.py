import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
source_dir = 'dataset'  # Where your original 5 images/class are
target_dir = 'augmented_dataset'  # New folder to store all augmented images
img_height, img_width = 64, 64
augment_count = 495  # Augmented images per original

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create augmented directory structure
for label in os.listdir(source_dir):
    os.makedirs(os.path.join(target_dir, label), exist_ok=True)

# Perform augmentation
for label in os.listdir(source_dir):
    gen = datagen.flow_from_directory(
        source_dir,
        classes=[label],
        target_size=(img_height, img_width),
        batch_size=1,
        save_to_dir=os.path.join(target_dir, label),
        save_prefix='aug',
        save_format='jpeg'
    )
    count = 0
    for _ in gen:
        count += 1
        if count >= augment_count:
            break
    print(f"✓ Augmented {label}")
