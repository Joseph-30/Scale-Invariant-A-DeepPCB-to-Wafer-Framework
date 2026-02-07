import os
import shutil
import random

# CONFIG
SOURCE_DIR = "IESA_Hackathon_Dataset"
TARGET_DIR = "Final_Submission_Dataset"
SPLIT_RATIO = (0.7, 0.15, 0.15)  # 70% Train, 15% Val, 15% Test

def split_data():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} not found!")
        return

    for class_name in os.listdir(SOURCE_DIR):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir): continue
        
        # Get all images
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        # Calculate split counts
        total = len(images)
        train_count = int(total * SPLIT_RATIO[0])
        val_count = int(total * SPLIT_RATIO[1])
        
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count+val_count]
        test_imgs = images[train_count+val_count:]
        
        # Move files
        for subset, imgs in [("Train", train_imgs), ("Validation", val_imgs), ("Test", test_imgs)]:
            dest_dir = os.path.join(TARGET_DIR, subset, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(os.path.join(class_dir, img), os.path.join(dest_dir, img))
                
    print(f"Dataset split complete! Check folder: {TARGET_DIR}")

if __name__ == "__main__":
    split_data()