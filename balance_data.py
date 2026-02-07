import os
import random
from PIL import Image

# CONFIG
DATASET_DIR = "IESA_Hackathon_Dataset"
TARGET_COUNT_PER_CLASS = 200  # Aim for this many images per class
MAX_OPENS = 250               # Don't let Opens exceed this

def balance_classes():
    print(f"Balancing dataset in: {DATASET_DIR}")
    
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(files)
        print(f"Processing {class_name}: Found {count} images.")

        # CASE 1: TOO MANY (Undersample) -> "2_Opens"
        if count > MAX_OPENS:
            print(f"  -> Too many! Deleting {count - MAX_OPENS} random files...")
            files_to_delete = random.sample(files, count - MAX_OPENS)
            for f in files_to_delete:
                os.remove(os.path.join(class_path, f))
            print(f"  -> Now has {MAX_OPENS} images.")

        # CASE 2: TOO FEW (Augment) -> Shorts, Protrusions, etc.
        elif count < TARGET_COUNT_PER_CLASS and count > 0:
            print(f"  -> Too few! Augmenting to reach target...")
            
            # We need to generate roughly enough to hit the target
            # Strategies: Rotate 90, 180, 270, Flip LR, Flip TB
            
            augmentations_needed = TARGET_COUNT_PER_CLASS - count
            generated = 0
            
            while generated < augmentations_needed:
                for f in files:
                    if generated >= augmentations_needed:
                        break
                        
                    try:
                        img_path = os.path.join(class_path, f)
                        img = Image.open(img_path)
                        
                        # Apply random transform
                        op = random.choice(["rot90", "rot180", "rot270", "flipLR", "flipTB"])
                        
                        if op == "rot90": new_img = img.rotate(90)
                        elif op == "rot180": new_img = img.rotate(180)
                        elif op == "rot270": new_img = img.rotate(270)
                        elif op == "flipLR": new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif op == "flipTB": new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        
                        # Save
                        new_name = f"{os.path.splitext(f)[0]}_{op}_{random.randint(100,999)}.jpg"
                        new_img.save(os.path.join(class_path, new_name))
                        generated += 1
                    except Exception as e:
                        print(f"Error augmenting {f}: {e}")
                        
            print(f"  -> Generated {generated} new images.")

    print("\n--- BALANCING COMPLETE ---")

if __name__ == "__main__":
    balance_classes()
