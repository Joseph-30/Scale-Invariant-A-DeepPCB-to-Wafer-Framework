import os
import random
import shutil
from PIL import Image

# CONFIG
SOURCE_DIR = "IESA_Hackathon_Dataset"
TARGET_DIR = "Final_Submission_Dataset"
CLASS_NAME = "7_Crack"
TARGET_COUNT = 200
SPLITS = {"Train": 0.7, "Validation": 0.15, "Test": 0.15}

def augment_and_distribute_cracks():
    print(f"Augmenting {CLASS_NAME} to {TARGET_COUNT}...")
    
    # 1. Locate Source Images
    class_source_path = os.path.join(SOURCE_DIR, CLASS_NAME)
    if not os.path.exists(class_source_path):
        print(f"Error: {class_source_path} does not exist.")
        return

    # 2. Get existing files
    files = [f for f in os.listdir(class_source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    initial_count = len(files)
    print(f"Found {initial_count} initial images.")
    
    if initial_count == 0:
        print("No images found to augment.")
        return

    # 3. Create a temporary list of all images (original + new augmented ones)
    # We will generate them in memory/temp list first, then distribute directly to final folders
    
    # We need to generate this many more
    needed = TARGET_COUNT - initial_count
    generated_files = [] 
    
    # First, let's just use the originals
    all_images = [] # List of tuples (filename, image_object)
    
    for f in files:
        img_path = os.path.join(class_source_path, f)
        img = Image.open(img_path)
        all_images.append((f, img))
        
    print(f"Generating {needed} augmented images...")
    
    generated_count = 0
    while len(all_images) < TARGET_COUNT:
        # Pick a random original image to augment
        original_name, original_img = random.choice(all_images[:initial_count])
        
        op = random.choice(["rot90", "rot180", "rot270", "flipLR", "flipTB", "transpose"])
        
        try:
            if op == "rot90": new_img = original_img.rotate(90)
            elif op == "rot180": new_img = original_img.rotate(180)
            elif op == "rot270": new_img = original_img.rotate(270)
            elif op == "flipLR": new_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif op == "flipTB": new_img = original_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif op == "transpose": new_img = original_img.transpose(Image.TRANSPOSE)
            
            new_name = f"{os.path.splitext(original_name)[0]}_aug_{generated_count}_{op}.jpg"
            all_images.append((new_name, new_img))
            generated_count += 1
            
        except Exception as e:
            print(f"Error augmenting: {e}")

    print(f"Total images prepared: {len(all_images)}")

    # 4. Shuffle and Split
    random.shuffle(all_images)
    
    total_imgs = len(all_images)
    train_end = int(total_imgs * SPLITS["Train"])
    val_end = train_end + int(total_imgs * SPLITS["Validation"])
    
    train_set = all_images[:train_end]
    val_set = all_images[train_end:val_end]
    test_set = all_images[val_end:]
    
    print(f"Split Plan -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # 5. Write to Final_Submission_Dataset
    # We overwrite/add to the existing folders in Final_Submission_Dataset
    
    def save_set(dataset, split_name):
        dest_dir = os.path.join(TARGET_DIR, split_name, CLASS_NAME)
        
        # Ensure clean slate for this class in this split if desired, 
        # OR just overwrite. Since we want to update the split count, 
        # let's just clear the specific class folder in destination first to avoid duplicates
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        
        for name, img in dataset:
            save_path = os.path.join(dest_dir, name)
            img.save(save_path)
            
    save_set(train_set, "Train")
    save_set(val_set, "Validation")
    save_set(test_set, "Test")

    print(f"Successfully updated {CLASS_NAME} in {TARGET_DIR}")

if __name__ == "__main__":
    augment_and_distribute_cracks()
