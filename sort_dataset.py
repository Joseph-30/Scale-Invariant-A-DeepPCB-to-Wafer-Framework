import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = "PCBData"  # The folder containing groupXXXXX
TARGET_DIR = "IESA_Hackathon_Dataset"

# Hackathon Class Mapping (Folder Names)
CLASSES = {
    1: "2_Opens",
    2: "1_Shorts",
    3: "3_LER",          # Mousebite -> Line Edge Roughness
    4: "4_Protrusions",  # Spur -> Protrusions (Extra credit class)
    5: "5_Foreign_Material", # Copper -> Foreign Material
    6: "6_Pin_hole"      # Pin hole
}

def create_folders():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    # Create Clean folder
    os.makedirs(os.path.join(TARGET_DIR, "0_Clean"), exist_ok=True)
    
    # Create Defect folders
    for class_name in CLASSES.values():
        os.makedirs(os.path.join(TARGET_DIR, class_name), exist_ok=True)

    # Create Synthetic folder (placeholder for later)
    os.makedirs(os.path.join(TARGET_DIR, "7_Synthetic_Augmented"), exist_ok=True)

def process_dataset():
    create_folders()
    
    # Walk through all group folders (group00041, group12000, etc.)
    for group_name in os.listdir(SOURCE_DIR):
        group_path = os.path.join(SOURCE_DIR, group_name)
        if not os.path.isdir(group_path):
            continue
            
        # Inside group folder, usually logic is: 
        # /[GroupID] -> images
        # /[GroupID]_not -> annotations
        
        # Finding the image and annotation subfolders
        subfolders = os.listdir(group_path)
        img_subfolder = None
        ann_subfolder = None
        
        for sf in subfolders:
            if sf.endswith("_not"):
                ann_subfolder = os.path.join(group_path, sf)
            elif "." not in sf and "not" not in sf: # Simple check for image folder
                img_subfolder = os.path.join(group_path, sf)
                
        if not img_subfolder or not ann_subfolder:
            continue

        print(f"Processing {group_name}...")

        # Process Annotations to find Defects
        for ann_file in os.listdir(ann_subfolder):
            if not ann_file.endswith(".txt"):
                continue
                
            # Annotation format: x1, y1, x2, y2, type
            # Filename match: 00041000.txt -> 00041000_test.jpg
            base_id = ann_file.replace(".txt", "")
            
            with open(os.path.join(ann_subfolder, ann_file), 'r') as f:
                lines = f.readlines()
                
            # We take the FIRST defect found in the file to decide the class 
            # (Simplification for Phase 1)
            if lines:
                try:
                    # Parse the last digit (type) from the first line
                    # format: "354 120 376 155 1" -> type is 1 (Space separated)
                    first_defect = lines[0].strip().split()
                    defect_type = int(first_defect[-1])
                    
                    if defect_type in CLASSES:
                        src_img = os.path.join(img_subfolder, f"{base_id}_test.jpg")
                        dst_folder = os.path.join(TARGET_DIR, CLASSES[defect_type])
                        dst_img = os.path.join(dst_folder, f"{base_id}_test.jpg")
                        
                        if os.path.exists(src_img):
                            shutil.copy2(src_img, dst_img)
                except:
                    pass

        # Process Clean Images (Templates)
        # We just grab the _temp.jpg files
        count = 0
        for img_file in os.listdir(img_subfolder):
            if img_file.endswith("_temp.jpg") and count < 50: # Limit to 50 per group to avoid bloat
                src = os.path.join(img_subfolder, img_file)
                dst = os.path.join(TARGET_DIR, "0_Clean", img_file)
                shutil.copy2(src, dst)
                count += 1

    print("Sorting Complete! Check the folder:", TARGET_DIR)

if __name__ == "__main__":
    process_dataset()