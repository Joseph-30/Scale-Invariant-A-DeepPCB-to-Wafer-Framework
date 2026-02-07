import os

SOURCE_DIR = "PCBData"
CLASSES = {1, 2, 3, 4, 5}
unknown_types = set()
files_with_unknown_types = []

for group_name in os.listdir(SOURCE_DIR):
    group_path = os.path.join(SOURCE_DIR, group_name)
    if not os.path.isdir(group_path):
        continue
        
    subfolders = os.listdir(group_path)
    ann_subfolder = None
    
    for sf in subfolders:
        if sf.endswith("_not"):
            ann_subfolder = os.path.join(group_path, sf)
            break
            
    if not ann_subfolder:
        continue

    for ann_file in os.listdir(ann_subfolder):
        if not ann_file.endswith(".txt"):
            continue
            
        with open(os.path.join(ann_subfolder, ann_file), 'r') as f:
            lines = f.readlines()
            
        if lines:
            try:
                first_defect = lines[0].strip().split()
                defect_type = int(first_defect[-1])
                
                if defect_type not in CLASSES:
                    unknown_types.add(defect_type)
                    files_with_unknown_types.append((ann_file, defect_type))
            except:
                pass

print("Unknown defect types found:", unknown_types)
print("Number of files skipped due to unknown type:", len(files_with_unknown_types))
if files_with_unknown_types:
    print("Example skipped files:", files_with_unknown_types[:5])
