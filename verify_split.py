import os

DATASET_DIR = "Final_Submission_Dataset"
SPLITS = ["Train", "Validation", "Test"]

print(f"{'Class':<25} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total':<10}")
print("-" * 75)

class_names = set()
# First gather all class names from all splits to be safe
for split in SPLITS:
    split_path = os.path.join(DATASET_DIR, split)
    if os.path.exists(split_path):
        for d in os.listdir(split_path):
            if os.path.isdir(os.path.join(split_path, d)):
                class_names.add(d)

sorted_classes = sorted(list(class_names))

total_train = 0
total_val = 0
total_test = 0

for cls in sorted_classes:
    counts = {}
    for split in SPLITS:
        path = os.path.join(DATASET_DIR, split, cls)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[split] = count
        else:
            counts[split] = 0
    
    total = counts['Train'] + counts['Validation'] + counts['Test']
    print(f"{cls:<25} | {counts['Train']:<10} | {counts['Validation']:<10} | {counts['Test']:<10} | {total:<10}")
    
    total_train += counts['Train']
    total_val += counts['Validation']
    total_test += counts['Test']

print("-" * 75)
print(f"{'TOTAL':<25} | {total_train:<10} | {total_val:<10} | {total_test:<10} | {total_train+total_val+total_test:<10}")
