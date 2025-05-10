import os

# Directories containing the label files
label_dirs = [
    r"C:\Users\YOURDESKTOP\Desktop\DogIdwithYOLO\.venv\dataset\labels\train",
    r"C:\Users\YOURDESKTOP\Desktop\DogIdwithYOLO\.venv\dataset\labels\valid",
    r"C:\Users\YOURDESKTOP\Desktop\DogIdwithYOLO\.venv\dataset\labels\test",
]

# Checking file paths to determine labels
for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"Warning: {label_dir} not found.")
        continue

    # Check filenames in the directory
    for folder in os.listdir(label_dir):
        folder_path = os.path.join(label_dir, folder)

        # If the folder contains 'dog' or 'human', assign the corresponding label
        if os.path.isdir(folder_path):
            label = None
            if 'dog' in folder.lower():
                label = 0  # Assign label 0 for Dog
            elif 'human' in folder.lower():
                label = 1  # Assign label 1 for Human

            if label is not None:
                # Check the .txt files in this folder
                for filename in os.listdir(folder_path):
                    if filename.endswith(".txt"):
                        filepath = os.path.join(folder_path, filename)

                        # If the file doesn't exist, skip it
                        if not os.path.exists(filepath):
                            print(f"Warning: {filepath} not found.")
                            continue  # Skip this file

                        with open(filepath, "r") as f:
                            lines = f.readlines()

                        # Update the class ID in each line
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                parts[0] = str(label)  # Update the class ID in each line
                                new_lines.append(" ".join(parts))

                        # Write the updated lines back to the file
                        with open(filepath, "w") as f:
                            f.write("\n".join(new_lines) + "\n")

print("Labels successfully updated based on folder names!")
