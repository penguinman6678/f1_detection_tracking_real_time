import os
from pathlib import Path



"""
Based on the curated F1 dataset, we want to change all the lines in the .txt files (for train / test / valid)
to have a 0 as the first entry (as we just want to detect cars or not).
The original has separate team labels, but we want to train a separate CNN for that
"""

def change_label_to_zero(data_dir=None):

    directory = Path(data_dir)

    for txt in directory.glob("*.txt"):
        with txt.open("r") as f:
            lines = f.readlines()
        
        modified_lines = [] # In case there are more than one car in the image
        for line in lines:
            stripped = line.strip()
            numbers = stripped.split()
            numbers[0] = "0" # car class
            modified_lines.append(" ".join(numbers) + "\n")
        
        with txt.open("w") as f:
            f.writelines(modified_lines)
        
        print(f"Updated txt file: {txt}")

    return


if __name__ == "__main__":

    train_data_dir = "./train/labels"
    test_data_dir = "./test/labels"
    val_data_dir = "./valid/labels"

    for data_dir in [train_data_dir, test_data_dir, val_data_dir]:
        change_label_to_zero(data_dir=data_dir)