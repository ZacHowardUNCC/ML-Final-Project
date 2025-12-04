#!/usr/bin/env python3
from pathlib import Path
import shutil


# Flattens the parent directory. WARNING: deletes the base and sub directories
def flatten_and_move_up(base_dir):
    project_dir = (Path(__file__).parent.parent).resolve(strict=True)
    base_dir = (project_dir / base_dir).resolve(strict=True)

    for sub_dir in ["train", "test", "val"]:
        sub_path = (base_dir / sub_dir).resolve(strict=True)
        for file in sub_path.iterdir():
            shutil.move(str(file), base_dir.parent)

        try:
            sub_path.rmdir()
        except OSError:
            print(f"Could not delete {sub_path}")

    try:
        base_dir.rmdir()
    except OSError:
            print(f"Could not delete {base_dir}")

# User confirmation helper
def confirm_action():
    print("WARNING: Running this script will attempt to flatten the dataset.")
    print("This will move files and delete the original '100k', 'train', 'test', 'val' directories.")
    response = input("Are you sure you want to continue? (Y/n): ").strip().lower()

    if response in ("n", "no"):
        raise RuntimeError("Operation cancelled by user.")
    
    # anything else (including empty string) counts as Yes
    return

def main():
    
    confirm_action()

    # WARNING: deletes the base and sub directories
    flatten_and_move_up("data/images/100k")
    flatten_and_move_up("data/labels/100k")
    print("Flattening complete!")

if __name__ == "__main__":
    main()