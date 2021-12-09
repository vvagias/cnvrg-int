import os
import shutil


def move_files():
    for k, v in os.environ.items():
        if k.startswith("INPUT") and os.listdir(v):
            shutil.copytree(v, "./output")
            print(f"Moved contents of {v} to the output dir")
            break
