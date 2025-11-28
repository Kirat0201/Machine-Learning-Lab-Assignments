import os
from PIL import Image
import numpy as np

def img_to_array(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    img = Image.open(full_path)
    arr = np.array(img)
    if img.mode == 'L':
        save_path = os.path.join(script_dir, 'image_gray.txt')
        np.savetxt(save_path, arr, fmt='%d')
        print("Saved grayscale image to image_gray.txt")
    elif img.mode in ['RGB', 'RGBA']:
        arr_reshaped = arr.reshape(-1, arr.shape[2])
        save_path = os.path.join(script_dir, 'image_rgb.txt')
        np.savetxt(save_path, arr_reshaped, fmt='%d')
        print("Saved RGB image to image_rgb.txt")
    else:
        print("Unsupported mode:", img.mode)

if __name__ == "__main__":
    img_to_array('bg10.jpg')
