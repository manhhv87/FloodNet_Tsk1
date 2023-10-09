import cv2
import os
from tqdm.auto import tqdm


def resize_and_save(path_scr, path_dist, new_size=(400, 300), dir_temp_root="/content/dataset/", dir_local_root="/content/FloodNet/"):
    for img_name in tqdm(os.listdir(os.path.join(dir_temp_root, path_scr))):
        img = cv2.imread(os.path.join(dir_temp_root, path_scr, img_name))
        img = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(dir_local_root, path_dist, img_name), img)
