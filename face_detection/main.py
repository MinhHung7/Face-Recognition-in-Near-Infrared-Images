from yoloface_face_detection import yoloface_detection
import cv2
import copy
import os
from pathlib import Path

def get_crop_images(crop_images, path):
    """ crop_images is a list of images(numpy array of list)
        path is the path to the folder where the images are to be saved
        model name is the name of the model used to detect faces(inside path model_name folder is created and images are saved)    
    """
    for i in range(len(crop_images)):
        image = crop_images[i]
        file_path = os.path.join(path, str(i))
        print(image)
        cv2.imwrite(file_path+".jpg", image)

def get_all_faces(images_folders_path):
    for folder in os.listdir(images_folders_path):

        yolo_cropped_images = []

        images_folder_path = os.path.join(images_folders_path, folder)
        for file in os.listdir(images_folder_path):
            if not file.endswith(".jpg"):
                continue
            
            image_path = os.path.join(images_folder_path, file)
            img = cv2.imread(image_path)


            yolo_cropped = yoloface_detection(img)
            
            yolo_cropped_images.extend(yolo_cropped)

        os.makedirs(os.path.join("/content/yolo_cropped_images", folder), exist_ok=True)
        yolo_cropped_images_folder = os.path.join("/content/yolo_cropped_images", folder)
        get_crop_images(yolo_cropped_images, yolo_cropped_images_folder)


def main():
    get_all_faces('/content/TD_NIR_A_Set')


if __name__ == '__main__':
    main()
