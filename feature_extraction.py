
# perform feature extraction here
# return the feature vector
import cv2
import os
from matplotlib import pyplot as plt


def extract_features(images):
    """
    use opencv-sift
    """
    all_features = []
    sift = cv2.SIFT_create()
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        all_features.append((keypoints, descriptors))
    return all_features


# C:\Users\WilliamLee\Desktop\信息安全专业课\大三下\计算机视觉\大作业\code_template
current_path = os.getcwd()
folder = 'images'
img_folder = os.path.join(current_path, folder)

img_files = os.listdir(img_folder)
image_paths = [os.path.join(img_folder, file) for file in img_files]
images = [cv2.imread(image_path) for image_path in image_paths]

features = extract_features(images)

# 为了方便测试先注释掉
# for i, (image, (keypoints, _)) in enumerate(zip(images, features)):
#     image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     plt.imshow(image_with_keypoints)
#     plt.axis('off')
#     plt.show()



