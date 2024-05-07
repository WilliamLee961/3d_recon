# perform feature matching here
import cv2
import os
from feature_extraction import extract_features
from matplotlib import pyplot as plt
import numpy as np
# return the matching result


def draw_matches(image1, keypoints1, image2, keypoints2, matches, output_folder=None):
    """Draw matches between two images.
    """
    # convert to uint8
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    # if output_folder is not None:
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     img_name = "matches.jpg"
    #     output_path = os.path.join(output_folder, img_name)
    #     cv2.imwrite(output_path, image_matches)

    plt.imshow(image_matches)
    plt.axis('off')
    plt.show()
    return image_matches


def match_features(features1, features2):
    """
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    keypoints1, descriptors1 = features1
    keypoints2, descriptors2 = features2

    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)
    matches = good_matches  # list
    return matches

# 检验函数输出
current_path = os.getcwd()
folder = 'images'
img_folder = os.path.join(current_path, folder)

img_files = os.listdir(img_folder)
image_paths = [os.path.join(img_folder, file) for file in img_files]
images = [cv2.imread(image_path) for image_path in image_paths]

features = extract_features(images)

# draw_matched中的参数
image1 = images[0]
image2 = images[1]

#type tuple

feature1 = features[0]
feature2 = features[1]
print(feature1)
keypoints1 = feature1[0]
print(keypoints1)
keypoints2 = feature2[0]

matches = match_features(feature1, feature2)
# draw_matches(image1, keypoints1, image2, keypoints2, matches)

