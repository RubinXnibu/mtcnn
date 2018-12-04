#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

image = cv2.imread('C:/Users/smdsbz/Desktop/test.jpg')
results = detector.detect_faces(image)
print(results)

for result in results:
    bounding_box = result['box']
    keypoints = result['keypoints']

    cv2.rectangle(
        image,
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
        (0,155,255), 2
    )

    for keypoint in keypoints:
        cv2.circle(image, keypoint, 2, (0,155,255), 2)

cv2.imwrite('ivan_drawn.jpg', image)
