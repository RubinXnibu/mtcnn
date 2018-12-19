#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

image = cv2.imread('C:/Users/smdsbz/Desktop/test.jpg')
print('image.shape: {}'.format(image.shape))
results = detector.detect_faces(image)
# print(results)

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

exit()

cv2.imwrite('ivan_drawn.jpg', image)


import time

start_time = time.time()

print('100x test starting at timestamp {}'.format(start_time))
for _ in range(100):
    _ = detector.detect_faces(image)
end_time = time.time()

print('100x test ended at {}'.format(end_time))

print('avg time: {}'.format((end_time - start_time) / 100))
