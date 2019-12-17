# -*- coding: UTF-8 -*-
import sys
import os
import dlib
import glob
import numpy
from skimage import io
import cv2
import imutils

video_capture = cv2.VideoCapture(0)
# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()
# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()
descriptors = []
person_list=['aaron', 'tommy', 'tony']
for person in person_list:
    faces_folder_path = '.\\Image\\' + person + '\\'
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)
        # 人臉辨識
        dets = detector(img, 0)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # 特徵
            shape = sp(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

            # 3.取得Vector，128維特徵向量
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # 轉換numpy array格式
            v = numpy.array(face_descriptor)
            descriptors.append(v)
        win.add_overlay(dets)
        # dlib.hit_enter_to_continue()
    numpy.save(person+'.npy',descriptors)
