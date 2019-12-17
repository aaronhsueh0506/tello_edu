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
faces_folder_path = '.\\Image\\tony\\'

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()
# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()
descriptors = []

### take feature and save###

# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(f)
#
#     win.clear_overlay()
#     win.set_image(img)
#     # 人臉辨識
#     dets = detector(img, 0)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # 特徵
#         shape = sp(img, d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
#                                                   shape.part(1)))
#         # Draw the face landmarks on the screen.
#         win.add_overlay(shape)
#
#         # 3.取得Vector，128維特徵向量
#         face_descriptor = facerec.compute_face_descriptor(img, shape)
#
#         # 轉換numpy array格式
#         v = numpy.array(face_descriptor)
#         descriptors.append(v)
#     win.add_overlay(dets)
#     # dlib.hit_enter_to_continue()
# numpy.save('data.npy',descriptors)

# a=numpy.mean(numpy.load('aaron.npy'), axis=1)
# b=numpy.mean(numpy.load('tommy.npy'), axis=1)
# c=numpy.mean(numpy.load('tony.npy'), axis=1)

candidate = ['aaron', 'tommy', 'tony']
for i in candidate:
    descriptors.append(numpy.mean(numpy.load(i + '.npy'), axis=0))
    
while(video_capture.isOpened()):
    ret , frame = video_capture.read()

    dets = detector(frame, 0)

    dist = []
    for k, d in enumerate(dets):
      shape = sp(frame, d)
      face_descriptor = facerec.compute_face_descriptor(frame, shape)
      d_test = numpy.array(face_descriptor)

      x1 = d.left()
      y1 = d.top()
      x2 = d.right()
      y2 = d.bottom()
      # 以方框標示偵測的人臉
      cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)

      # 計算歐式距離
      for i in descriptors:
        dist_ = numpy.linalg.norm(i-d_test)
        dist.append(dist_)
    
    # 將比對人名和比對出來的歐式距離組成一個dict
    c_d = dict( zip(candidate,dist))
    if not c_d :
        print('no face dectect')
        
        frame = imutils.resize(frame, width = 600)
        #frame = cv2.cvtColor(frame,cv2. COLOR_BGR2RGB)
        cv2.imshow( "Face Recognition", frame)
    else:
        # 根據歐式距離由小到大排序
        cd_sorted = sorted(c_d.items(), key = lambda d:d[ 1])
        
        # 取得最短距離就為辨識出的人名
        rec_name = cd_sorted[0][0]
        if cd_sorted[0][1]>0.45: rec_name = 'Unrecognizeable'
        
        # 將辨識出的人名印到圖片上面
        cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)

        frame = imutils.resize(frame, width = 600)
        #frame = cv2.cvtColor(frame,cv2. COLOR_BGR2RGB)
        cv2.imshow( "Face Recognition", frame)
    #隨意Key一鍵結束程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
