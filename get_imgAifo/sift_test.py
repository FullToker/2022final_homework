import cv2
import numpy as np

img_leftname="./data/jpeg_data/HX1-Ro_GRAS_NaTeCamB-F-001_SCI_N_20210702031746_20210702031746_00048.jpeg"
img_rightname="./data/jpeg_data/HX1-Ro_GRAS_NaTeCamA-F-001_SCI_N_20210702031729_20210702031729_00048.jpeg"

img_left=cv2.imread(img_leftname)
img_right=cv2.imread(img_rightname)
cv2.imshow("left",img_left)
cv2.imshow("right",img_right)

gary_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
gray_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
#SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()
keypoint_left, des_left = sift.detectAndCompute(gary_left, None)
keypoint_right, des_right = sift.detectAndCompute(gray_right, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)

goodMatches = []
ransacMatches=[]
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatches.append(m)
        ransacMatches.append(m)

print("%d good maches"%(len(goodMatches)))
# 增加一个维度
goodMatch2 = np.expand_dims(goodMatches, 1)
img_out = cv2.drawMatchesKnn(gary_left, keypoint_left, gray_right, keypoint_right, goodMatch2, None, flags=2)
cv2.imshow("out",img_out)

#RANSAC剔除点
left_pts = np.float32([keypoint_left[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
right_pts = np.float32([keypoint_right[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
#在这里调用RANSAC方法得到解H
ransacReprojThreshold=5.0
H, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, ransacReprojThreshold)
# print(mask)
print(mask.shape)
matchesMask = mask.ravel().tolist()

error_nums=0
for i in range(len(goodMatches)):
    if matchesMask[i]==0:
        ransacMatches.pop(i-error_nums)
        error_nums+=1

print("After Ransac: {}matches".format(len(ransacMatches)))
goodMatch3 = np.expand_dims(ransacMatches, 1)
img_ransac = cv2.drawMatchesKnn(gary_left, keypoint_left, gray_right, keypoint_right, goodMatch3, None, flags=2)
cv2.imshow("ransac",img_ransac)

cv2.waitKey(0)