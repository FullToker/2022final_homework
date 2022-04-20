from tarfile import PAX_FORMAT
from urllib import robotparser
from xml.dom.minidom import parse
import xml.dom.minidom
import math
import cv2
from cv2 import drawKeypoints


# img顺序从右到左
path_imgs=["./data/jpeg_data/HX1-Ro_GRAS_NaTeCamA-F-001_SCI_N_20210702031729_20210702031729_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamB-F-001_SCI_N_20210702031746_20210702031746_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamA-F-002_SCI_N_20210702031904_20210702031904_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamB-F-002_SCI_N_20210702031921_20210702031921_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamA-F-003_SCI_N_20210702032039_20210702032039_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamB-F-003_SCI_N_20210702032056_20210702032056_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamA-F-004_SCI_N_20210702032214_20210702032214_00048.jpeg",
"./data/jpeg_data/HX1-Ro_GRAS_NaTeCamB-F-004_SCI_N_20210702032231_20210702032231_00048.jpeg",

]
path_2cl=["./data/ori_data/20210702031729_20210702031729_00048_A (1).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (121).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (6).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (126).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (35).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (155).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (64).2CL",
"./data/ori_data/20210702031729_20210702031729_00048_A (184).2CL",

]

import numpy as np
def Euler_2_R(pitch,roll,yaw):
    pitch*=math.pi/180
    roll*=math.pi/180
    yaw*=math.pi/180
    R_x=np.eye(3,3)
    R_y=np.eye(3,3)
    R_z=np.eye(3,3)
    R_x[1][1]=math.cos(pitch) 
    R_x[1][2]=-math.sin(pitch) 
    R_x[2][1]=math.sin(pitch) 
    R_x[2][2]=math.cos(pitch) 
    R_y[0][0]=math.cos(roll) 
    R_y[0][2]=math.sin(roll) 
    R_y[2][0]=-math.sin(roll) 
    R_y[2][2]=math.cos(roll)
    R_z[0][0]=math.cos(yaw) 
    R_z[0][1]=-math.sin(yaw) 
    R_z[1][0]=math.sin(yaw) 
    R_z[1][1]=math.cos(yaw)
    RR=np.dot(np.dot(R_y,R_x),R_z)
    return  RR
def Rt_2_T(R,x,y,z):
    T=np.eye(4,4)
    T[0:3,0:3]=R
    T[0][3]=x
    T[1][3]=y
    T[2][3]=z
    return T
def inv_T(T):
    R=T[0:3,0:3]
    t=np.mat([[0],[0],[0]])
    t[0][0]=T[0][3]
    t[1][0]=T[1][3]
    t[2][0]=T[2][3]
    T_inv=np.eye(4,4)
    T_inv[0:3,0:3]=R.transpose()
    T_abs=np.dot(-R.transpose(),t)
    T_inv[0][3]=T_abs[0]   
    T_inv[1][3]=T_abs[1]  
    T_inv[2][3]=T_abs[2]   
    return T_inv


class rover_img():
    def __init__(self,path_img,path_2cl):
        self.img=cv2.imread(path_img)
        self.path_2cl=path_2cl
        self.grayImg=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
         
        # 获取影像的参数
        DOMTree = xml.dom.minidom.parse(path_2cl)
        product = DOMTree.documentElement
        img_info=product.getElementsByTagName("Axis_Array")
        for img in img_info:
            axis_name=img.getElementsByTagName('axis_name')[0].childNodes[0].data
            if axis_name=="Line":
                self.img_height=int(img.getElementsByTagName('elements')[0].childNodes[0].data)
            if axis_name=="Sample":
                self.img_weight=int(img.getElementsByTagName('elements')[0].childNodes[0].data)  
            if axis_name=="Band":
                self.img_bandNums=int(img.getElementsByTagName('elements')[0].childNodes[0].data)
        
        instru_params=product.getElementsByTagName("Instrument_Parm")
        for param in instru_params:
            self.pixel_size=float(param.getElementsByTagName('pixel_size')[0].childNodes[0].data)*0.001
            self.focal_length=float(param.getElementsByTagName('focal_length')[0].childNodes[0].data)
            ppc=param.getElementsByTagName('Principal_Point_Coordinate')
            for xy in ppc:
                self.x0=float(xy.getElementsByTagName('x0')[0].childNodes[0].data)
                self.y0=float(xy.getElementsByTagName('y0')[0].childNodes[0].data)
        
        rover_atti=product.getElementsByTagName("Rover_Attitude")
        for param in rover_atti:
            self.pitch_b=float(param.getElementsByTagName('pitch')[0].childNodes[0].data)
            self.roll_b=float(param.getElementsByTagName('roll')[0].childNodes[0].data)
            self.yaw_b=float(param.getElementsByTagName('yaw')[0].childNodes[0].data)

        rover_xyz=product.getElementsByTagName("Rover_Location_xyz")
        for param in rover_xyz:
            self.x_b=float(param.getElementsByTagName('x')[0].childNodes[0].data)
            self.y_b=float(param.getElementsByTagName('y')[0].childNodes[0].data)
            self.z_b=float(param.getElementsByTagName('z')[0].childNodes[0].data)

        EOE=product.getElementsByTagName("Exterior_Orientation_Elements")
        for param in EOE:
            self.x_c=float(param.getElementsByTagName('camera_center_position_x')[0].childNodes[0].data)
            self.y_c=float(param.getElementsByTagName('camera_center_position_y')[0].childNodes[0].data)
            self.z_c=float(param.getElementsByTagName('camera_center_position_z')[0].childNodes[0].data)
            self.pitch_c=float(param.getElementsByTagName('camera_rotation_angle_pitch')[0].childNodes[0].data)
            self.roll_c=float(param.getElementsByTagName('camera_rotation_angle_roll')[0].childNodes[0].data)
            self.yaw_c=float(param.getElementsByTagName('camera_rotation_angle_yaw')[0].childNodes[0].data)
        
        '''
        fai: roll
        omg: pitch
        kei: yaw
        '''
        self.bc_fai=self.roll_c*math.pi/180
        self.bc_omg=self.pitch_c*math.pi/180
        self.bc_kei=self.yaw_c*math.pi/180

    def cal_matrix(self):
        K=np.eye(4,4)
        K[0][0]=self.focal_length/self.pixel_size
        K[0][2]=(self.img_weight/2+self.x0/self.pixel_size)
        K[1][1]=(self.focal_length/self.pixel_size)
        K[1][2]=(self.img_height/2-self.y0/self.pixel_size)
        self.K=K

        self.r_wb=Euler_2_R(self.pitch_b,self.roll_b,self.yaw_b)
        self.T_wb=Rt_2_T(self.r_wb,self.x_b,self.y_b,self.z_b)

        self.r_bc=Euler_2_R(self.pitch_c,self.roll_c,self.yaw_c)
        self.T_bc=Rt_2_T(self.r_bc,self.x_c,self.y_c,self.z_c)
    
    def sift_kt(self):
        sift = cv2.xfeatures2d.SIFT_create()
        self.sift_keypt, self.des = sift.detectAndCompute(self.grayImg, None)
    
    # 只有一个重叠区域的图像（第一张和 最后一张）k=0:重叠区域在左(trainIdx) k=1:重叠区域在右(queryIdx)
    def filter_pt1(self,goodmatch,k):
        idx=[]
        for i in range(len(goodmatch)):
            if k==0:
               idx.append(goodmatch[i].trainIdx)
            if k==1:
                 idx.append(goodmatch[i].queryIdx)
        
        normal_index=[]
        for i in range(len(self.sift_keypt)):
            if i not in idx:
                normal_index.append(i)
        self.normal_pts = np.float32([self.sift_keypt[m].pt for m in normal_index]).reshape(-1, 1, 2)
    # 有两个重叠区域的图像（中间）goodmatch1:与右片的重合 goodmatch2：与左区的重合
    def filter_pt2(self,goodmatch1,goodmatch2):
        idx=[]
        for i in range(len(goodmatch1)):
            idx.append(goodmatch1[i].queryIdx)
        for i in range(len(goodmatch2)):
            idx.append(goodmatch2[i].trainIdx)
        
        normal_index=[]
        for i in range(len(self.sift_keypt)):
            if i not in idx:
                normal_index.append(i)
        self.normal_pts = np.float32([self.sift_keypt[m].pt for m in normal_index]).reshape(-1, 1, 2)

    # 获取摄站点的坐标（共线方程求常数项）
    # 摄站点
    # def cal_point(self):



num=8
imgs=list(range(num))

for i in range(num):
    img=rover_img(path_imgs[i],path_2cl[i])
    img.cal_matrix()
    img.sift_kt()
    imgs[i]=img

# SIFT算子与FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

'''
matches = flann.knnMatch(imgs[0].des, imgs[1].des, k=2)
goodMatches = []
ransacMatches=[]
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatches.append(m)
        ransacMatches.append(m)


left_pts = np.float32([imgs[0].sift_keypt[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
right_pts = np.float32([imgs[1].sift_keypt[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
#print(left_pts[0][0])
#print(right_pts[0][0])
#print(len(right_pts))
# left_pts[0][0]  第二个0不变  才是一个坐标 [0]i [1]j


# 寻找不在匹配区内的点
ransacReprojThreshold=5.0
H, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, ransacReprojThreshold)
matchesMask = mask.ravel().tolist()
error_nums=0
for i in range(len(goodMatches)):
    if matchesMask[i]==0:
        ransacMatches.pop(i-error_nums)
        error_nums+=1

left_pts = np.float32([imgs[0].sift_keypt[m.queryIdx].pt for m in ransacMatches]).reshape(-1, 1, 2)
right_pts = np.float32([imgs[1].sift_keypt[m.trainIdx].pt for m in ransacMatches]).reshape(-1, 1, 2)
#print(left_pts[0][0])
#print(right_pts[0][0])
print(len(right_pts))
#print(ransacMatches[5].queryIdx)
#print(ransacMatches[5].trainIdx)
# Pw为真实世界坐标 Px 为相机坐标(i j)
Px=np.matrix([[left_pts[557][0][0]],[left_pts[557][0][1]],[1],[1]])
print(Px)
Rt=inv_T(np.dot(np.dot(imgs[0].K,imgs[0].T_wb),imgs[0].T_bc))
#print(Rt)
Pw=np.dot(Rt,imgs[0].focal_length*0.001*Px)
print(Pw)

Px=np.matrix([[right_pts[0][0][0]],[right_pts[0][0][1]],[1],[1]])
print(Px)
Rt=inv_T(np.dot(np.dot(imgs[1].K,imgs[1].T_wb),imgs[1].T_bc))
#print(Rt)
Pw=np.dot(Rt,Px)
print(Pw)

# 根据索引来判断是否为连接点
qidx=[]
tidx=[]
for i in range(len(ransacMatches)):
    qidx.append(ransacMatches[i].queryIdx)
    tidx.append(ransacMatches[i].trainIdx)
# print(qidx)


# 
Px=np.matrix([[0],[0],[0],[1]])
Rt=inv_T(np.dot(imgs[0].T_wb,imgs[0].T_bc))
Pw=np.dot(Rt,imgs[0].focal_length*0.001*Px)
print(Pw)
'''
# img_matches
img_matches=list(range(num-1))
control_Lpt=list(range(num-1))
control_Rpt=list(range(num-1))
for i in range(num-1):
    matches=flann.knnMatch( imgs[i+1].des, imgs[i].des, k=2)
    goodMatches = []
    ransacMatches=[]
    for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.40*n.distance:
            goodMatches.append(m)
            ransacMatches.append(m)
    #RANSAC剔除点
    right_nopts = np.float32([imgs[i].sift_keypt[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    left_nopts = np.float32([imgs[i+1].sift_keypt[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    #在这里调用RANSAC方法得到解H
    ransacReprojThreshold=5.0
    H, mask = cv2.findHomography(left_nopts, right_nopts, cv2.RANSAC, ransacReprojThreshold)
    # print(mask)
    # print(mask.shape)
    matchesMask = mask.ravel().tolist()
    error_nums=0
    for j in range(len(goodMatches)):
        if matchesMask[j]==0:
            ransacMatches.pop(j-error_nums)
            error_nums+=1
    
    # 写入连接点list
    right_pts = np.float32([imgs[i].sift_keypt[m.trainIdx].pt for m in ransacMatches]).reshape(-1, 1, 2)
    left_pts = np.float32([imgs[i+1].sift_keypt[m.queryIdx].pt for m in ransacMatches]).reshape(-1, 1, 2)
    match3=np.expand_dims(ransacMatches, 1)
    print("{:d} and {:d} have {:d} matches".format(i+1,i+2,len(goodMatches)))
    print("after ransac, {:d} and {:d} have {:d} good matches".format(i+1,i+2,len(ransacMatches)))
    #img_ransac = cv2.drawMatchesKnn(imgs[i+1].grayImg, imgs[i+1].sift_keypt, imgs[i].grayImg, imgs[i].sift_keypt, match3, None, flags=2)
    #result="00"+str(i+1)+"00"+str(i+2)+"ransac"
    #cv2.imwrite("./data/program/SIFT/"+result+".jpeg",img_ransac)

    img_matches[i]=ransacMatches
    control_Lpt[i]=left_pts
    control_Rpt[i]=right_pts


# print(img_matches)
# print(control_Lpt)

# 得到四张影像的公共点
# 四个数组 存放四张图像共有的点
control_maxRpt=list(range(num-3))      # img001
control_middleRpt=list(range(num-3))   # img002
control_middleLpt=list(range(num-3))   # img003
control_maxLpt=list(range(num-3))     # img004

for i in range(num-3):   # num-3
    m1=img_matches[i]
    m2=img_matches[i+1]
    num_12,num_23=[],[]
    pt1,pt2=[],[]
    for m in m1:
        for n in m2:
            if m.queryIdx==n.trainIdx:
                pt1.append(m.trainIdx)
                pt2.append(n.trainIdx)   # 便于提取点
                num_12.append(n.queryIdx)
                break
    m3=img_matches[i+2]
    pt3,pt4=[],[]
    for m in m2:
        for n in m3:
            if m.queryIdx==n.trainIdx:
                # print(m.queryIdx)
                pt3.append(m.queryIdx)
                pt4.append(n.queryIdx)
                num_23.append(m.queryIdx)
                break
    ptall=0
    npt1,npt2,npt3,npt4=[],[],[],[]
    for p in range(len(num_12)):
        for q in range(len(num_23)):
            if num_12[p]==num_23[q]:
                npt1.append(pt1[p])
                npt2.append(pt2[p])
                npt3.append(pt3[q])
                npt4.append(pt4[q])
                ptall+=1
                break
    if len(npt1)==len(npt2)==len(npt3)==len(npt4):
        print("finish and {:d}pts!".format(len(npt1)))
    maxright=np.float32([imgs[i].sift_keypt[m].pt for m in npt1]).reshape(-1, 1, 2)
    middleright=np.float32([imgs[i+1].sift_keypt[m].pt for m in npt2]).reshape(-1, 1, 2)
    middleleft=np.float32([imgs[i+2].sift_keypt[m].pt for m in npt3]).reshape(-1, 1, 2)
    maxleft=np.float32([imgs[i+3].sift_keypt[m].pt for m in npt4]).reshape(-1, 1, 2)
    # 绘制keypoint
    Dmaxright=[imgs[i].sift_keypt[m] for m in npt1]
    Dmiddleright=[imgs[i+1].sift_keypt[m] for m in npt2]
    Dmiddleleft=[imgs[i+2].sift_keypt[m] for m in npt3]
    Dmaxleft=[imgs[i+3].sift_keypt[m] for m in npt4]

    control_maxRpt[i]=maxright
    control_middleRpt[i]=middleright   # img002
    control_middleLpt[i]=middleleft   # img003
    control_maxLpt[i]=maxleft
'''
draw1=imgs[0].img
draw2=imgs[1].img
draw3=imgs[2].img
draw4=imgs[3].img

cv2.drawKeypoints(imgs[0].img,Dmaxright,draw1)
cv2.drawKeypoints(imgs[1].img,Dmiddleright,draw2)
cv2.drawKeypoints(imgs[2].img,Dmiddleleft,draw3)
cv2.drawKeypoints(imgs[3].img,Dmaxleft,draw4)
cv2.imwrite("./data/program/Bundle adjustment/draw1.jpeg",draw1)
cv2.imwrite("./data/program/Bundle adjustment/draw2.jpeg",draw2)
cv2.imwrite("./data/program/Bundle adjustment/draw3.jpeg",draw3)
cv2.imwrite("./data/program/Bundle adjustment/draw4.jpeg",draw4)
'''




for i in range(num):
    if i==0:
        imgs[i].filter_pt1(img_matches[0],0)
    elif i==num-1:
        imgs[i].filter_pt1(img_matches[num-2],1)
    else:
        imgs[i].filter_pt2(img_matches[i-1],img_matches[i])

# print(imgs[0].normal_pts)
# print(imgs[1].normal_pts)
for i in range(num):
    print("{}img has {} normal pts".format(i+1,len(imgs[i].normal_pts)))

'''
Bundle Adjustment

'''

# 先计算摄站坐标
S_coor=np.zeros((num,3))
for i in range(num):
    Px=np.matrix([[0],[0],[0],[1]])
    Rt=np.dot(imgs[i].T_wb,imgs[i].T_bc)
    Pw=np.dot(Rt,np.dot(np.linalg.inv(imgs[i].K),Px))
    S_coor[i][0]=Pw[0][0]
    S_coor[i][1]=Pw[1][0]
    S_coor[i][2]=Pw[2][0]
print(S_coor)

# 计算控制点坐标

def cal_l(x,y,f,R,X,Y,Z,x0,y0):
    x0*=0.001
    y0*=0.001
    a1=R[0][0]
    a2=R[0][1]
    a3=R[0][2]
    b1=R[1][0]
    b2=R[1][1]
    b3=R[1][2]
    c1=R[2][0]
    c2=R[2][1]
    c3=R[2][2]
    l1=f*a1+(x)*a3
    l2=f*b1+x*b3
    l3=f*c1+x*c3
    lx=f*a1*X+f*b1*Y+f*c1*Z+x*a3*X+x*b3*Y+x*c3*Z
    l4=f*a2+y*a3
    l5=f*b2+y*b3
    l6=f*c2+y*c3
    ly=f*a2*X+f*b2*Y+f*c2*Z+y*a3*X+y*b3*Y+y*c3*Z
    Ax=np.zeros((2,3))
    l=np.zeros((2,1))
    Ax[0][0]=l1
    Ax[0][1]=l2
    Ax[0][2]=l3
    l[0][0]=lx
    Ax[1][0]=l4
    Ax[1][1]=l5
    Ax[1][2]=l6
    l[1][0]=ly
    return Ax,l

'''
# 设置存放控制点坐标的数组
control_pts=list(range(num-1))

for i in range(num-1):
    # print("the {} pts:".format(i+1))
    # print(len(control_Lpt[i]))
    control_pts[i]=list(range(len(control_Lpt[i])))
    print("the {} pair has {} pts".format(i+1,len(control_Lpt[i])))
    for j in range(1):  #len(control_Lpt[i])

        Px=np.matrix([[control_Rpt[i][j][0][1]],[control_Rpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i].focal_length*0.001
        Px[1][0]*=imgs[i].focal_length*0.001
        Px[2][0]*=imgs[i].focal_length*0.001
        Rtr=np.dot(imgs[i].T_wb,imgs[i].T_bc)
        # Rtr=imgs[i].T_bc
        PcR=np.dot(np.linalg.inv(imgs[i].K),Px)
        PcR[2][0]=-PcR[2][0]
        # print(PcR)
        PwR=np.dot(Rtr,PcR)
        # print(PwR)

        Px=np.matrix([[control_Lpt[i][j][0][1]],[control_Lpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i+1].focal_length*0.001
        Px[1][0]*=imgs[i+1].focal_length*0.001
        Px[2][0]*=imgs[i+1].focal_length*0.001
        Rtl=np.dot(imgs[i+1].T_wb,imgs[i+1].T_bc)
        # Rtl=imgs[i+1].T_bc
        PcL=np.dot(np.linalg.inv(imgs[i+1].K),Px)
        PcL[2][0]=-PcL[2][0]
        PwL=np.dot(Rtl,PcL)
        # print(PwL)
        
        # 利用共线方程严密求解控制点坐标(像辅坐标)
        Ax=np.zeros((4,3))
        l=np.zeros((4,1))
        # left
        x=PcL[0][0]
        y=PcL[1][0]
        f=-PcL[2][0]
        # XS=S_coor[i+1][0]
        # YS=S_coor[i+1][1]
        # ZS=S_coor[i+1][2]
        Px=np.matrix([[0],[0],[0],[1]])
        L_left=np.dot(imgs[i+1].T_bc,np.dot(np.linalg.inv(imgs[i+1].K),Px))
        L_right=np.dot(imgs[i].T_bc,np.dot(np.linalg.inv(imgs[i].K),Px))
        Xs=L_left[0][0]
        Ys=L_left[1][0]
        Zs=L_left[2][0]
        x0=imgs[i+1].x0
        y0=imgs[i+1].y0
        Ax[0:2,],l[0:2,]=cal_l(x,y,f,Rtl,Xs,Ys,Zs,x0,y0)

        x=PcR[0][0]
        y=PcR[1][0]
        f=-PcR[2][0]
        # XS=S_coor[i][0]
        # YS=S_coor[i][1]
        # ZS=S_coor[i][2]
        Xs=L_right[0][0]
        Ys=L_right[1][0]
        Zs=L_right[2][0]
        x0=imgs[i].x0
        y0=imgs[i].y0
        Ax[2:4,],l[2:4,]=cal_l(x,y,f,Rtr,Xs,Ys,Zs,x0,y0)
        Ax=Ax[0:3,]
        l=l[0:3,]
        # print(Ax)
        # print(l)
        final_pw=np.dot(np.linalg.inv(Ax), l)
        # print(final_pw)
        control_pts[i][j]=final_pw
        # print(final_pw)
        # print("next point")
'''

# 计算四张影像连接点的坐标
control4_pts=list(range(num-3))
for i in range(num-3):
    control4_pts[i]=list(range(len(control_maxRpt[i])))
    print("the {} pair(all 4 imgs) has {} pts".format(i+1,len(control_maxRpt[i])))
    for j in range(len(control_maxRpt[i])):   # len(control_maxRpt[i])
        Px=np.matrix([[control_maxRpt[i][j][0][1]],[control_maxRpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i].focal_length*0.001
        Px[1][0]*=imgs[i].focal_length*0.001
        Px[2][0]*=imgs[i].focal_length*0.001
        # Rtr=np.dot(imgs[i].T_wb,imgs[i].T_bc)
        maxRtr=imgs[i].T_bc
        PcRmax=np.dot(np.linalg.inv(imgs[i].K),Px)
        PcRmax[2][0]=-PcRmax[2][0]
        # print(PcR)
        PwRmax=np.dot(maxRtr,PcRmax)
        # print(PwRmax)

        Px=np.matrix([[control_middleRpt[i][j][0][1]],[control_middleRpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i+1].focal_length*0.001
        Px[1][0]*=imgs[i+1].focal_length*0.001
        Px[2][0]*=imgs[i+1].focal_length*0.001
        # Rtr=np.dot(imgs[i+1].T_wb,imgs[i+1].T_bc)
        middleRtr=imgs[i+1].T_bc
        PcRmiddle=np.dot(np.linalg.inv(imgs[i+1].K),Px)
        PcRmiddle[2][0]=-PcRmiddle[2][0]
        # print(PcR)
        PwRmiddle=np.dot(middleRtr,PcRmiddle)
        # print(PwRmiddle)

        Px=np.matrix([[control_middleLpt[i][j][0][1]],[control_middleLpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i+2].focal_length*0.001
        Px[1][0]*=imgs[i+2].focal_length*0.001
        Px[2][0]*=imgs[i+2].focal_length*0.001
        # Rtr=np.dot(imgs[i+2].T_wb,imgs[i+2].T_bc)
        middleLtr=imgs[i+2].T_bc
        PcLmiddle=np.dot(np.linalg.inv(imgs[i+2].K),Px)
        PcLmiddle[2][0]=-PcLmiddle[2][0]
        # print(PcR)
        PwLmiddle=np.dot(middleLtr,PcLmiddle)
        # print(PwLmiddle)

        Px=np.matrix([[control_maxLpt[i][j][0][1]],[control_maxLpt[i][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[i+3].focal_length*0.001
        Px[1][0]*=imgs[i+3].focal_length*0.001
        Px[2][0]*=imgs[i+3].focal_length*0.001
        # Rtr=np.dot(imgs[i+3].T_wb,imgs[i+3].T_bc)
        maxLtr=imgs[i+3].T_bc
        PcLmax=np.dot(np.linalg.inv(imgs[i+3].K),Px)
        PcLmax[2][0]=-PcLmax[2][0]
        # print(PcR)
        PwLmax=np.dot(maxLtr,PcLmax)
        # print(PwLmax)

        # print("next")

        # 利用平均值求解控制点坐标(像辅坐标rover system)
        Xs=0.25*(PwRmax[0,0]+PwRmiddle[0,0]+PwLmiddle[0,0]+PwLmax[0,0])
        Ys=0.25*(PwRmax[1,0]+PwRmiddle[1,0]+PwLmiddle[1,0]+PwLmax[1,0])
        Zs=0.25*(PwRmax[2,0]+PwRmiddle[2,0]+PwLmiddle[2,0]+PwLmax[2,0])
        final_S=np.array([[Xs],[Ys],[Zs]])
        control4_pts[i][j]=final_S



'''
###  bundle adjustment
        fai: roll
        omg: pitch
        kei: yaw

'''

#计算参数近似值
def CalCSjs(fai,omg,k):
    a1=math.cos(fai)*math.cos(k)-math.sin(fai)*math.sin(omg)*math.sin(k)
    a2=-math.cos(fai)*math.sin(k)-math.sin(fai)*math.sin(omg)*math.cos(k)
    a3=-math.sin(fai)*math.cos(omg)
    b1=math.cos(omg)*math.sin(k)
    b2=math.cos(omg)*math.cos(k)
    b3=-math.sin(omg)
    c1=math.sin(fai)*math.cos(k)+math.cos(fai)*math.sin(omg)*math.sin(k)
    c2=-math.sin(fai)*math.sin(k)+math.cos(fai)*math.sin(omg)*math.cos(k)
    c3=math.cos(fai)*math.cos(omg)
    R_x=np.eye(3,3)
    R_y=np.eye(3,3)
    R_z=np.eye(3,3)
    R_x[1][1]=math.cos(omg) 
    R_x[1][2]=-math.sin(omg) 
    R_x[2][1]=math.sin(omg) 
    R_x[2][2]=math.cos(omg) 
    R_y[0][0]=math.cos(fai) 
    R_y[0][2]=math.sin(fai) 
    R_y[2][0]=-math.sin(fai) 
    R_y[2][2]=math.cos(fai)
    R_z[0][0]=math.cos(k) 
    R_z[0][1]=-math.sin(k) 
    R_z[1][0]=math.sin(k) 
    R_z[1][1]=math.cos(k)
    RR=np.dot(np.dot(R_y,R_x),R_z)
    a1=RR[0,0]
    a2=RR[0,1]
    a3=RR[0,2]
    b1=RR[1,0]
    b2=RR[1,1]
    b3=RR[1,2]
    c1=RR[2,0]
    c2=RR[2,1]
    c3=RR[2,2]

    return a1,a2,a3,b1,b2,b3,c1,c2,c3

# 计算系数用到的XYZ杠
def cal_xyz_(a1,a2,a3,b1,b2,b3,c1,c2,c3,xa,ya,za,xs,ys,zs):
    X_=a1*(xa-xs) + b1*(ya-ys) + c1*(za-zs)
    Y_=a2*(xa-xs) + b2*(ya-ys) + c2*(za-zs)
    Z_=a3*(xa-xs) + b3*(ya-ys) + c3*(za-zs)
    return X_,Y_,Z_

# 计算误差方程式系数
def cal_a(fai,omg,kei,Xa,Ya,Za,Xs,Ys,Zs,x0,y0,f,Pc):
    a1,a2,a3,b1,b2,b3,c1,c2,c3=CalCSjs(fai,omg,kei)
    X_,Y_,Z_=cal_xyz_(a1,a2,a3,b1,b2,b3,c1,c2,c3,Xa,Ya,Za,Xs,Ys,Zs)
    x=Pc[0][0]
    y=Pc[1][0]
    z=Pc[2][0]
    a11=(a1*f+a3*(x-x0))/Z_
    a12=(b1*f+b3*(x-x0))/Z_
    a13=(c1*f+c3*(x-x0))/Z_
    a21=(a2*f+a3*(y-y0))/Z_
    a22=(b2*f+b3*(y-y0))/Z_
    a23=(c2*f+c3*(y-y0))/Z_
    a14=(y-y0)*math.sin(omg) - math.cos(omg)*( f*math.cos(kei)+ (x-x0)/f *((x-x0)*math.cos(kei)-(y-y0)*math.sin(kei)) )
    a15=-f*math.sin(kei)-(x-x0)/f * ((x-x0)*math.sin(kei)+(y-y0)*math.cos(kei))
    a16=y-y0
    a24=-(x-x0)*math.sin(omg) - math.cos(omg)*(-f*math.sin(kei)+ (y-y0)/f *((x-x0)*math.cos(kei)-(y-y0)*math.sin(kei)) )
    a25=-f*math.cos(kei)-(y-y0)/f * ((x-x0)*math.sin(kei)+(y-y0)*math.cos(kei))
    a26=-(x-x0)
    a17= (x-x0)/f
    a27=(y-y0)/f
    a18,a29=1,1
    a19,a28=0,0
    return a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29
    
def CalxyJS(fai,omg,k,Xa,Ya,Za,EX,EY,EH,x0,y0,f):
    a1,a2,a3,b1,b2,b3,c1,c2,c3=CalCSjs(fai,omg,k)
    xjs=-f*((a1*(Xa-EX)+b1*(Ya-EY)+c1*(Za-EH))/(a3*(Xa-EX)+b3*(Ya-EY)+c3*(Za-EH)))+x0
    yjs=-f*((a2*(Xa-EX)+b2*(Ya-EY)+c2*(Za-EH))/(a3*(Xa-EX)+b3*(Ya-EY)+c3*(Za-EH)))+y0
    zba=a3*(Xa-EX)+b3*(Xa-EY)+c3*(Za-EH)
    return xjs,yjs


### 先拿四张图像试一下
### 1 2 3 4 1pair 10个点 1个点8个方程 (1234*xy)->80行
### A1 A2 A3 A4 x1...X10 => 6*4+10*3=54列

A_all=np.zeros((80,54))
l_all=np.zeros((80,1))
P_all=np.eye(80,80)

# 计算摄站的ROVER SYSTEM 坐标
Px=np.matrix([[0],[0],[0],[1]])
maxL=np.dot(imgs[3].T_bc,np.dot(np.linalg.inv(imgs[3].K),Px))
middleL=np.dot(imgs[2].T_bc,np.dot(np.linalg.inv(imgs[2].K),Px))
middleR=np.dot(imgs[1].T_bc,np.dot(np.linalg.inv(imgs[1].K),Px))
maxR=np.dot(imgs[0].T_bc,np.dot(np.linalg.inv(imgs[0].K),Px))

maxLXs=maxL[0,0]
maxLYs=maxL[1,0]
maxLZs=maxL[2,0]
middleLXs=middleL[0,0]
middleLYs=middleL[1,0]
middleLZs=middleL[2,0]
middleRXs=middleR[0,0]
middleRYs=middleR[1,0]
middleRZs=middleR[2,0]
maxRXs=maxR[0,0]
maxRYs=maxR[1,0]
maxRZs=maxR[2,0]

# 得到角元素
maxL_fai=imgs[3].bc_fai
maxL_omg=imgs[3].bc_omg
maxL_kei=imgs[3].bc_kei
middleL_fai=imgs[2].bc_fai
middleL_omg=imgs[2].bc_omg
middleL_kei=imgs[2].bc_kei
middleR_fai=imgs[1].bc_fai
middleR_omg=imgs[1].bc_omg
middleR_kei=imgs[1].bc_kei
maxR_fai=imgs[0].bc_fai
maxR_omg=imgs[0].bc_omg
maxR_kei=imgs[0].bc_kei
# 得到内方
maxL_x0=imgs[3].x0
maxL_y0=imgs[3].y0
maxL_f=imgs[3].focal_length*0.001
middleL_x0=imgs[2].x0
middleL_y0=imgs[2].y0
middleL_f=imgs[2].focal_length*0.001
middleR_x0=imgs[1].x0
middleR_y0=imgs[1].y0
middleR_f=imgs[1].focal_length*0.001
maxR_x0=imgs[0].x0
maxR_y0=imgs[0].y0
maxR_f=imgs[0].focal_length*0.001

# 补全误差方程式的系数阵
def give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,a_all,imgnum:int, pt_num:int, img_pt:int):
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+0]=a11
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+1]=a12
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+2]=a13
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+3]=a14
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+4]=a15
    a_all[2*img_pt*imgnum+pt_num*2][6*imgnum+5]=a16
    a_all[2*img_pt*imgnum+pt_num*2][6*4+3*pt_num]=-a11
    a_all[2*img_pt*imgnum+pt_num*2][6*4+3*pt_num+1]=-a12
    a_all[2*img_pt*imgnum+pt_num*2][6*4+3*pt_num+2]=-a13
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+0]=a21
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+1]=a22
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+2]=a23
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+3]=a24
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+4]=a25
    a_all[2*img_pt*imgnum+pt_num*2+1][6*imgnum+5]=a26
    a_all[2*img_pt*imgnum+pt_num*2+1][6*4+3*pt_num]=-a21
    a_all[2*img_pt*imgnum+pt_num*2+1][6*4+3*pt_num+1]=-a22
    a_all[2*img_pt*imgnum+pt_num*2+1][6*4+3*pt_num+2]=-a23

# 补全系数阵
def give_l(imgnum:int, pt_num:int, img_pt:int, l_all, x_gx,y_gx, x_l,y_l):
    l_all[2*imgnum*img_pt+2*pt_num][0] = x_l-x_gx
    l_all[2*imgnum*img_pt+2*pt_num+1][0] = y_l-y_gx


# 生成系数阵
for j in range(len(control4_pts[0])):
    # 均值得到的初值
    Xa=control4_pts[0][j][0,0]
    Ya=control4_pts[0][j][1,0]
    Za=control4_pts[0][j][2,0]

    Px=np.matrix([[control_maxLpt[0][j][0][1]],[control_maxLpt[0][j][0][0]],[1],[1]])
    Px[0][0]*=imgs[3].focal_length*0.001
    Px[1][0]*=imgs[3].focal_length*0.001
    Px[2][0]*=imgs[3].focal_length*0.001
    PcLmax=np.dot(np.linalg.inv(imgs[3].K),Px)
    PcLmax[2][0]=-PcLmax[2][0]

    # leftmax x y 
    a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(maxL_fai,maxL_omg,maxL_kei,
                Xa,Ya,Za,maxLXs,maxLYs,maxLZs,maxL_x0,maxL_y0,maxL_f,PcLmax)
    x_gx,y_gx=CalxyJS(maxL_fai,maxL_omg,maxL_kei,Xa,Ya,Za,
                        maxLXs,maxLYs,maxLZs,maxL_x0,maxL_y0,maxL_f)
    x_l=PcLmax[0,0]
    y_l=PcLmax[1,0]
    give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 3, j, len(control4_pts[0]))
    give_l(3,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)


    Px=np.matrix([[control_middleLpt[0][j][0][1]],[control_middleLpt[0][j][0][0]],[1],[1]])
    Px[0][0]*=imgs[2].focal_length*0.001
    Px[1][0]*=imgs[2].focal_length*0.001
    Px[2][0]*=imgs[2].focal_length*0.001
    PcLmiddle=np.dot(np.linalg.inv(imgs[2].K),Px)
    PcLmiddle[2][0]=-PcLmiddle[2][0]
    a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(middleL_fai,middleL_omg,middleL_kei,
                Xa,Ya,Za,middleLXs,middleLYs,middleLZs,middleL_x0,middleL_y0,middleL_f,PcLmiddle)
    x_gx,y_gx=CalxyJS(middleL_fai,middleL_omg,middleL_kei,Xa,Ya,Za,
                        middleLXs,middleLYs,middleLZs,middleL_x0,middleL_y0,middleL_f)
    x_l=PcLmiddle[0,0]
    y_l=PcLmiddle[1,0]
    give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 2, j, len(control4_pts[0]))
    give_l(2,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

    Px=np.matrix([[control_middleRpt[0][j][0][1]],[control_middleRpt[0][j][0][0]],[1],[1]])
    Px[0][0]*=imgs[1].focal_length*0.001
    Px[1][0]*=imgs[1].focal_length*0.001
    Px[2][0]*=imgs[1].focal_length*0.001
    PcRmiddle=np.dot(np.linalg.inv(imgs[1].K),Px)
    PcRmiddle[2][0]=-PcRmiddle[2][0]
    a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(middleR_fai,middleR_omg,middleR_kei,
                Xa,Ya,Za,middleRXs,middleRYs,middleRZs,middleR_x0,middleR_y0,middleR_f,PcRmiddle)
    x_gx,y_gx=CalxyJS(middleR_fai,middleR_omg,middleR_kei,Xa,Ya,Za,
                        middleRXs,middleRYs,middleRZs,middleR_x0,middleR_y0,middleR_f)
    x_l=PcRmiddle[0,0]
    y_l=PcRmiddle[1,0]
    give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 1, j, len(control4_pts[0]))
    give_l(1,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

    Px=np.matrix([[control_maxRpt[0][j][0][1]],[control_maxRpt[0][j][0][0]],[1],[1]])
    Px[0][0]*=imgs[i].focal_length*0.001
    Px[1][0]*=imgs[i].focal_length*0.001
    Px[2][0]*=imgs[i].focal_length*0.001
    PcRmax=np.dot(np.linalg.inv(imgs[0].K),Px)
    PcRmax[2][0]=-PcRmax[2][0]
    a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(maxR_fai,maxR_omg,maxR_kei,
                Xa,Ya,Za,maxRXs,maxRYs,maxRZs,maxR_x0,maxR_y0,maxR_f,PcRmax)
    x_gx,y_gx=CalxyJS(maxR_fai,maxR_omg,maxR_kei,Xa,Ya,Za,
                        maxRXs,maxRYs,maxRZs,maxR_x0,maxR_y0,maxR_f)
    x_l=PcRmax[0,0]
    y_l=PcRmax[1,0]
    give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 0, j, len(control4_pts[0]))
    give_l(0,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

# print(l_all)
N = np.dot(np.dot(A_all.T,P_all),A_all)
APL = np.dot(np.dot(A_all.T,P_all),l_all)
U, S, V=np.linalg.svd(N)

# 设置奇异值截取的范围
t=0
nSt=[]
for va in range(S.size):
    if S[va]>1:
        t+=1
        nSt.append(S[va])
#print(t)
St=np.eye(t,t)
for va in range(t):
    St[va,va]=nSt[va]

Ut=U[:,0:t]
Vt=U[:,0:t]
fr=np.dot(np.dot(Vt,np.linalg.inv(St)),Ut.T)
final_X=np.dot(fr,APL)
#print(final_X)
#print(final_X.size)


'''
bundle adjustment final circulation
'''
### 设置初值
BCmaxR=[ imgs[0].x_c,imgs[0].y_c,imgs[0].z_c, imgs[0].bc_fai,imgs[0].bc_omg,imgs[0].bc_kei]
BCmiddleR=[imgs[1].x_c,imgs[1].y_c,imgs[1].z_c, imgs[1].bc_fai,imgs[1].bc_omg,imgs[1].bc_kei]
BCmiddleL=[ imgs[2].x_c,imgs[2].y_c,imgs[2].z_c, imgs[2].bc_fai,imgs[2].bc_omg,imgs[2].bc_kei]
BCmaxL=[imgs[3].x_c,imgs[3].y_c,imgs[3].z_c, imgs[3].bc_fai,imgs[3].bc_omg,imgs[3].bc_kei]
## 根据外方构建旋转阵
def cal_Rbc(BC:list):
    Afai=BC[3]
    Aomg=BC[4]
    Akei=BC[5]
    tX=BC[0]
    tY=BC[1]
    tZ=BC[2]
    '''
    a1=math.cos(Afai)*math.cos(Akei) + math.sin(Afai)*math.sin(Aomg)*math.sin(Akei)
    a2=-math.cos(Afai)*math.sin(Akei) + math.sin(Afai)*math.sin(Aomg)*math.cos(Akei)
    a3= math.sin(Afai)*math.cos(Aomg)
    b1=math.cos(Aomg)*math.sin(Akei)
    b2=math.cos(Aomg)*math.cos(Akei)
    b3=-math.sin(Aomg)
    c1=-math.sin(Afai)*math.cos(Akei) + math.cos(Afai)*math.sin(Aomg)*math.sin(Akei)
    c2=math.sin(Afai)*math.sin(Akei) + math.cos(Afai)*math.sin(Aomg)*math.cos(Akei)
    c3=math.cos(Afai)*math.cos(Aomg)
    '''
    R_x=np.eye(3,3)
    R_y=np.eye(3,3)
    R_z=np.eye(3,3)
    R_x[1][1]=math.cos(Aomg) 
    R_x[1][2]=-math.sin(Aomg) 
    R_x[2][1]=math.sin(Aomg) 
    R_x[2][2]=math.cos(Aomg) 
    R_y[0][0]=math.cos(Afai) 
    R_y[0][2]=math.sin(Afai) 
    R_y[2][0]=-math.sin(Afai) 
    R_y[2][2]=math.cos(Afai)
    R_z[0][0]=math.cos(Akei) 
    R_z[0][1]=-math.sin(Akei) 
    R_z[1][0]=math.sin(Akei) 
    R_z[1][1]=math.cos(Akei)
    RR=np.dot(np.dot(R_y,R_x),R_z)
    a1=RR[0,0]
    a2=RR[0,1]
    a3=RR[0,2]
    b1=RR[1,0]
    b2=RR[1,1]
    b3=RR[1,2]
    c1=RR[2,0]
    c2=RR[2,1]
    c3=RR[2,2]
    R=np.array([[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]])
    Rbc=np.eye(4,4)
    Rbc[0:3,0:3]=R
    Rbc[0][3]=tX
    Rbc[1][3]=tY
    Rbc[2][3]=tZ
    return Rbc

# test=cal_Rbc(BCmaxR)
# print(test)
# print(imgs[0].T_bc)
# 坐标初值 需要遍历 len(control4_pts[0])
# Xa,Ya,Za=control4_pts[0][j][0,0],control4_pts[0][j][1,0],control4_pts[0][j][2,0]

# 内方不变
maxL_x0=imgs[3].x0
maxL_y0=imgs[3].y0
maxL_f=imgs[3].focal_length*0.001
middleL_x0=imgs[2].x0
middleL_y0=imgs[2].y0
middleL_f=imgs[2].focal_length*0.001
middleR_x0=imgs[1].x0
middleR_y0=imgs[1].y0
middleR_f=imgs[1].focal_length*0.001
maxR_x0=imgs[0].x0
maxR_y0=imgs[0].y0
maxR_f=imgs[0].focal_length*0.001

# 设置新迭代中新的外方矩阵
new_BCmaxR=[imgs[0].x_c,imgs[0].y_c,imgs[0].z_c, imgs[0].bc_fai,imgs[0].bc_omg,imgs[0].bc_kei]
new_BCmiddleR=[imgs[1].x_c,imgs[1].y_c,imgs[1].z_c, imgs[1].bc_fai,imgs[1].bc_omg,imgs[1].bc_kei ]
new_BCmiddleL=[imgs[2].x_c,imgs[2].y_c,imgs[2].z_c, imgs[2].bc_fai,imgs[2].bc_omg,imgs[2].bc_kei]
new_BCmaxL=[imgs[3].x_c,imgs[3].y_c,imgs[3].z_c, imgs[3].bc_fai,imgs[3].bc_omg,imgs[3].bc_kei ]

# 设置新的物方坐标矩阵
nXYZa=list(range(len(control4_pts[0])))
for i in range(len(nXYZa)):
    nXYZa[i]=[control4_pts[0][i][0,0],control4_pts[0][i][1,0],control4_pts[0][i][2,0]]


# 判断限差是否满足条件
def judge_dx(A1:list,A2:list,A3:list,A4:list,epAngle,epXYZ):
    item_num=len(A1)
    result=[0,0,0,0]
    if A1[3]< epAngle and A1[4]< epAngle and A1[5]<epAngle and A1[0]<epXYZ and A1[1]<epXYZ and A1[2]<epXYZ:
        result[0]=1
    if A2[3]<epAngle and A2[5]<epAngle and A2[5]<epAngle and A2[0]<epXYZ and A2[1]<epXYZ and A2[2]<epXYZ:
        result[1]=1
    if A3[3]<epAngle and A3[5]<epAngle and A3[5]<epAngle and A3[0]<epXYZ and A3[1]<epXYZ and A3[2]<epXYZ:
        result[2]=1
    if A4[3]<epAngle and A4[5]<epAngle and A4[5]<epAngle and A4[0]<epXYZ and A4[1]<epXYZ and A4[2]<epXYZ:
        result[3]=1
    if result[0] and result[1] and result[2] and result[3]:
        return  True
    else:
        return  False


# 存放外放限差的list(先只判断外方的改正值)
deA=[0,0,0,0]
for i in range(4):
    deA[i]=list(range(len(control4_pts[0])))

# 设置循环次数
cir=0
while True:
    A_all=np.zeros((80,54))
    l_all=np.zeros((80,1))
    P_all=np.eye(80,80)

    # 计算摄站的ROVER SYSTEM 坐标
    Px=np.matrix([[0],[0],[0],[1]])
    S_maxL=np.dot(np.linalg.inv(imgs[3].K),Px)
    S_middleL=np.dot(np.linalg.inv(imgs[2].K),Px)
    S_middleR=np.dot(np.linalg.inv(imgs[1].K),Px)
    S_maxR=np.dot(np.linalg.inv(imgs[0].K),Px)

    RBC_maxL=cal_Rbc(new_BCmaxL)
    RBC_middleL=cal_Rbc(new_BCmiddleL)
    RBC_middleR=cal_Rbc(new_BCmiddleR)
    RBC_maxR=cal_Rbc(new_BCmaxR)


    maxL=np.dot(RBC_maxL,S_maxL)
    middleL=np.dot(RBC_middleL,S_middleL)
    middleR=np.dot(RBC_middleR,S_middleR)
    maxR=np.dot(RBC_maxR,S_maxR)

    maxLXs=maxL[0,0]
    maxLYs=maxL[1,0]
    maxLZs=maxL[2,0]
    middleLXs=middleL[0,0]
    middleLYs=middleL[1,0]
    middleLZs=middleL[2,0]
    middleRXs=middleR[0,0]
    middleRYs=middleR[1,0]
    middleRZs=middleR[2,0]
    maxRXs=maxR[0,0]
    maxRYs=maxR[1,0]
    maxRZs=maxR[2,0]

   
    # 生成系数阵
    for j in range(len(control4_pts[0])):
    # 均值得到的初值
        #Xa=control4_pts[0][j][0,0]
        #Ya=control4_pts[0][j][1,0]
        #Za=control4_pts[0][j][2,0]
        Xa=nXYZa[j][0]
        Ya=nXYZa[j][1]
        Za=nXYZa[j][2]

        # leftmax x y 
        Px=np.matrix([[control_maxLpt[0][j][0][1]],[control_maxLpt[0][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[3].focal_length*0.001
        Px[1][0]*=imgs[3].focal_length*0.001
        Px[2][0]*=imgs[3].focal_length*0.001
        PcLmax=np.dot(np.linalg.inv(imgs[3].K),Px)
        PcLmax[2][0]=-PcLmax[2][0]
        a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(new_BCmaxL[3],new_BCmaxL[4],new_BCmaxL[5],
                Xa,Ya,Za,maxLXs,maxLYs,maxLZs,maxL_x0,maxL_y0,maxL_f,PcLmax)
        x_gx,y_gx=CalxyJS(maxL_fai,maxL_omg,maxL_kei,Xa,Ya,Za,
                        maxLXs,maxLYs,maxLZs,maxL_x0,maxL_y0,maxL_f)
        x_l=PcLmax[0,0]
        y_l=PcLmax[1,0]
        give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 3, j, len(control4_pts[0]))
        give_l(3,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

        # leftmiddle x y 
        Px=np.matrix([[control_middleLpt[0][j][0][1]],[control_middleLpt[0][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[2].focal_length*0.001
        Px[1][0]*=imgs[2].focal_length*0.001
        Px[2][0]*=imgs[2].focal_length*0.001
        PcLmiddle=np.dot(np.linalg.inv(imgs[2].K),Px)
        PcLmiddle[2][0]=-PcLmiddle[2][0]
        a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(new_BCmiddleL[3],new_BCmiddleL[4],new_BCmiddleL[5],
                Xa,Ya,Za,middleLXs,middleLYs,middleLZs,middleL_x0,middleL_y0,middleL_f,PcLmiddle)
        x_gx,y_gx=CalxyJS(middleL_fai,middleL_omg,middleL_kei,Xa,Ya,Za,
                        middleLXs,middleLYs,middleLZs,middleL_x0,middleL_y0,middleL_f)
        x_l=PcLmiddle[0,0]
        y_l=PcLmiddle[1,0]
        give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 2, j, len(control4_pts[0]))
        give_l(2,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

        # rightmiddle x y 
        Px=np.matrix([[control_middleRpt[0][j][0][1]],[control_middleRpt[0][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[1].focal_length*0.001
        Px[1][0]*=imgs[1].focal_length*0.001
        Px[2][0]*=imgs[1].focal_length*0.001
        PcRmiddle=np.dot(np.linalg.inv(imgs[1].K),Px)
        PcRmiddle[2][0]=-PcRmiddle[2][0]
        a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(new_BCmiddleR[3],new_BCmiddleR[4],new_BCmiddleR[5],
                Xa,Ya,Za,middleRXs,middleRYs,middleRZs,middleR_x0,middleR_y0,middleR_f,PcRmiddle)
        x_gx,y_gx=CalxyJS(middleR_fai,middleR_omg,middleR_kei,Xa,Ya,Za,
                        middleRXs,middleRYs,middleRZs,middleR_x0,middleR_y0,middleR_f)
        x_l=PcRmiddle[0,0]
        y_l=PcRmiddle[1,0]
        give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 1, j, len(control4_pts[0]))
        give_l(1,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)

        # rightmax x y 
        Px=np.matrix([[control_maxRpt[0][j][0][1]],[control_maxRpt[0][j][0][0]],[1],[1]])
        Px[0][0]*=imgs[0].focal_length*0.001
        Px[1][0]*=imgs[0].focal_length*0.001
        Px[2][0]*=imgs[0].focal_length*0.001
        PcRmax=np.dot(np.linalg.inv(imgs[0].K),Px)
        PcRmax[2][0]=-PcRmax[2][0]
        a11,a12,a13,a14,a15,a16,a17,a18,a19,a21,a22,a23,a24,a25,a26,a27,a28,a29=cal_a(new_BCmaxR[3],new_BCmaxR[4],new_BCmaxR[5],
                Xa,Ya,Za,maxRXs,maxRYs,maxRZs,maxR_x0,maxR_y0,maxR_f,PcRmax)
        x_gx,y_gx=CalxyJS(maxR_fai,maxR_omg,maxR_kei,Xa,Ya,Za,
                        maxRXs,maxRYs,maxRZs,maxR_x0,maxR_y0,maxR_f)
        x_l=PcRmax[0,0]
        y_l=PcRmax[1,0]
        give_a(a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26,A_all, 0, j, len(control4_pts[0]))
        give_l(0,j,len(control4_pts[0]),l_all,x_gx,y_gx,x_l,y_l)
    
    # 求解X
    N = np.dot(np.dot(A_all.T,P_all),A_all)
    APL = np.dot(np.dot(A_all.T,P_all),l_all)
    U, S, V=np.linalg.svd(N)

    # 设置奇异值截取的范围
    t=0
    nSt=[]
    for va in range(S.size):
        if S[va]>5:
            t+=1
            nSt.append(S[va])
    St=np.eye(t,t)
    for va in range(t):
        St[va,va]=nSt[va]

    Ut=U[:,0:t]
    Vt=U[:,0:t]
    fr=np.dot(np.dot(Vt,np.linalg.inv(St)),Ut.T)
    final_X=np.dot(fr,APL)
    print(final_X)
    for item in range(4):
        deA[item][0]=final_X[6*item+0,0]
        deA[item][1]=final_X[6*item+1,0]
        deA[item][2]=final_X[6*item+2,0]
        deA[item][3]=final_X[6*item+3,0]
        deA[item][4]=final_X[6*item+4,0]
        deA[item][5]=final_X[6*item+5,0]
    
    # 写入改正值
    for item in range(len(control4_pts[0])):
        nXYZa[item][0]+=final_X[6*4+3*item+0,0]
        nXYZa[item][1]+=final_X[6*4+3*item+1,0]
        nXYZa[item][2]+=final_X[6*4+3*item+2,0]
    for item in range(6):
        new_BCmaxR[item]+=deA[0][item]
        new_BCmiddleR[item]+=deA[1][item]
        new_BCmiddleL[item]+=deA[2][item]
        new_BCmaxL[item]+=deA[3][item]
    
    cir+=1
    if judge_dx(deA[0],deA[1],deA[2],deA[3],0.000001,0.001):
        break
    if cir>10:
        break

print("all {} circulations".format(cir))
print("after adjustment:")
print(new_BCmaxR)
img001=[imgs[0].x_c, imgs[0].y_c,imgs[0].z_c, imgs[0].bc_fai,imgs[0].bc_omg,imgs[0].bc_kei]
print("before adjustment:")
print(img001)
img002=[imgs[1].x_c, imgs[1].y_c,imgs[1].z_c, imgs[1].bc_fai,imgs[1].bc_omg,imgs[1].bc_kei]
print("after adjustment:")
print(new_BCmiddleR)
print("before adjustment:")
print(img002)








