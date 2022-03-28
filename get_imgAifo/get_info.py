from urllib import robotparser
from xml.dom.minidom import parse
import xml.dom.minidom
import math
import numpy as np

path_2cl="./data/ori_data/20210702031729_20210702031729_00048_A (1).2CL"
DOMTree = xml.dom.minidom.parse(path_2cl)
product = DOMTree.documentElement
names = product.getElementsByTagName("Source_Product_Internal")

for name in names:
    print(name.getElementsByTagName('lidvid_reference')[0].childNodes[0].data)


img_info=product.getElementsByTagName("Axis_Array")
for img in img_info:
    axis_name=img.getElementsByTagName('axis_name')[0].childNodes[0].data
    if axis_name=="Line":
        img_height=int(img.getElementsByTagName('elements')[0].childNodes[0].data)
    if axis_name=="Sample":
        img_weight=int(img.getElementsByTagName('elements')[0].childNodes[0].data)  
    if axis_name=="Band":
        img_bandNums=int(img.getElementsByTagName('elements')[0].childNodes[0].data)

print("height:{:d}  weight:{:d}  band:{:d}".format(img_height,img_weight,img_bandNums))

instru_params=product.getElementsByTagName("Instrument_Parm")
for param in instru_params:
    pixel_size=float(param.getElementsByTagName('pixel_size')[0].childNodes[0].data)*0.001
    focal_length=float(param.getElementsByTagName('focal_length')[0].childNodes[0].data)
    ppc=param.getElementsByTagName('Principal_Point_Coordinate')
    for xy in ppc:
        x0=float(xy.getElementsByTagName('x0')[0].childNodes[0].data)
        y0=float(xy.getElementsByTagName('y0')[0].childNodes[0].data)

print("pixel size:{:f}   f:{:f}  ".format(pixel_size,focal_length))
print("x0:{:f}   y0:{:f}  ".format(x0,y0))
# K矩阵
K=np.eye(4,4)
K[0][0]=focal_length/pixel_size
K[0][2]=img_weight/2+x0/pixel_size
K[1][1]=focal_length/pixel_size
K[1][2]=img_height/2-y0/pixel_size
print(K)

rover_atti=product.getElementsByTagName("Rover_Attitude")
for param in rover_atti:
    pitch_b=float(param.getElementsByTagName('pitch')[0].childNodes[0].data)
    roll_b=float(param.getElementsByTagName('roll')[0].childNodes[0].data)
    yaw_b=float(param.getElementsByTagName('yaw')[0].childNodes[0].data)
print("pitch_b:{:f}  roll_b:{:f}  yaw_b:{:f}".format(pitch_b,roll_b,yaw_b))

rover_xyz=product.getElementsByTagName("Rover_Location_xyz")
for param in rover_xyz:
    x_b=float(param.getElementsByTagName('x')[0].childNodes[0].data)
    y_b=float(param.getElementsByTagName('y')[0].childNodes[0].data)
    z_b=float(param.getElementsByTagName('z')[0].childNodes[0].data)
print("x_b:{:f}  y_b:{:f}  z_b:{:f}".format(x_b,y_b,z_b))

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
    R_y[0][2]=-math.sin(roll) 
    R_y[2][0]=-math.sin(roll) 
    R_y[2][2]=math.cos(roll)
    R_z[0][0]=math.cos(yaw) 
    R_z[0][1]=-math.sin(yaw) 
    R_z[1][0]=math.sin(yaw) 
    R_z[1][1]=math.cos(yaw)
    RR=np.dot(np.dot(R_x,R_y),R_z)
    return  RR

EOE=product.getElementsByTagName("Exterior_Orientation_Elements")
for param in EOE:
    x_c=float(param.getElementsByTagName('camera_center_position_x')[0].childNodes[0].data)
    y_c=float(param.getElementsByTagName('camera_center_position_y')[0].childNodes[0].data)
    z_c=float(param.getElementsByTagName('camera_center_position_z')[0].childNodes[0].data)
    pitch_c=float(param.getElementsByTagName('camera_rotation_angle_pitch')[0].childNodes[0].data)
    roll_c=float(param.getElementsByTagName('camera_rotation_angle_roll')[0].childNodes[0].data)
    yaw_c=float(param.getElementsByTagName('camera_rotation_angle_yaw')[0].childNodes[0].data)

print("x_c:{:f}  y_c:{:f}  z_c:{:f}".format(x_c,y_c,z_c))
print("pitch_c:{:f}  roll_c:{:f}  yaw_c:{:f}".format(pitch_c,roll_c,yaw_c))

def Rt_2_T(R,x,y,z):
    T=np.eye(4,4)
    T[0:3,0:3]=R
    T[0][3]=x
    T[1][3]=y
    T[2][3]=z
    return T

r_wb=Euler_2_R(pitch_b,roll_b,yaw_b)
T_wb=Rt_2_T(r_wb,x_b,y_b,z_b)
print(T_wb)

r_bc=Euler_2_R(pitch_c,roll_c,yaw_c)
T_bc=Rt_2_T(r_bc,x_c,y_c,z_c)
print(T_bc)