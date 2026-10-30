import cv2,numpy as np
from PIL import Image
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg')
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(int)
H,S,V=hsv[...,0]*2,hsv[...,1]/255,hsv[...,2]/255
a=np.array(Image.open('sb_full.png'))[...,3]
inbowl=a>128
salmon=(H>=4)&(H<=30)&(S>0.45)&(V>0.45)
green=(H>=45)&(H<=110)&(S>0.30)&(V>0.30)
purple=((H>=260)|(H<=4))&(S>0.12)&(V>0.35)&(S<0.6)
roe=((H<=8)|(H>=340))&(S>0.6)&(V>0.35)
cream=(H>=15)&(H<=45)&(S>0.08)&(S<0.42)&(V>0.70)
white=(S<0.12)&(V>0.72)
col={1:(0,120,255),2:(0,200,0),3:(200,0,200),4:(0,0,255),5:(120,200,255),6:(255,255,255)}
out=np.zeros_like(im)
for k,m in [(6,white),(5,cream),(3,purple),(2,green),(1,salmon),(4,roe)]:
    out[m&inbowl]=col[k]
cv2.imwrite('cls.png',np.hstack([cv2.resize(im,(640,640)),cv2.resize(out,(640,640))]))
