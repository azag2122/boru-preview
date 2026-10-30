import cv2,numpy as np
from PIL import Image
from scipy import ndimage as nd
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg')
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(int)
H,S,V=hsv[...,0]*2,hsv[...,1]/255,hsv[...,2]/255
yy,xx=np.mgrid[:1024,:1024]
A=np.array(Image.open('sb_full.png'))[...,3]
inb=(A>128)&(((xx-509)**2+(yy-506)**2)<360**2)
def clean(m,k=9,fill=True,minsz=400):
    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    if fill: m=nd.binary_fill_holes(m).astype(np.uint8)
    lab,n=nd.label(m); sz=nd.sum(m,lab,range(1,n+1))
    keep=np.isin(lab,[i+1 for i,s in enumerate(sz) if s>=minsz])
    return keep&inb
salmon=(H>=4)&(H<=32)&(S>0.42)&(V>0.40)
prot=clean(salmon&(yy<560),k=25)&~(((S<0.2)&(V>0.6))&(yy>380))
green=(H>=38)&(H<=115)&(S>0.22)&(V>0.28)
chop=(S<0.10)&(V>0.55)&(np.abs((xx-590)-(yy-75)*0.52)<40)  # rough chopstick band
purple=((H>=250)|(H<=6))&(S>0.10)&(V>0.33)&(S<0.65)&(xx>620)&(yy<640)&(yy>300)
rice=(S<0.2)&(V>0.6)
gr=clean(green&~rice,k=7,fill=False,minsz=300)
gr=clean(gr,k=5,fill=True,minsz=300)&~(rice&~green)
pu=clean(purple&~chop,k=7,fill=False,minsz=50)
greens=(gr|pu)&~prot
roe=((H<=10)|(H>=335))&(S>0.55)&(V>0.30)
sauce=(H>=8)&(H<=45)&(S>0.16)&(S<0.45)&(V>0.68)&(yy>560)
chives=(H>=60)&(H<=130)&(S>0.35)&(yy>420)&(xx>420)&(xx<700)
sc=clean(sauce|roe,k=15,minsz=200)
sc=(sc|clean(roe,k=5,minsz=10)|((chives)&~gr))&~greens&~prot
np.savez_compressed('masks.npz',prot=prot,greens=greens,sauce=sc)
vis=im.copy()
vis[prot>0]=(0,120,255);vis[greens>0]=(0,200,0);vis[sc>0]=(255,0,255)
cv2.imwrite('masks.png',cv2.resize(vis,(640,640)))
