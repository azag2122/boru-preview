import cv2,numpy as np
from PIL import Image
from scipy import ndimage as nd
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg')
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(int)
H,S,V=hsv[...,0]*2,hsv[...,1]/255,hsv[...,2]/255
yy,xx=np.mgrid[:1024,:1024]
A=np.array(Image.open('sb_full.png'))[...,3]
def poly(pts,dil=0):
    m=np.zeros((1024,1024),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],1)
    if dil: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dil,dil)))
    return m>0
def clean(m,k=5,fill=True,minsz=60):
    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    if fill:
        f=nd.binary_fill_holes(m); h=f&~(m>0); lab,n=nd.label(h)
        if n:
            sz=nd.sum(h,lab,range(1,n+1)); m=(m>0)|np.isin(lab,[i+1 for i,s in enumerate(sz) if s<250])
    lab,n=nd.label(m); sz=nd.sum(m,lab,range(1,n+1))
    return np.isin(lab,[i+1 for i,s in enumerate(sz) if s>=minsz])
dark=V<0.33
rice=(S<0.15)&(V>0.55)
band=xx-(625+0.53*(yy-200)); chop=(band>-12)&(band<48)&(S<0.14)&(yy<700)
greenc=(H>=38)&(H<=115)&(S>0.20)&(V>0.25)
SAL=[(235,445),(240,400),(275,360),(330,330),(400,295),(470,262),(560,252),(620,250),(640,285),(635,330),(650,355),(690,365),(700,395),(690,425),(650,440),(600,450),(560,455),(500,460),(460,475),(420,495),(380,510),(330,520),(310,500),(280,490),(245,470)]
prot=clean(poly(SAL,30)&~(greenc&(H>=42)&(H<=75)&(yy>460))&~dark&~chop&~(rice&(yy>470)),k=9)
AVO=[(205,505),(250,478),(310,482),(360,505),(390,540),(400,560),(420,600),(445,625),(445,660),(420,670),(380,690),(330,700),(270,690),(240,665),(220,620),(210,560)]
avo=clean(poly(AVO,24)&~dark&~rice&~prot,k=7)
ASH=[(345,540),(500,536),(600,538),(700,548),(772,566),(770,592),(700,600),(560,604),(400,604),(345,608)]
ASD=[(598,438),(622,428),(628,452),(602,522),(562,592),(522,652),(494,722),(470,740),(440,734),(448,690),(478,638),(518,578),(558,508)]
asp=clean((poly(ASH,16)|poly(ASD,16))&greenc&~prot,k=7)
ON=[(650,330),(700,345),(760,370),(810,410),(820,480),(815,560),(800,610),(760,620),(720,600),(680,560),(650,520),(640,460),(645,400)]
onion=clean(poly(ON,10)&~dark&~chop&~asp&~prot&~(greenc&(S>0.3)),k=3,fill=False,minsz=20)
greens=(avo|asp|onion)
SAU=[(370,690),(420,662),(470,680),(520,700),(600,690),(680,640),(720,580),(752,560),(760,620),(740,690),(700,750),(620,790),(520,805),(430,795),(375,760),(360,720)]
sauce_c=(S>0.15)&(V>0.6)&(H>=5)&(H<=45)
roe=((H<=12)|(H>=330))&(S>0.5)&(V>0.25)
chives=(H>=60)&(H<=140)&(S>0.30)&(V>0.15)
sau=clean(poly(SAU,20)&sauce_c&~greens,k=11)
top=clean((roe|chives)&poly([(440,560),(760,560),(760,800),(440,800)])&~greens,k=3,fill=True,minsz=8)
sauce=(sau|top)&~greens&~prot
prot&=~dark&~rice; greens&=~dark&~(rice&~onion); sauce&=~dark
L={'prot':prot,'greens':greens,'sauce':sauce}
np.savez_compressed('masks.npz',**L)
vis=im.copy()
vis[prot]=(0,120,255);vis[greens]=(0,200,0);vis[sauce]=(255,0,255)
cv2.imwrite('masks3.png',vis[180:880,150:880])
