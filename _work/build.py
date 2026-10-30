import cv2,numpy as np
from PIL import Image
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg')
M=np.load('masks.npz'); prot,greens,sauce=M['prot'],M['greens'],M['sauce']
A=np.array(Image.open('sb_full.png'))[...,3]
allm=(prot|greens|sauce).astype(np.uint8)
k=lambda n:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
hole=cv2.dilate(allm,k(9))
# low-res smooth fill
sm=cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA); hm=(cv2.resize(hole.astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
f=cv2.inpaint(sm,hm,6,cv2.INPAINT_TELEA)
f=cv2.GaussianBlur(cv2.resize(f,(1024,1024),interpolation=cv2.INTER_CUBIC),(0,0),6)
# texture: high-pass of bowl ring
ring=cv2.GaussianBlur(im,(0,0),4).astype(np.float32); hp=im.astype(np.float32)-ring
# sample texture from bottom-left bowl area tiled
tex=hp[700:900,180:380]
T=np.tile(tex,(6,6,1))[:1024,:1024]
fill=np.clip(f.astype(np.float32)+T*0.9,0,255).astype(np.uint8)
fine=cv2.inpaint(im,hole,7,cv2.INPAINT_TELEA)
# distance-based blend: near boundary use fine inpaint, deep use smooth fill
d=cv2.distanceTransform(hole,cv2.DIST_L2,5); w=np.clip((d-6)/18,0,1)[...,None]
base=np.where(hole[...,None]>0,(fine*(1-w)+fill*w),im).astype(np.uint8)
cv2.imwrite('base.png',base)
def rgba(img,mask,feather=1.2):
    a=(cv2.GaussianBlur(mask.astype(np.float32),(0,0),feather)*255).clip(0,255)
    a=np.minimum(a,A.astype(np.float32))
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.dstack([rgb,a.astype(np.uint8)]))
lay={'l0-rice':rgba(base,A>0,0.01),'l1-protein':rgba(im,prot),'l2-greens':rgba(im,greens),'l3-sauce':rgba(im,sauce)}
# base alpha = rembg alpha exactly
b=cv2.cvtColor(base,cv2.COLOR_BGR2RGB); lay['l0-rice']=Image.fromarray(np.dstack([b,A]))
for n,L in lay.items(): L.save(n+'.png')
# stage composites
canvas=Image.new('RGBA',(1024,1024),(9,24,47,255)); outs=[]
for n in ['l0-rice','l1-protein','l2-greens','l3-sauce']:
    canvas=Image.alpha_composite(canvas,lay[n]); outs.append(canvas.resize((512,512)))
s=Image.new('RGB',(1024,1024)); [s.paste(o,((i%2)*512,(i//2)*512)) for i,o in enumerate(outs)]; s.save('stages.png')
