import cv2,numpy as np
from PIL import Image
from scipy import ndimage as nd
exec(open('seg3.py').read().split("L={'prot'")[0])   # reuse masks & polygons
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg'); imf=im.astype(np.float32)
k=lambda n:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
cx,cy=509,506
food=(prot|greens|sauce)
RICE=[(425,478),(470,458),(560,452),(612,458),(640,478),(652,525),(660,600),(645,660),(605,700),(545,722),(480,726),(442,702),(422,640),(416,560)]
mound=poly(RICE)
mound=cv2.GaussianBlur(mound.astype(np.float32),(0,0),6)>0.5
chopm=cv2.dilate(((band>-14)&(band<50)&(S<0.16)&(V>0.30)&(yy<720)).astype(np.uint8),k(3))>0
R=cv2.dilate((food|poly(ON,24)|poly(SAL,24)|poly(AVO,24)|poly(SAU,24)|mound).astype(np.uint8),k(15))>0
R&=(A>128)
# polar model of empty bowl
r=np.hypot(xx-cx,yy-cy); th=(np.arctan2(yy-cy,xx-cx)+np.pi)/(2*np.pi)
NA,NR=96,int(400/3)
ai=(th*NA).astype(int)%NA; ri=np.clip((r/3).astype(int),0,NR-1)
ok=(A>200)&~R&~chopm
acc=np.zeros((NA,NR,3)); cnt=np.zeros((NA,NR))
np.add.at(acc,(ai[ok],ri[ok]),imf[ok]); np.add.at(cnt,(ai[ok],ri[ok]),1)
model=np.zeros((NA,NR,3)); have=cnt>3
model[have]=acc[have]/cnt[have][:,None]
# fill missing: nearest valid along angle, iterate
for it in range(200):
    miss=~have
    if not miss.any(): break
    nb=np.zeros_like(model); nc=np.zeros((NA,NR))
    for da,dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        sh=np.roll(model*have[...,None],(da,dr),(0,1)); sc=np.roll(have,(da,dr),(0,1))
        if dr==1: sh[:,0]=0;sc[:,0]=0
        if dr==-1: sh[:,-1]=0;sc[:,-1]=0
        nb+=sh;nc+=sc
    new=miss&(nc>0); model[new]=nb[new]/nc[new][:,None]; have=have|new
# smooth with wrap in angle
mp=np.concatenate([model[-8:],model,model[:8]]); mp=cv2.GaussianBlur(mp.astype(np.float32),(0,0),sigmaX=1.5,sigmaY=2.5)[8:-8]
empty=mp[ai,ri]
tex=imf[760:900,200:340]-cv2.GaussianBlur(imf,(0,0),3)[760:900,200:340]
T=np.tile(tex,(8,8,1))[:1024,:1024]
empty=np.clip(empty+T,0,255)
# rice mound: visible rice + clone fill
vis_rice=mound&~food&(V>0.45)
vis_rice=cv2.erode(vis_rice.astype(np.uint8),k(3))>0
riceimg=imf.copy(); filled=vis_rice.copy(); target=mound&~filled
rng=np.random.default_rng(3)
for p in range(60):
    if not target.any(): break
    dx,dy=rng.integers(-110,110,2)
    if abs(dx)+abs(dy)<30: continue
    sx=np.clip(xx+dx,0,1023); sy=np.clip(yy+dy,0,1023)
    take=target&filled[sy,sx]
    riceimg[take]=riceimg[sy[take],sx[take]]; filled|=take; target&=~take
print('unfilled',target.sum())
# shading: darken rice towards mound edge, add contact shadow on bowl
md=cv2.distanceTransform(mound.astype(np.uint8),cv2.DIST_L2,5)
rice_sh=riceimg*np.clip(0.78+md/60,0.78,1.0)[...,None]
od=cv2.distanceTransform((~mound).astype(np.uint8),cv2.DIST_L2,5)
empty=empty*(1-0.45*np.exp(-od/14))[...,None]
base=imf.copy()
Rs=cv2.GaussianBlur(R.astype(np.float32),(0,0),3)[...,None]
base=base*(1-Rs)+empty*Rs
ma=cv2.GaussianBlur(mound.astype(np.float32),(0,0),1.5)[...,None]
base=base*(1-ma)+rice_sh*ma
cm=cv2.GaussianBlur(chopm.astype(np.float32),(0,0),0.8)[...,None]
base=base*(1-cm)+imf*cm
base=base.clip(0,255).astype(np.uint8)
cv2.imwrite('base2.png',base)
# residual assignment so final == original
diff=np.abs(base.astype(int)-im.astype(int)).sum(2)>18
res=diff&~food&(A>128)
onR=poly(ON,40); salR=poly(SAL,30)
greens2=greens|(res&onR)|(res&poly(AVO,10)); prot2=prot|(res&salR&~onR)
sauce2=sauce|(res&~onR&~salR&~poly(AVO,10))
def rgba(img,mask,feather=0.9):
    a=(cv2.GaussianBlur(mask.astype(np.float32),(0,0),feather)*255)
    a=np.minimum(a,A.astype(np.float32)).clip(0,255)
    return Image.fromarray(np.dstack([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),a.astype(np.uint8)]))
lay=[Image.fromarray(np.dstack([cv2.cvtColor(base,cv2.COLOR_BGR2RGB),A])),rgba(im,prot2),rgba(im,greens2),rgba(im,sauce2)]
for n,L in zip(['l0','l1','l2','l3'],lay): L.save(n+'.png')
canvas=Image.new('RGBA',(1024,1024),(9,24,47,255)); outs=[]
for L in lay:
    canvas=Image.alpha_composite(canvas,L); outs.append(canvas.resize((512,512)))
s=Image.new('RGB',(1024,1024)); [s.paste(o,((i%2)*512,(i//2)*512)) for i,o in enumerate(outs)]; s.save('stages2.png')
