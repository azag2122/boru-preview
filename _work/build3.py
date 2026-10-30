import cv2,numpy as np
from PIL import Image
from scipy import ndimage as nd
exec(open('seg3.py').read().split("L={'prot'")[0])   # reuse masks & polygons
im=cv2.imread('/workspace/slides/boru-lulu/img/salmon_bowl_v2.jpg'); imf=im.astype(np.float32)
k=lambda n:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
cx,cy=509,506
food=(prot|greens|sauce)
RICE=[(425,478),(470,458),(560,452),(612,458),(640,478),(652,525),(660,600),(645,660),(605,700),(545,722),(480,726),(442,702),(422,640),(416,560)]
nz=cv2.GaussianBlur(np.random.default_rng(5).normal(0,1,(1024,1024)).astype(np.float32),(0,0),4); nz/=nz.std()
ell=((xx-530)/138.0)**2+((yy-590)/150.0)**2
mound=(ell+0.06*nz)<1.0
mound=cv2.GaussianBlur(mound.astype(np.float32),(0,0),2)>0.5
x1=580+0.563*(yy-100); x2=600+0.587*(yy-100)
chopm=((np.abs(xx-x1)<8.5)|(np.abs(xx-x2)<8.5))&(yy<770)
R=cv2.dilate((food|poly(ON,60)|poly(SAL,24)|poly(AVO,24)|poly(SAU,24)|mound).astype(np.uint8),k(15))>0
R&=(A>128)
# polar model of empty bowl
r=np.hypot(xx-cx,yy-cy); th=(np.arctan2(yy-cy,xx-cx)+np.pi)/(2*np.pi)
NA,NR=96,int(400/3)
ai=(th*NA).astype(int)%NA; ri=np.clip((r/3).astype(int),0,NR-1)
Rbig=cv2.dilate(R.astype(np.uint8),k(41))>0
ok=(A>200)&~Rbig&~chopm&(V<0.36)&(r<372)&~((H>=60)&(H<=130))
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
hp=imf-cv2.GaussianBlur(imf,(0,0),3)
sd=hp[740:820,240:330].std()
T=cv2.GaussianBlur(np.random.default_rng(1).normal(0,1,(1024,1024)).astype(np.float32),(0,0),0.7)
T=(T/T.std()*sd*0.45)[...,None]*np.ones(3)
empty=np.clip(empty+T,0,255)
# rice mound: clean rice texture by inpainting specks, then quilting
clean_src=(S<0.17)&(V>0.60)
specks=cv2.dilate(((~clean_src)&poly(RICE)).astype(np.uint8),k(5))
ricecl=cv2.inpaint(im,specks,4,cv2.INPAINT_TELEA).astype(np.float32)
# candidate source windows fully clean
src=np.zeros((1024,1024),bool); src[452:532,415:528]=True; src[598:662,428:472]=True
bad=(((H<=14)|(H>=330))&(S>0.3)) | ((H>=55)&(H<=150)&(S>0.2)) | (S>0.24) | (V<0.55)
badm=cv2.dilate((bad&src).astype(np.uint8),k(5))
srcimg=cv2.inpaint(im,badm,5,cv2.INPAINT_TELEA).astype(np.float32)
okw=src&~food&(V>0.4)
P=36; cand=[(x,y) for y in range(440,680,3) for x in range(405,540,3) if okw[y:y+P,x:x+P].mean()>0.98]
print('cand',len(cand))
rng=np.random.default_rng(7)
riceimg=np.zeros((1024,1024,3),np.float32); have=np.zeros((1024,1024),np.float32)
f=np.ones((P,P),np.float32); e=10
ramp=np.minimum(np.arange(P)+1,np.arange(P)[::-1]+1)/e; ramp=np.clip(ramp,0,1)
f=np.outer(ramp,ramp).astype(np.float32)
ys,xs=np.where(mound)
pts=[(ty,tx) for ty in range(ys.min()-P//2,ys.max(),P-e*2) for tx in range(xs.min()-P//2,xs.max(),P-e*2)]
for ty,tx in pts:
    sx,sy=cand[rng.integers(len(cand))]
    patch=srcimg[sy:sy+P,sx:sx+P]
    if rng.random()<.5: patch=np.rot90(patch).copy()
    if rng.random()<.5: patch=patch[::-1]
    if rng.random()<.5: patch=patch[:,::-1]
    patch=patch*(np.median(srcimg[okw],axis=0)/np.median(patch.reshape(-1,3),axis=0))
    ty0,tx0=max(ty,0),max(tx,0)
    pa=patch[ty0-ty:,tx0-tx:]; fa=f[ty0-ty:,tx0-tx:]
    hh,ww=pa.shape[:2]
    reg=riceimg[ty0:ty0+hh,tx0:tx0+ww]; hv=have[ty0:ty0+hh,tx0:tx0+ww]
    al=np.where(hv>0,fa,1.0)[...,None]
    reg[:]=reg*(1-al)+pa*al; have[ty0:ty0+hh,tx0:tx0+ww]=1
ricecl=imf
# keep real visible rice where clean, else synthesized
real=np.zeros((1024,1024),bool)
real=cv2.GaussianBlur(cv2.erode(real.astype(np.uint8),k(5)).astype(np.float32),(0,0),2)[...,None]
riceimg=ricecl*real+riceimg*(1-real)
# shading: darken rice towards mound edge, add contact shadow on bowl
md=cv2.distanceTransform(mound.astype(np.uint8),cv2.DIST_L2,5)
lite=np.clip(1.04-0.13*((xx-500)/130.0)-0.11*((yy-555)/140.0),0.80,1.1)
rice_sh=riceimg*(np.clip(0.70+md/40,0.70,1.0)*lite)[...,None]
od=cv2.distanceTransform((~mound).astype(np.uint8),cv2.DIST_L2,5)
empty=empty*(1-0.55*np.exp(-od/16))[...,None]
base=imf.copy()
Rs=cv2.GaussianBlur(R.astype(np.float32),(0,0),3)[...,None]
base=base*(1-Rs)+empty*Rs
ma=cv2.GaussianBlur(mound.astype(np.float32),(0,0),1.2)[...,None]
base=base*(1-ma)+rice_sh*ma
cm=cv2.GaussianBlur(chopm.astype(np.float32),(0,0),0.8)[...,None]
base=base*(1-cm)+imf*cm
base=base.clip(0,255).astype(np.uint8)
cv2.imwrite('base3.png',base)
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
s=Image.new('RGB',(1024,1024)); [s.paste(o,((i%2)*512,(i//2)*512)) for i,o in enumerate(outs)]; s.save('stages3.png')
