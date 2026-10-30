from playwright.sync_api import sync_playwright
import sys
R='/workspace/boru/web_concepts/'
tag=sys.argv[1] if len(sys.argv)>1 else ''
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome')
    for name,w,h,mob in (('desktop',1440,900,False),('mobile',390,844,True)):
        pg=b.new_page(viewport={'width':w,'height':h},device_scale_factor=1,is_mobile=mob,has_touch=mob)
        pg.goto(f'file://{R}V3/index.html'); pg.wait_for_load_state('networkidle')
        pg.evaluate("Promise.all([...document.images].map(i=>{i.loading='eager';return i.decode().catch(()=>{})}))")
        pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(600)
        print(name,'scrollW',pg.evaluate("document.documentElement.scrollWidth"),pg.evaluate("[...document.fonts].filter(f=>f.status=='loaded').map(f=>f.family).join(',')"))
        # frames: center of each step
        tops=pg.evaluate("[...document.querySelectorAll('.step')].map(s=>innerWidth<=820?s.getBoundingClientRect().top+scrollY-innerHeight*.57:s.getBoundingClientRect().top+scrollY+s.offsetHeight/2-innerHeight/2)")
        pos=[0]+tops
        for i,y in enumerate(pos):
            pg.evaluate(f"window.scrollTo(0,{y})"); pg.wait_for_timeout(1300)
            pg.screenshot(path=f'{R}V3/_work/{name}_f{i}{tag}.png')
        pg.evaluate("window.scrollTo(0,0)"); pg.wait_for_timeout(400)
        pg.screenshot(path=f'{R}V3_{name}.png',full_page=True)
        import shutil; shutil.copy(f'{R}V3_{name}.png',f'{R}V3/_work/full_{name}{tag}.png')
    b.close()
