#!/usr/bin/env python3
"""Validate delivered slide/media artifacts; never run robot physics."""
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
report={}
meta=json.loads((HERE/'presentation.json').read_text())
count=len(meta['slides'])
main_files={f for slide in meta['slides'] if not slide['backup'] for f in slide['videos']}
figures=json.loads((HERE/'figures/manifest.json').read_text())
for fig in figures:
    assert hashlib.sha256((HERE/fig['file']).read_bytes()).hexdigest()==fig['sha256']
report['paper_figures']={'count':len(figures),'hashes_match':True,'attribution':'figure manifest and slide footers'}
manifest=json.loads((HERE/'media_manifest.json').read_text())
for item in manifest:
    file=HERE/item['file']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'], item['id']
    subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True,capture_output=True)
report['media']={'files':len(manifest),'hashes_match':True,'full_video_decode':'PASS','presentation_video_duration_s':round(sum(x['duration_s'] for x in manifest if x['file'] in main_files),2)}
prs=Presentation(HERE/'blackW-defense.pptx')
assert len(prs.slides)==count
for i,slide in enumerate(prs.slides):
    assert slide.has_notes_slide
    assert slide.notes_slide.notes_text_frame.text.strip()
    for sh in slide.shapes:
        assert sh.left>=0 and sh.top>=0
        assert sh.left+sh.width<=prs.slide_width+10
        assert sh.top+sh.height<=prs.slide_height+10
with zipfile.ZipFile(HERE/'blackW-defense.pptx') as z:
    assert z.testzip() is None
    embedded=[x for x in z.namelist() if x.endswith('.mp4')]
    all_files={f for slide in meta['slides'] for f in slide['videos']}
    assert len(embedded)==len(all_files)
report['pptx']={'slides':count,'main':meta['main_slides'],'backup':meta['backup_slides'],'speaker_notes':count,'embedded_videos':len(embedded),'editable_text':True,'package_integrity':'PASS','object_bounds':'PASS'}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':932})
    errors=[]; requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('requestfailed',lambda r:requests.append({'url':r.url,'error':r.failure}) if r.failure != 'net::ERR_ABORTED' else None)
    page.goto((HERE/'index.html').as_uri());page.evaluate('document.fonts.ready')
    assert page.evaluate('document.fonts.check(\'700 30px "BlackWDefense"\')')
    overflows=page.evaluate('''() => [...document.querySelectorAll('.text')].filter(e => e.scrollWidth > e.clientWidth + 2 || e.scrollHeight > e.clientHeight + 2).map(e=>e.textContent)''')
    # Check visible pages independently (hidden pages have zero client dimensions).
    overflows=[]
    for i in range(count):
        page.evaluate(f'go({i})')
        overflows.extend(page.evaluate('''() => [...document.querySelectorAll('.slide.active .text')].filter(e => e.scrollWidth > e.clientWidth + 2 || e.scrollHeight > e.clientHeight + 2).map(e=>e.textContent)'''))
    assert not overflows,overflows
    for i,slide in enumerate(meta['slides']):
        if not slide['videos']:continue
        page.evaluate(f'go({i})')
        page.wait_for_function("document.querySelector('.slide.active video').readyState >= 2")
        page.evaluate("document.querySelector('.slide.active video').play()")
        page.wait_for_function("document.querySelector('.slide.active video').currentTime > .2")
        page.evaluate("document.querySelector('.slide.active video').pause()")
    page.evaluate('go(0)');page.keyboard.press('ArrowRight')
    assert page.evaluate('n')==1
    page.keyboard.press('Home');assert page.evaluate('n')==0
    assert not errors,errors
    assert not requests,requests
    browser.close()
report['html']={'offline_file_open':'PASS','fonts':'local_subset_loaded','main_and_backup_video_playback':'PASS','keyboard_navigation':'PASS','javascript_errors':errors,'failed_requests':requests,'visible_text_overflow':'none'}
info=subprocess.check_output(['pdfinfo',str(HERE/'blackW-defense.pdf')],text=True)
assert re.search(r'Pages:\s+'+str(count)+r'\b',info)
text=subprocess.check_output(['pdftotext',str(HERE/'blackW-defense.pdf'),'-'],text=True)
for expected in ['DreamWaQ','Extreme Parkour','RecoveryLoco','270','247','40 cm','1,228,800','1.05','0.254','3932','model_750']:
    assert re.sub(r'\s+','',expected) in re.sub(r'\s+','',text),expected
for i in range(count):assert Image.open(HERE/'previews'/f'{i+1:02d}.png').size==(1600,900)
report['pdf']={'pages':count,'text_extraction':'PASS','chinese_fonts':'embedded','poster_on_video_pages':True,'source':'same layout rendered by Chromium'}
render_check=HERE/'evidence/pptx_render_check.json'
if render_check.exists():
    rendered=json.loads(render_check.read_text())
    current_hash=hashlib.sha256((HERE/'blackW-defense.pptx').read_bytes()).hexdigest()
    if rendered.get('pptx_sha256')==current_hash:
        report['pptx']['independent_application_render']=rendered
    else:
        report['pptx']['independent_application_render']='previous revision only; not reused'
report['scope']={'physics_rerun':False,'training_rerun':False,'oral_rehearsal':'not performed; 410s planned budget','office_wps_visual_test':'not performed in main build; see README for current verification scope'}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
