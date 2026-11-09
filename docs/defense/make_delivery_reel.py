#!/usr/bin/env python3
"""Rebuild the Part 1 / Part 2 reel from bundled clips at original speed."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PARTS = [('lowbar', 'Part 1 · 低姿态通过限高杆'),
         ('plum_handover', 'Part 2 · 六桩与出口交接')]
font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',28,index=2)
with tempfile.TemporaryDirectory(prefix='blackw-defense-') as tmp:
    tmp = Path(tmp)
    for name, label in PARTS:
        background = Image.new('RGB',(1280,720),'#0b1423')
        ImageDraw.Draw(background).text((32,16),label,font=font,fill='#eef5ff')
        background.save(tmp/f'{name}.png')
        subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(tmp/f'{name}.png'),
          '-i',str(HERE/'media'/f'{name}.mp4'),'-filter_complex',
          '[1:v]scale=1200:644:force_original_aspect_ratio=decrease,pad=1200:644:(ow-iw)/2:(oh-ih)/2:black[v];[0:v][v]overlay=40:68:shortest=1,setsar=1',
          '-an','-r','25','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(tmp/f'{name}.mp4')],check=True)
    (tmp/'concat.txt').write_text(''.join(f"file '{tmp/name}.mp4'\n" for name,_ in PARTS))
    out = HERE/'media/delivery_reel.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(tmp/'concat.txt'),
       '-c','copy','-movflags','+faststart',str(out)],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-ss','4','-i',str(out),'-frames:v','1',str(out.with_suffix('.jpg'))],check=True)
manifest = json.loads((HERE/'media_manifest.json').read_text())
manifest = [v for v in manifest if v['id']!='delivery_reel']
manifest.append(dict(id='delivery_reel',file='media/delivery_reel.mp4',
    sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    duration_s=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(out)])),
    composition=[n for n,_ in PARTS],scope='Part1 v104低杆与Part2 v56六桩片段依次展示；原速；只加页外标题与等比例黑边。'))
(HERE/'media_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(manifest[-1])
