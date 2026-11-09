#!/usr/bin/env python3
"""Compose three independent demonstration clips; never a continuous-run claim."""
from pathlib import Path
import subprocess, json, hashlib
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent/'media'
f=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',26,index=2)
f2=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19,index=2)
parts=[('platform40','历史独立单关 · 40 cm 高台','adaptive vx基准0.5 · sidecar success / 200'),('wall40','历史独立单关 · 40 cm 高墙','adaptive vx基准0.5 · sidecar success / 200'),('plum_handover','六桩场地 · 过桩与出口交接','旧25段v56的同次连续片段 · 本段支撑点回放 success')]
for id,title,caption in parts:
 im=Image.new('RGB',(1280,720),'#0b1220'); draw=ImageDraw.Draw(im);draw.text((36,12),title,font=f,fill='#f1f5f9');draw.text((36,54),caption,font=f2,fill='#9aadc4');im.save(p/f'{id}_background.png')
 subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(p/f'{id}_background.png'),'-i',str(p/f'{id}.mp4'),'-filter_complex','[1:v]scale=1200:616:force_original_aspect_ratio=decrease,pad=1200:616:(ow-iw)/2:(oh-ih)/2:black[v];[0:v][v]overlay=40:94:shortest=1,setsar=1','-an','-r','25','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',f'/tmp/defense_{id}.mp4'],check=True)
(p/'concat.txt').write_text(''.join("file '/tmp/defense_%s.mp4'\n"%id for id,*_ in parts))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(p/'concat.txt'),'-c','copy','-movflags','+faststart',str(p/'capability_reel.mp4')],check=True)
subprocess.run(['ffmpeg','-v','error','-y','-ss','3','-i',str(p/'capability_reel.mp4'),'-frames:v','1',str(p/'capability_reel.jpg')],check=True)
(p/'concat.txt').unlink()
for id,*_ in parts: (p/f'{id}_background.png').unlink()
m=json.loads((p.parent/'media_manifest.json').read_text()); file=p/'capability_reel.mp4'
m=[x for x in m if x['id']!='capability_reel']
m.append(dict(id='capability_reel',file='media/capability_reel.mp4',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),duration_s=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(file)])),composition=[x[0] for x in parts],scope='三段独立证据的顺序展示；不是一次连续通过三个障碍；只加页外标题和等比例黑边，无加速'))
(p.parent/'media_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print(m[-1])
