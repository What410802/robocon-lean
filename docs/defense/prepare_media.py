#!/usr/bin/env python3
"""Optional import of the locally archived source videos; not required to present."""
from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path(__file__).resolve().parents[2]; d=r/'docs/defense'; old=Path('/home/bis/RoboCon')
h=old/'artifacts/vtaudit-20261031/videos/part2'
clips=[
 ('platform40',h/'vt_mode_v2__adaptive_vx0.5__high_platform_40cm__success.mp4',0,None,'历史独立单关；adaptive vx基准0.5；成功以sidecar/campaign为据'),
 ('wall40',h/'vt_mode_v2__adaptive_vx0.5__high_wall_40cm__success.mp4',0,None,'历史独立单关；adaptive vx基准0.5；视频HUD末帧仍running'),
 ('plum_handover',r/'artifacts/route_fix/part2_full_live_v56_cli.mp4',1054,8.44,'旧25段v56同次连续片段；六桩；深度专家→出口轮式专家；该段严格投影success200'),
 ('lowbar',r/'artifacts/route_fix/part1_lowbar_full_v104_cli.mp4',0,None,'当前源码Part1 v104全程片段；仅路线预检，judge关闭；源仿真123–134s')]
manifest=[]
for name,src,start,duration,scope in clips:
 out=d/'media'/f'{name}.mp4'
 if duration:
  subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-i',str(src),'-t',str(duration),'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
 else: shutil.copy2(src,out)
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height','-of','json',str(out)]))
 poster_time={'platform40':3.5,'wall40':3,'plum_handover':2.5,'lowbar':5.5}[name]
 subprocess.run(['ffmpeg','-v','error','-y','-ss',str(poster_time),'-i',str(out),'-frames:v','1',str(d/'media'/f'{name}.jpg')],check=True)
 manifest.append(dict(id=name,source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),file=f'media/{name}.mp4',sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_video_start_s=start,duration_s=float(info['format']['duration']),scope=scope,probe=info))
for name,src in [
 ('platform40_sidecar.json',h/'vt_mode_v2__adaptive_vx0.5__high_platform_40cm__success.json'),
 ('wall40_sidecar.json',h/'vt_mode_v2__adaptive_vx0.5__high_wall_40cm__success.json'),
 ('historical_adaptive_campaign.json',old/'artifacts/vtaudit-20261031/campaigns/vt_mode_v2__adaptive05/summary.json'),
 ('part1_v104.json',r/'artifacts/route_fix/part1_lowbar_full_v104_evidence.json'),
 ('part2_v56_eval.json',r/'artifacts/route_fix/part2_full_live_v56_eval.json'),
 ('part2_v56_provenance.json',r/'artifacts/route_fix/part2_full_live_v56_provenance.json'),
 ('plum_final_manifest.json',r/'demo/rl_sar/policy/blackW/plum_depth_final_v1/manifest.json'),
 ('bc_depth_r4.json',r/'artifacts/plum/bc/bc_depth_r4.json')]:
 shutil.copy2(src,d/'evidence'/name)
(d/'media_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(id=x['id'],duration_s=x['duration_s']) for x in manifest],ensure_ascii=False))
