#!/usr/bin/env python3
"""Rebuild offline font subsets after editing slide text. Uses installed Noto CJK."""
from pathlib import Path
from fontTools import subset
from fontTools.ttLib import TTFont
p=Path(__file__).resolve().parent;(p/'fonts').mkdir(exist_ok=True)
text=''.join((p/x).read_text() for x in ['script.md','build.py'])+'0123456789 ←→上一页下一页播放暂停全屏方向键翻页主讲备份问答分秒'
for weight in ['Regular','Bold']:
 font=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc',fontNumber=2)
 opts=subset.Options();opts.flavor='woff2';opts.recalc_timestamp=False
 sub=subset.Subsetter(options=opts);sub.populate(text=text);sub.subset(font)
 # A distinct family name identifies this subset.
 for rec in font['name'].names:
  if rec.nameID in [1,4,6,16]:
   name='BlackWDefenseSubset'+('-Bold' if weight=='Bold' else '')
   rec.string=name.encode(rec.getEncoding())
 font.flavor='woff2';font.save(p/'fonts'/f'blackw-{weight.lower()}.woff2')
(p/'fonts'/'LICENSE.txt').write_text(Path('/usr/share/doc/fonts-noto-cjk/copyright').read_text())
print('Subset fonts saved')
