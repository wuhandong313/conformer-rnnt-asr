#!/usr/bin/env python3
"""Render the paper comparison rows as a high-resolution results table."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/research/phone69_all_results_20261117/results.json'
OUT = ROOT / 'artifacts/research/phone69_paper_results_table_20261117'
GROUPS = [
    ('固定步长基线', ['S1', 'S2', 'E09']),
    ('单阶段压缩', ['E05', 'P_L1', 'E02', 'P_L3', 'E06', 'E07', 'P_L9']),
    ('两阶段压缩', ['P_MEL_L3', 'E10', 'E08']),
]
SCALE = 2
WIDTH, HEIGHT = 1280, 1510
INK, MUTED = '#172B43', '#5C6F83'
BLUE, GREEN = '#215C9D', '#147D66'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    rows = {row['id']: row for row in source['models']}
    selected = [identity for _, group in GROUPS for identity in group]
    expected = {row['id'] for row in rows.values()
                if row['inclusion'] in ('existing_paper_selection', 'new_placement_comparison')}
    assert len(selected) == len(set(selected)) == 13 and set(selected) == expected
    assert source['complete'] and source['duration_seconds'] == 30
    for identity in selected:
        assert rows[identity]['utterances'] == 2620
        assert rows[identity]['reference_phones'] == 189023

    cjk = '/System/Library/Fonts/Hiragino Sans GB.ttc'
    latin = '/System/Library/Fonts/Supplemental/Arial.ttf'
    latin_bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    fonts = {}

    def font(size, numeric=False, bold=False):
        path = (latin_bold if bold else latin) if numeric else cjk
        key = path, size
        if key not in fonts:
            fonts[key] = ImageFont.truetype(path, round(size * SCALE))
        return fonts[key]

    image = Image.new('RGB', (WIDTH * SCALE, HEIGHT * SCALE), '#FFFFFF')
    draw = ImageDraw.Draw(image)

    def rectangle(box, fill, radius=0):
        box = tuple(round(v * SCALE) for v in box)
        if radius:
            draw.rounded_rectangle(box, radius=round(radius * SCALE), fill=fill)
        else:
            draw.rectangle(box, fill=fill)

    def line(y, color='#E1E8EF', width=1):
        draw.line((80 * SCALE, y * SCALE, 1200 * SCALE, y * SCALE),
                  fill=color, width=round(width * SCALE))

    def text(value, x, y, size=24, color=INK, align='left', numeric=False, bold=False):
        f = font(size, numeric, bold)
        box = draw.textbbox((0, 0), value, font=f)
        tw, th = box[2] - box[0], box[3] - box[1]
        left = x * SCALE - (tw / 2 if align == 'center' else tw if align == 'right' else 0)
        top = y * SCALE - th / 2 - box[1]
        assert left >= 0 and left + tw < image.width, 'Text exceeds canvas: ' + value
        assert top + box[1] >= 0 and top + box[3] < image.height
        draw.text((round(left), round(top)), value, fill=color, font=f)

    rectangle((80, 51, 88, 76), BLUE, 3)
    text('PHONE69  /  LIBRISPEECH', 104, 64, 20, MUTED, numeric=True, bold=True)
    text('性能与计算量对比', 80, 124, 46)
    text('test-clean · 2,620句 · FP32 · batch 1 · beam 20', 82, 180, 21, MUTED)
    text('固定30秒输入  ·  编码器与压缩模块', 82, 220, 20, BLUE)

    top, row_height, group_height = 260, 65, 42
    rectangle((80, top, 1200, top + 62), '#193A59', 9)
    text('实验配置', 104, top + 31, 23, '#FFFFFF')
    text('主干维度', 682, top + 31, 22, '#FFFFFF', 'center')
    text('PER (%) ↓', 876, top + 31, 22, '#FFFFFF', 'right')
    text('GMAC / 30 s ↓', 1176, top + 31, 22, '#FFFFFF', 'right')
    y = top + 62
    best_per = min(rows[i]['per_percent'] for i in selected)
    best_mac = min(rows[i]['encoder_compression_gmac_30s'] for i in selected)
    for group_name, identities in GROUPS:
        rectangle((80, y, 1200, y + group_height), '#ECF1F6')
        text(group_name, 104, y + group_height / 2, 20, MUTED)
        y += group_height
        for index, identity in enumerate(identities):
            row = rows[identity]
            new = identity == 'P_MEL_L3'
            if new:
                rectangle((80, y, 1200, y + row_height), '#EAF4FF')
                rectangle((80, y, 85, y + row_height), BLUE)
            elif index % 2 == 1:
                rectangle((80, y, 1200, y + row_height), '#F8FAFC')
            center = y + row_height / 2
            label = {'E09': 'Stride 4', 'E08': 'L2 + L5*',
                     'P_MEL_L3': 'Mel + L3', 'E10': 'Mel + L5'}.get(identity, row['system'])
            text(label, 104, center, 26, BLUE if new else INK,
                 numeric=label.isascii(), bold=new)
            if new:
                rectangle((298, y + 19, 358, y + 45), '#D4E6FC', 5)
                text('新增', 328, center, 16, BLUE, 'center')
            text(str(row['backbone']), 682, center, 25, MUTED, 'center', numeric=True)
            per_best = row['per_percent'] == best_per
            mac_best = row['encoder_compression_gmac_30s'] == best_mac
            text(f"{row['per_percent']:.4f}", 876, center, 27,
                 GREEN if per_best else INK, 'right', numeric=True, bold=per_best)
            text(f"{row['encoder_compression_gmac_30s']:.4f}", 1176, center, 27,
                 GREEN if mac_best else INK, 'right', numeric=True, bold=mac_best)
            y += row_height
            line(y)

    text('绿色加粗：本表最优值；PER与MAC均越低越好。', 82, y + 33, 19, GREEN)
    text('* L2 + L5：L2为hard三帧邻域，L5为learned五帧融合。', 82, y + 69, 18, MUTED)
    text('PER来自完整test-clean；其余动态压缩采用learned五帧融合。', 82, y + 100, 18, MUTED)
    text('30秒对应3,001个Mel帧，MAC由各checkpoint实测逐层条件保留率校准。',
         82, y + 131, 18, MUTED)
    text('主干维度与各模型实际配置一致；所有checkpoint均按dev-clean验证损失选择。',
         82, y + 162, 18, MUTED)
    text('2026-11-17', 1200, HEIGHT - 34, 16, MUTED, 'right', numeric=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'paper_results_30s.png'
    image.save(path, optimize=True, dpi=(240, 240))
    snapshot = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
                    script_sha256=sha(__file__), image=path.name, image_sha256=sha(path),
                    duration_seconds=30, scope='encoder and compression major MACs',
                    selection='Original paper comparison and newly evaluated placement comparisons; '
                              'historical exclusions and training-loss checkpoint diagnosis omitted.',
                    selected_ids=selected, models=[rows[i] for i in selected],
                    excluded_ids=[i for i in rows if i not in selected])
    (OUT / 'data.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(image=str(path), rows=len(selected), size=image.size,
                          bytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
