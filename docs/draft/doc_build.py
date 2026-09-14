"""Build the A4 "図面と解析結果" sheet (system python: PyMuPDF + Pillow + playwright)
from the customer drawing PDF (plan view + section details) and the assets of
``doc_render_assets.py``. Prints it through Chromium to prove it is one page
and writes the PDF next to the HTML. Adapted from fangate's docs/draft/doc_build.py.

    python3 docs/draft/doc_build.py <settings_dir> <drawing.pdf> <asset_dir> <out.html>
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import fitz
from PIL import Image

sdir, pdf_path, ad, out_path = (Path(a) for a in sys.argv[1:5])
DPI = 300
page = fitz.open(pdf_path)[0]  # rotation 90: clip is in the rotated (display) frame, pt


def crop(box, rot=0):
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72).prerotate(rot), clip=fitz.Rect(*box))
    return Image.open(io.BytesIO(pix.tobytes("png")))


plan = crop((330, 15, 1640, 1070))
sec_step = crop((95, 195, 235, 425), rot=90)  # rim t1 / body t4 step
sec_runner = crop((95, 815, 235, 1085), rot=90)  # runner: 30 to the gate, t2.5, gate


def b64img(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64file(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


s = json.loads((sdir / "settings.json").read_text(encoding="utf-8"))
m = json.loads((sdir / "metadata.json").read_text(encoding="utf-8"))
m2 = json.loads((sdir / "two_phase_metadata.json").read_text(encoding="utf-8"))
info = json.loads((ad / "info.json").read_text(encoding="utf-8"))
c = s["geometry"]["config"]
T = info["T"]
bo, tp, tw = info["bottom"], info["top"], info["two_phase"]
h_ratio = (c["inner_thk_mm"] / c["frame_thk_mm"]) ** 3
sdir_win = "C:\\Users\\shost\\Downloads\\" + sdir.name
pdf_name = pdf_path.name
inner_w = c["plate_w_mm"] - 2 * c["frame_w_mm"]
inner_h = c["plate_h_mm"] - 2 * c["frame_w_mm"]
bal_txt = (
    f"肉盗み ▽ 幅 {c['balancer_w_mm']:g} × 高さ {c['balancer_h_mm']:g} t{c['balancer_thk_mm']:g}" if c["balancer_on"] else "肉盗み無し"
)
inner_bands = "、".join(f"y={a:g}〜{b:g}" for a, b in tw["adv_inner_y_bands"]) or "無し"
frame_bands = "、".join(f"y={a:g}〜{b:g}" for a, b in tw["adv_frame_y_bands"]) or "無し"

reading_fill = (
    f"樹脂はゲート φ{c['gate_d_mm']:g} からランナ（t{c['runner_edge_thk_mm']:g}→{c['runner_thk_mm']:g}）を扇状に広がり、製品エッジの中央 (x=0) に {bo['min_t']:.3f} s、"
    f"エッジの両端 (x=±{abs(bo['max_x']):.0f}) に {bo['max_t']:.3f} s で届く。ランナ側の額縁（下辺 20 幅）は平均 {info['rim_bottom_mean_t']:.3f} s で最初に埋まり、"
    f"内側 t{c['inner_thk_mm']:g} へは段差の中央 (x={info['inner_first_xy'][0]:g}, y={info['inner_first_xy'][1]:g}) から {info['inner_first_t']:.3f} s で入って上へ扇状に進む。"
    f"額縁 t{c['frame_thk_mm']:g} は内側の 1/{h_ratio:.0f}（h³）しか流れないので、左右の額縁は内側から横に押し出される形で遅れて埋まり（側辺の平均 {info['rim_sides_mean_t']:.2f} s）、"
    f"内側の最後は上の両隅 (±{abs(info['inner_last_xy'][0]):.0f}, {info['inner_last_xy'][1]:g}) で {info['inner_last_t']:.3f} s、上辺の額縁が最後（平均 {info['rim_top_mean_t']:.2f} s、"
    f"上辺中央 {tp['center']:.3f} s、四隅 {T:.3f} s、赤）。到着時刻の平均は内側 {info['inner_mean_t']:.2f} s に対し額縁 {info['frame_mean_t']:.2f} s。"
)
short_cm3 = m2["cavity_volume_final_cm3"] - tw["shot_cm3"]
shot_vs_cav = "＝キャビティ体積" if abs(short_cm3) < 0.005 else f"キャビティ体積 {m2['cavity_volume_final_cm3']:.2f} cm³ より {short_cm3:.2f} cm³ 少ない"
final_pct = f"{tw['final_fraction']:.0%}" if tw["final_fraction"] >= 1.0 else f"{tw['final_fraction']:.1%}"
reading_tp = (
    f"計量 {tw['shot_cm3']:.1f} cm³（{shot_vs_cav}）を {s['injection']['injection_volume_flow_cm3s']:g} cm³/s で射出（{tw['injection_time_s']:.3f} s）。"
    f"内側だけ型を {s['compression_molding']['stroke_mm']:g} mm 開いた状態で射出中に埋まるのは {tw['injection_fraction']:.0%}（{m2['injection_cells']:,} セル）で、"
    f"残り {1 - tw['injection_fraction']:.0%}（{tw['adv_cells']:,} セル: 額縁 {tw['adv_frame']:,}［側辺 {tw['adv_rim_sides']:,}・上辺 {tw['adv_rim_top']:,}］・内側 {tw['adv_inner']:,}・ランナ {tw['adv_runner']}）は"
    f"型閉じの圧縮で埋まる。圧縮されるのは内側 t{c['inner_thk_mm']:g} だけだが、押し潰された内側から出た樹脂が額縁へ回る: 圧縮で埋まる領域は左右の額縁（{frame_bands}）と上辺の額縁全部、"
    f"内側では上の両隅（|x| ≥ {tw['adv_inner_absx_min']:g}、{inner_bands}）。ランナ側の額縁とランナは射出中に埋まっている。"
    + (
        "ショートショット・未充填は無い（圧縮後 100%）。"
        if tw["final_fraction"] >= 1.0 and m2["compression_unreachable_cells"] == 0
        else f"計量がキャビティ体積を {short_cm3:.2f} cm³ 下回るので圧縮後も {tw['final_fraction']:.1%}、"
        f"未充填 {m2['cavity_cells'] - m2['final_cells']:,} セル（≈ {short_cm3:.2f} cm³）のショートショット"
        f"（圧縮で届かないセルは {m2['compression_unreachable_cells']}）。"
    )
)

wc = s["wall_cooling"]
if wc["model"] == "skin":
    wall_txt = f"スキン層 c_skin = {wc['skin_growth_constant']:g}、{'定圧時計' if wc['skin_clock_mode'] == 'constant_pressure' else wc['skin_clock_mode']}、反復 {m.get('skin_iterations', '?')}（{'収束' if m.get('skin_converged') else '未収束'}）"
    fill_txt = f"{T:.4f} s（等温 {m['T_fill_baseline_s']:.4f} s × スキン抵抗 {m['T_fill_inflation']:.2f}）、封止 {m['sealed_off_cells']} / 未充填 {m['unfillable_cells']}"
    tp_skin = f"、射出相スキン最大 {m2['injection_skin_max_mm']:.2f} mm"
else:
    wall_txt = "なし（等温 Hele-Shaw）"
    fill_txt = f"{T:.4f} s（100% 充填）"
    tp_skin = ""
html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>図面と流動解析結果の対応 — 12.3 インチ 額縁プレート ファン状ランナ（{sdir.name}）</title>
<style>
@page {{ size: A4 portrait; margin: 10mm 12mm 8mm 12mm; }}
html, body {{ margin: 0; padding: 0; background: #fff; color: #111; font-family: "Meiryo", "Yu Gothic", "Hiragino Sans", "Noto Sans JP", sans-serif; }}
.page {{ width: 186mm; margin: 0 auto; }}
@media screen {{ body {{ background: #e5e5e5; }} .page {{ background: #fff; padding: 12mm; margin: 8mm auto; box-shadow: 0 0 4mm rgba(0,0,0,.25); }} }}
h1 {{ font-size: 11.5pt; margin: 0 0 1mm; letter-spacing: .02em; }}
.sub {{ font-size: 8.2pt; color: #444; margin: 0 0 1.5mm; }}
h2 {{ font-size: 9.5pt; margin: 2.2mm 0 1mm; padding: 0 0 .8mm 2mm; border-left: 1mm solid #333; border-bottom: .2mm solid #bbb; }}
.fig {{ width: 100%; display: block; }}
.cap {{ font-size: 7.2pt; color: #333; margin: .8mm 0 0; line-height: 1.4; }}
table {{ border-collapse: collapse; font-size: 7.2pt; width: 100%; }}
th, td {{ border: .2mm solid #999; padding: .4mm 1.2mm; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; font-weight: 600; white-space: nowrap; width: 22mm; }}
.three {{ display: flex; gap: 3mm; align-items: flex-start; }}
.three > div {{ flex: 1 1 0; min-width: 0; }}
.two {{ display: flex; gap: 4mm; align-items: flex-start; margin-top: 2mm; }}
.two .l {{ flex: 0 0 98mm; }} .two .r {{ flex: 1 1 auto; }}
.draw {{ display: flex; gap: 3mm; align-items: flex-start; }}
.draw .plan {{ flex: 0 0 110mm; }} .draw .secs {{ flex: 0 0 60mm; min-width: 0; }}
.secs img {{ border: .2mm solid #ccc; width: 58mm; }}
.three img {{ width: 94%; margin: 0 auto; }}
.foot {{ font-size: 6.6pt; color: #666; margin-top: 1mm; border-top: .2mm solid #bbb; padding-top: 1mm; }}
.arrow {{ text-align: center; font-size: 9pt; color: #555; margin: 1mm 0; }}
</style></head>
<body><div class="page">
<h1>図面と流動解析結果の対応　— 12.3 インチ 額縁プレート（ファン状ランナ、圧縮は内側のみ）</h1>
<p class="sub">検討用資料　2026/9/14　ベース: mold-flow-fangate2 {s['app_version']} の実行設定（{sdir.name}）＝ 図面 {pdf_name}（2026/9/14）／ 製品 {c['plate_w_mm']:g}×{c['plate_h_mm']:g}、額縁 {c['frame_w_mm']:g} 幅 t{c['frame_thk_mm']:g}、内側 {inner_w:g}×{inner_h:g} t{c['inner_thk_mm']:g}（圧縮部）、ランナ 幅 {c['runner_w_mm']:g}・フランク {c['fan_flank_deg']:g}°・軸まで {c['runner_len_mm']:g}・R{c['runner_end_d_mm'] / 2:g} 丸端、肉厚 {c['runner_edge_thk_mm']:g}→{c['runner_thk_mm']:g}、ゲート φ{c['gate_d_mm']:g}、{bal_txt}。</p>

<h2>1. 図面（平面図 ／ 製品断面の抜粋）　{pdf_name} より</h2>
<div class="draw">
<div class="plan"><img class="fig" src="{b64img(plan)}" alt="平面図"></div>
<div class="secs">
<img class="fig" src="{b64img(sec_step)}" alt="製品断面（額縁 t1 → 内側 t4 の段差）">
<p class="cap" style="margin:.5mm 0 2mm">製品断面（額縁側）: 額縁 t{c['frame_thk_mm']:g} と内側 t{c['inner_thk_mm']:g} の段差。図面の縦向きの断面を 90° 回して掲載。</p>
<img class="fig" src="{b64img(sec_runner)}" alt="製品断面（ランナ側）">
<p class="cap" style="margin:.5mm 0 0">製品断面（ランナ側）: 額縁 t{c['frame_thk_mm']:g} に続くランナは製品エッジから {c['runner_edge_flat_mm']:g} mm は t{c['runner_edge_thk_mm']:g} のまま、{c['runner_ramp_end_mm']:g} mm で t{c['runner_thk_mm']:g} へ傾斜、以降 t{c['runner_thk_mm']:g} で R{c['runner_end_d_mm'] / 2:g} の丸端まで。ゲート φ{c['gate_d_mm']:g} はエッジから {c['runner_len_mm']:g} の軸上（破線、固定側から）。</p>
</div>
</div>
<p class="cap">平面図: 内側 {inner_w:g}×{inner_h:g}（肉厚 {c['inner_thk_mm']:g} mm 部）の周りに {c['frame_w_mm']:g} 幅の額縁（肉厚 {c['frame_thk_mm']:g} mm 部、外形 {c['plate_w_mm']:g}×{c['plate_h_mm']:g}）。ランナは製品エッジで幅 {c['runner_w_mm']:g}、フランクは {c['fan_flank_deg']:g}° の直線で R{c['runner_end_d_mm'] / 2:g} の円に交差（接線ではない。延長の頂点は深さ {info['apex_depth']:.1f}）、円の中心がエッジから {c['runner_len_mm']:g} のゲート軸。図面の実線（深さ 20）が肉厚傾斜の終端。キャビティ体積 {info['volume_cm3']:.1f} cm³（製品 {info['product_cm3']:.1f} ＋ ランナ {info['volume_cm3'] - info['product_cm3']:.1f}）。</p>

<div class="arrow" style="margin:.5mm 0">▼ 上の図面を {info['cell']:g} mm 格子に落とした解析モデルと、それを mold-flow-fangate2 で解いた結果 ▼</div>

<h2>2. 解析モデル（肉厚）／ 充填時間分布（充填完了 t = {T:.4f} s、player.html 最終フレーム）／ 二相</h2>
<div class="three">
<div><img class="fig" src="{b64file(ad / 'thickness.png')}" alt="肉厚マップ（解析モデル）">
<p class="cap">解析モデルの肉厚（型閉じ後）。格子 {info['nx']} × {info['ny']}、セル {info['cell']:g} mm。明＝薄肉／暗＝厚肉。額縁 t{c['frame_thk_mm']:g}、内側 t{c['inner_thk_mm']:g}、ランナ t{c['runner_edge_thk_mm']:g}→{c['runner_thk_mm']:g}。赤枠＝圧縮部（内側だけ）、赤丸＝ゲート φ{c['gate_d_mm']:g}。</p></div>
<div><img class="fig" src="{b64file(ad / 'final_frame.png')}" alt="充填時間分布（最終フレーム）">
<p class="cap">色は各点の充填時刻（{s['output']['fill_cmap']}、赤が最後）、線は等時線 {s['output']['isochrone_levels']} 本。y=0 が製品エッジ（ランナ側）、y&lt;0 がランナ、赤丸がゲート。</p></div>
<div><img class="fig" src="{b64file(ad / 'two_phase.png')}" alt="二相ショートショット">
<p class="cap">計量 {tw['shot_cm3']:.1f} cm³ を内側だけ型開き {s['compression_molding']['stroke_mm']:g} mm で射出した時点の充填域（青、白線は等時線）と、型閉じの圧縮で埋まる域（橙）。</p></div>
</div>

<div class="two">
<div class="l"><table>
<tr><th>樹脂</th><td>{s['material']}（{m['material']}）</td></tr>
<tr><th>樹脂温 / 金型温</th><td>{s['injection']['melt_temperature_C']} ℃ / {s['injection']['mold_temperature_C']} ℃</td></tr>
<tr><th>射出</th><td>速度 {s['injection']['injection_velocity_mms']:g} mm/s、流量 {s['injection']['injection_volume_flow_cm3s']:g} cm³/s</td></tr>
<tr><th>射出圧縮</th><td>ON、ストローク {s['compression_molding']['stroke_mm']:g} mm（圧縮部＝内側 t{c['inner_thk_mm']:g} のみ、額縁とランナは固定）、圧縮比率 {s['compression_molding']['fraction']:g}</td></tr>
<tr><th>壁面冷却</th><td>{wall_txt}</td></tr>
<tr><th>メッシュ</th><td>{info['cell']:g} mm、{info['nx']} × {info['ny']}、キャビティ体積 {m['volume_cm3']:.2f} cm³</td></tr>
<tr><th>充填時間</th><td>{fill_txt}</td></tr>
<tr><th>二相ショートショット</th><td>計量 {m2['shot_volume_cm3']:.2f} cm³、射出 {m2['injection_time_s']:.3f} s で {m2['injection_fill_fraction']:.1%} → 圧縮後 {final_pct}、圧縮で届かないセル {m2['compression_unreachable_cells']}{tp_skin}</td></tr>
<tr><th>ソフト</th><td>mold-flow-fangate2 {s['app_version']}</td></tr>
</table></div>
<div class="r"><p class="cap" style="margin-top:0">読み方（充填）: {reading_fill}</p>
<p class="cap" style="margin-top:1.5mm">読み方（二相）: {reading_tp}</p></div>
</div>

<p class="foot">出所: {sdir_win}\\（settings.json / metadata.json / two_phase_metadata.json / player.html）。図面は {pdf_name}（1:1 スケール、寸法はベクタから実寸）。最終フレーム・肉厚マップ・二相マップは同じ設定を解き直して高解像度化したもの（τ_max・射出充填率は metadata と一致）。</p>
</div></body></html>"""
out_path.write_text(html, encoding="utf-8")
from playwright.sync_api import sync_playwright  # noqa: E402

pdf_out = out_path.with_suffix(".pdf")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto("file://" + str(out_path))
    pg.pdf(path=str(pdf_out), format="A4", prefer_css_page_size=True, print_background=True)
    b.close()
d = fitz.open(str(pdf_out))
d[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(str(ad / "check.png"))
print(f"{len(html) // 1024} KB, pages={len(d)}", out_path, pdf_out)
assert len(d) == 1, "A4 1 ページに収まっていない"
