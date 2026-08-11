#!/usr/bin/env python3
"""紙芝居スライド生成スクリプト

Results構成に対応したスライドを1枚1図で構成するHTML生成スクリプト。
Methods 2枚 + Results 6枚 = 計8枚。
各スライドに (a) タイトル、(b) 図表またはテキスト/HTMLコンテンツ、(c) 結論テキスト（1〜2文）を含む。
画像不在時はプレースホルダーテキスト「[図表未生成: {filename}]」を表示する。

出力: reports/paper_figs_v2/kamishibai_slides.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.paper_figs.feature_definitions import (
        get_classical_features,
        get_novel_features,
    )
except ModuleNotFoundError:
    from feature_definitions import get_classical_features, get_novel_features  # type: ignore[import-untyped]


@dataclass
class Slide:
    """1枚のスライドを表すデータ構造"""

    number: int
    title: str
    images: list[str]  # 画像ファイル名のリスト（1枚 or 複数枚）
    conclusion: str
    html_content: str = ""  # テキスト/HTMLコンテンツ（Methodsスライド等）
    methods_note: str = ""  # 各スライドの分析手法注釈（1〜2行）


# --- 特徴量分類テーブル生成 ---------------------------------------------------


def _build_feature_classification_table() -> str:
    """Classical vs Novel 特徴量のHTML比較テーブルを生成する"""
    classical = get_classical_features()
    novel = get_novel_features()

    rows: list[str] = []
    max_len = max(len(classical), len(novel))
    for i in range(max_len):
        c_name = classical[i].name if i < len(classical) else ""
        c_summary = classical[i].summary if i < len(classical) else ""
        n_name = novel[i].name if i < len(novel) else ""
        n_summary = novel[i].summary if i < len(novel) else ""
        rows.append(
            f"<tr><td>{i + 1}</td>"
            f"<td>{c_name}</td><td>{c_summary}</td>"
            f"<td>{n_name}</td><td>{n_summary}</td></tr>"
        )

    return (
        '<table class="feature-table">'
        "<thead><tr>"
        "<th>#</th>"
        "<th colspan=\"2\">Classical（既存研究ベース: PG + FILL = 9個）</th>"
        "<th colspan=\"2\">Novel（新規提案: IX + RESP = 9個）</th>"
        "</tr><tr>"
        "<th></th><th>特徴量名</th><th>概要</th><th>特徴量名</th><th>概要</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# --- スライド定義 -----------------------------------------------------------

SLIDES: list[Slide] = [
    # --- Methods スライド (1-2) ---
    Slide(
        number=1,
        title="データと手法",
        images=[],
        conclusion=(
            "N=120の自然会話データに対し、Ridge回帰＋置換検定で"
            "特徴量とBig5の関連を頑健に評価する設計である。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>データ</h3>"
            "<ul>"
            "<li><b>コーパス:</b> CEJC（日本語日常会話コーパス）home2サブセット</li>"
            "<li><b>品質フィルタ:</b> HQ1（高品質フィルタ適用済み）</li>"
            "<li><b>サンプルサイズ:</b> N = 120（conversation × speaker）</li>"
            "<li><b>特徴量:</b> 19説明変数（Classical 10 + Novel 9）</li>"
            "</ul>"
            "<h3>手法</h3>"
            "<ul>"
            "<li><b>回帰モデル:</b> Ridge回帰（α = 100）</li>"
            "<li><b>交差検証:</b> 5-fold subject-wise CV</li>"
            "<li><b>統計検定:</b> Permutation test（5,000回、seed = 42）</li>"
            "<li><b>安定性評価:</b> Bootstrap係数安定性分析</li>"
            "<li><b>教師ラベル:</b> 4 LLM教師のitem-level平均（アンサンブル）</li>"
            "</ul>"
            "</div>"
        ),
    ),
    Slide(
        number=2,
        title="特徴量の分類",
        images=[],
        conclusion=(
            "既存研究ベースのClassical 10特徴量と、"
            "会話分析・相互行為論に基づくNovel 9特徴量の2群で構成される。"
        ),
        html_content=_build_feature_classification_table(),
    ),
    # --- Results スライド (3-8) ---
    Slide(
        number=3,
        title="提案特徴量の分布",
        images=["fig_feature_distribution.png"],
        conclusion=(
            "19特徴量は適度なばらつきを持ち、"
            "個人差を捉える指標として有用である。"
        ),
        methods_note="CEJC home2 HQ1（N=120）から抽出した19特徴量（Classical 10 + Novel 9）のバイオリンプロット。",
    ),
    Slide(
        number=4,
        title="カテゴリ内/間相関",
        images=["fig_corr_heatmap_block.png"],
        conclusion=(
            "同一カテゴリ内で高相関を示す一方、"
            "カテゴリ間は独立性が高い。"
        ),
        methods_note="18特徴量間のPearson相関行列。カテゴリ順: PG→FILL→IX→RESP。",
    ),
    Slide(
        number=5,
        title="コーパス基本情報との関連",
        images=["fig_metadata_gender.png", "fig_metadata_age.png"],
        conclusion=(
            "性別・年齢と一部特徴量に有意な関連が認められた。"
        ),
        methods_note="性別: Mann-Whitney U検定（M=54, F=66）。年齢: Pearson r / Spearman ρ。",
    ),
    Slide(
        number=6,
        title="アンサンブルBig5 Permutation",
        images=["fig_ensemble_permutation.png"],
        conclusion=(
            "4教師item-level平均によるアンサンブルBig5で、"
            "O, C, A, Nの4次元が有意（Eのみ非有意）。"
        ),
        methods_note="4教師のIPIP-NEO-120 item-level平均 → Ridge（α=100）+ 5-fold CV + Permutation test（5,000回）。",
    ),
    Slide(
        number=7,
        title="3段階Ridge回帰比較",
        images=["fig_three_stage_comparison.png"],
        conclusion=(
            "人口統計→Classical→Novelの段階的追加により予測精度が向上し、"
            "各特徴量群の追加効果（Δr）を明示する。"
        ),
        methods_note="Stage 1: 人口統計のみ（2変数）→ Stage 2: +Classical（11変数）→ Stage 3: +Novel（20変数）。Ridge（α=100）+ 5-fold CV + Permutation test（5,000回）。",
    ),
    Slide(
        number=8,
        title="Bootstrap分散分析",
        images=["fig_bootstrap_variance.png"],
        conclusion=(
            "95%CIがゼロを跨がない特徴量を「影響が強い特徴量」として同定し、"
            "SD/CIベースで係数の安定性を評価する。"
        ),
        methods_note="Bootstrap 500回リサンプリング（N=120復元抽出）。各特徴量の回帰係数の平均・SD・95%CIを算出。",
    ),
    Slide(
        number=9,
        title="ベースライン検証（3条件比較）",
        images=[],
        conclusion=(
            "条件3（ランダム）で予測力が消失 → LLMはテキスト内容を利用している。"
            "条件2（要約のみ）でもr_obsが高いが、Cronbach's αは崩壊（平均0.04〜0.30）"
            " → 一貫した性格像は構成できていない。"
        ),
        methods_note=(
            "条件1: テキスト全文（既存結果）。条件2: 4統計量のみ（発話数・平均発話長・会話長・フィラー数）。"
            "条件3: 別話者テキスト（derangement, seed=42）。各条件で4教師アンサンブル → Ridge + Permutation test（5,000回）。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>3条件比較: r<sub>obs</sub>（Holm補正後）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>Trait</th><th>条件1<br/>テキスト</th><th>条件2<br/>要約のみ</th><th>条件3<br/>ランダム</th>"
            "</tr></thead>"
            "<tbody>"
            '<tr><td>O</td><td><b>0.410</b> *</td><td><b>0.528</b> *</td><td>0.161</td></tr>'
            '<tr><td>C</td><td><b>0.432</b> *</td><td><b>0.640</b> *</td><td>−0.080</td></tr>'
            '<tr><td>E</td><td>0.234</td><td><b>0.302</b> *</td><td>−0.072</td></tr>'
            '<tr><td>A</td><td><b>0.449</b> *</td><td><b>0.374</b> *</td><td>0.071</td></tr>'
            '<tr><td>N</td><td><b>0.317</b> *</td><td><b>0.711</b> *</td><td>−0.280</td></tr>'
            "</tbody></table>"
            "<h3>Cronbach's α（条件2 vs 条件1）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>Trait</th><th>条件1 α</th><th>条件2 α<br/>（4教師平均）</th><th>判定</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>O</td><td>≥ 0.78</td><td>0.063</td><td>崩壊</td></tr>"
            "<tr><td>C</td><td>≥ 0.78</td><td>0.176</td><td>崩壊</td></tr>"
            "<tr><td>E</td><td>≥ 0.78</td><td>0.041</td><td>崩壊</td></tr>"
            "<tr><td>A</td><td>≥ 0.78</td><td>0.194</td><td>崩壊</td></tr>"
            "<tr><td>N</td><td>≥ 0.78</td><td>0.298</td><td>崩壊</td></tr>"
            "</tbody></table>"
            "<h3>解釈</h3>"
            "<ul>"
            "<li>条件3の崩壊 → LLMはテキスト内容を読んで性格推定している</li>"
            "<li>条件2のα崩壊 → 統計量だけでは一貫した性格像を構成できない</li>"
            "<li>条件2のr_obs高値 → 19特徴量の一部（PG系・FILL系）が表層統計量と相関するため交絡経路で相関が生じる</li>"
            "<li>→ 特徴量によるハーネス（検証レイヤー）の必要性を裏付ける</li>"
            "</ul>"
            "</div>"
        ),
    ),
]


# --- HTML / CSS テンプレート -------------------------------------------------

CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Meiryo", sans-serif;
  background: #f5f5f5;
  color: #333;
}
.slide {
  width: 960px;
  min-height: 720px;
  margin: 40px auto;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.10);
  padding: 48px 56px 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  page-break-after: always;
}
.slide-number {
  align-self: flex-end;
  font-size: 14px;
  color: #999;
  margin-bottom: 8px;
}
.slide-title {
  font-size: 32px;
  font-weight: 700;
  text-align: center;
  margin-bottom: 24px;
  color: #1a1a2e;
}
.slide-images {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 24px;
  width: 100%;
  margin-bottom: 24px;
}
.slide-images img {
  max-width: 100%;
  max-height: 440px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
}
.slide-images.multi img {
  max-width: 48%;
  max-height: 380px;
}
.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 400px;
  height: 300px;
  background: #fafafa;
  border: 2px dashed #ccc;
  border-radius: 8px;
  color: #999;
  font-size: 16px;
  text-align: center;
  padding: 16px;
}
.slide-conclusion {
  font-size: 20px;
  line-height: 1.6;
  text-align: center;
  color: #444;
  padding: 16px 24px;
  background: #f0f4ff;
  border-radius: 6px;
  width: 100%;
}
.methods-note {
  font-size: 14px;
  line-height: 1.5;
  color: #666;
  background: #f8f8f0;
  border-left: 3px solid #b0b060;
  padding: 8px 16px;
  margin-bottom: 12px;
  width: 100%;
  border-radius: 0 4px 4px 0;
}
.methods-text {
  flex: 1;
  width: 100%;
  text-align: left;
  font-size: 18px;
  line-height: 1.8;
  padding: 8px 16px;
}
.methods-text h3 {
  font-size: 22px;
  color: #1a1a2e;
  margin: 16px 0 8px;
}
.methods-text ul {
  margin: 0 0 8px 24px;
}
.methods-text li {
  margin-bottom: 4px;
}
.feature-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 15px;
  margin: 8px 0;
}
.feature-table th, .feature-table td {
  border: 1px solid #ccc;
  padding: 6px 10px;
  text-align: left;
}
.feature-table thead th {
  background: #e8edf5;
  font-weight: 700;
  text-align: center;
}
.feature-table tbody tr:nth-child(even) {
  background: #f9f9f9;
}
.slide-content {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  margin-bottom: 24px;
  overflow-y: auto;
}
.nav {
  text-align: center;
  font-size: 14px;
  color: #888;
  margin: 8px 0 32px;
}
.nav-link {
  color: #2166ac;
  text-decoration: none;
  font-weight: 500;
}
.nav-link:hover {
  text-decoration: underline;
}
.slide-images img {
  cursor: pointer;
  transition: opacity 0.15s;
}
.slide-images img:hover {
  opacity: 0.85;
}
.modal-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.8);
  z-index: 1000;
  justify-content: center;
  align-items: center;
  cursor: zoom-out;
}
.modal-overlay.active {
  display: flex;
}
.modal-overlay img {
  max-width: 95vw;
  max-height: 95vh;
  border-radius: 8px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.5);
}
"""


def _image_html(filename: str, out_dir: Path) -> str:
    """画像が存在すればimgタグ、なければプレースホルダーを返す"""
    if (out_dir / filename).exists():
        return f'<img src="{filename}" alt="{filename}">'
    return f'<div class="placeholder">[図表未生成: {filename}]</div>'


def generate_slides_html(out_dir: Path) -> str:
    """9枚のスライド（Methods 2枚 + Results 7枚）を含む自己完結型HTMLを生成する"""
    total = len(SLIDES)
    slide_blocks: list[str] = []

    for slide in SLIDES:
        # コンテンツセクション: html_contentがあればそれを使い、なければ画像を表示
        if slide.html_content:
            content_html = (
                f'      <div class="slide-content">\n'
                f'        {slide.html_content}\n'
                f'      </div>'
            )
        else:
            multi_class = " multi" if len(slide.images) > 1 else ""
            images_html = "\n        ".join(
                _image_html(img, out_dir) for img in slide.images
            )
            content_html = (
                f'      <div class="slide-images{multi_class}">\n'
                f'        {images_html}\n'
                f'      </div>'
            )

        # Methods注釈（各スライドの分析手法）
        methods_note_html = ""
        if slide.methods_note:
            methods_note_html = (
                f'      <div class="methods-note">📐 {slide.methods_note}</div>\n'
            )

        # ナビゲーション
        nav_parts: list[str] = []
        if slide.number > 1:
            nav_parts.append(
                f'<a href="#slide-{slide.number - 1}" class="nav-link">← prev</a>'
            )
        nav_parts.append(f"{slide.number} / {total}")
        if slide.number < total:
            nav_parts.append(
                f'<a href="#slide-{slide.number + 1}" class="nav-link">next →</a>'
            )
        nav_html = " &nbsp;|&nbsp; ".join(nav_parts)

        block = f"""\
    <div class="slide" id="slide-{slide.number}">
      <div class="slide-number">Slide {slide.number} / {total}</div>
      <div class="slide-title">{slide.title}</div>
{methods_note_html}{content_html}
      <div class="slide-conclusion">{slide.conclusion}</div>
    </div>
    <div class="nav">{nav_html}</div>"""
        slide_blocks.append(block)

    slides_body = "\n\n".join(slide_blocks)

    html = f"""\
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>紙芝居スライド — 相互行為特徴量の定量化指標の提案</title>
  <style>
{CSS}  </style>
</head>
<body>

{slides_body}

<div class="modal-overlay" id="imgModal">
  <img id="modalImg" src="" alt="拡大表示">
</div>

<script>
// 画像クリックでモーダル拡大
document.querySelectorAll('.slide-images img').forEach(img => {{
  img.addEventListener('click', () => {{
    const modal = document.getElementById('imgModal');
    document.getElementById('modalImg').src = img.src;
    modal.classList.add('active');
  }});
}});
document.getElementById('imgModal').addEventListener('click', () => {{
  document.getElementById('imgModal').classList.remove('active');
}});
// Escキーでモーダルを閉じる
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('imgModal').classList.remove('active');
}});
// 左右キーでスライド移動
document.addEventListener('keydown', e => {{
  if (document.getElementById('imgModal').classList.contains('active')) return;
  const slides = document.querySelectorAll('.slide');
  let current = 0;
  slides.forEach((s, i) => {{
    const rect = s.getBoundingClientRect();
    if (rect.top < window.innerHeight / 2 && rect.bottom > 0) current = i;
  }});
  if (e.key === 'ArrowRight' && current < slides.length - 1) {{
    slides[current + 1].scrollIntoView({{ behavior: 'smooth' }});
  }} else if (e.key === 'ArrowLeft' && current > 0) {{
    slides[current - 1].scrollIntoView({{ behavior: 'smooth' }});
  }}
}});
</script>

</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="紙芝居スライド（1枚1図）のHTML生成"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports/paper_figs_v2/",
        help="出力ディレクトリ（画像もここから参照）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html = generate_slides_html(out_dir)
    out_path = out_dir / "kamishibai_slides.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ 紙芝居スライドを生成しました: {out_path}")


if __name__ == "__main__":
    main()
