#!/usr/bin/env python3
"""紙芝居スライド生成スクリプト

現行原稿（5次元均等・subject-wise split 版）の構成に合わせたHTML生成スクリプト。
Methods 2枚 + 本文Results 5枚 + 付録 2枚 = 計9枚。
各スライドに (a) タイトル、(b) 図表またはテキスト/HTMLコンテンツ、(c) 結論テキスト（1〜2文）を含む。
画像不在時はプレースホルダーテキスト「[図表未生成: {filename}]」を表示する。

原稿との対応:
  Slide 3-4 → 本文 記述統計量・相関分析
  Slide 5   → 本文 性格特性との関連（主結果。GroupKFold による subject-wise split）
  Slide 6   → 本文 各性格次元に寄与する相互行為特徴量（一致特徴量）
  Slide 7   → 本文 交絡変数の統制
  Slide 8-9 → 付録 コーパス基本情報との関連／ベースライン検証

3段階Ridge回帰は原稿で本文から外し付録の探索的検討に移したため、
このスライドでも扱わない。埋め込む図はすべて現行原稿の掲載図である。

数値の出所（すべて reports/paper_figs_v2/ 配下の生成表と一致させること）:
  Slide 5 → tab_ensemble_permutation.tex
  Slide 6 → tab_coef_all5.tex
  Slide 7 → tab_confound_all5.tex
  Slide 9 → tab_baseline_conditions.tex ＋ 付録ベースライン検証の本文

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
        f'<th colspan="2">Classical（既存研究ベース: {len(classical)}個）</th>'
        f'<th colspan="2">Novel（新規提案: {len(novel)}個）</th>'
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
            "N=120の日常会話から19の相互行為特徴量を自動計測し、"
            "subject-wise splitのRidge回帰＋置換検定で仮想Big5との関連を評価する設計である。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>データ</h3>"
            "<ul>"
            "<li><b>コーパス:</b> CEJC（日本語日常会話コーパス）home2サブセット</li>"
            "<li><b>品質フィルタ:</b> HQ1（隣接ペア80以上・2,000文字以上・質問直後ペア10以上）</li>"
            "<li><b>サンプルサイズ:</b> N = 120レコード（66会話 × 話者、74話者）</li>"
            "<li><b>話者重複:</b> 74名中25名が2件以上の会話に参加（重複話者由来のレコードが59.2%）</li>"
            "<li><b>特徴量:</b> 19説明変数（Classical 10 + Novel 9）</li>"
            "</ul>"
            "<h3>手法</h3>"
            "<ul>"
            "<li><b>回帰モデル:</b> Ridge回帰（α = 100 固定）</li>"
            "<li><b>交差検証:</b> 5-fold <b>GroupKFold</b>（cejc_person_id単位の subject-wise split）</li>"
            "<li><b>統計検定:</b> 置換検定 5,000回（目的変数のみ並べ替え、seed = 42）</li>"
            "<li><b>多重比較:</b> 5次元にわたる Holm 補正</li>"
            "<li><b>係数の評価:</b> 係数ごとの置換検定 ＋ Bootstrap 500回の安定性</li>"
            "<li><b>教師ラベル:</b> 4 LLM教師の IPIP-NEO-120 item-level平均（アンサンブル）</li>"
            "</ul>"
            '<div class="note-box">'
            "話者重複があるため通常のKFoldでは同一話者が訓練foldと検証foldに跨り、"
            "話者固有の特徴量パターンを介したリークが生じる。"
            "本研究は全解析を GroupKFold で統一している。"
            "</div>"
            "</div>"
        ),
    ),
    Slide(
        number=2,
        title="特徴量の分類",
        images=[],
        conclusion=(
            "既存研究ベースのClassical 10特徴量と、"
            "会話分析・相互行為論に基づくNovel 9特徴量の2群、計19変数で構成される。"
        ),
        html_content=_build_feature_classification_table(),
    ),
    # --- 本文 Results スライド (3-7) ---
    Slide(
        number=3,
        title="提案特徴量の分布",
        images=["fig_feature_distribution.png"],
        conclusion=(
            "19特徴量はいずれも十分なばらつきを持ち、"
            "話者間の個人差を捉える指標として機能する。"
        ),
        methods_note="CEJC home2 HQ1（N=120）から抽出した19特徴量（Classical 10 + Novel 9）のバイオリンプロット。",
    ),
    Slide(
        number=4,
        title="カテゴリ内/間相関",
        images=["fig_corr_heatmap_block.png"],
        conclusion=(
            "同一カテゴリ内では高相関を示す一方、カテゴリ間は121ペア中109ペアが |r| < 0.30 であり、"
            "4カテゴリはおおむね独立した情報を持つ。"
        ),
        methods_note="19特徴量間のPearson相関行列。カテゴリ順: PG→FILL→IX→RESP。",
    ),
    Slide(
        number=5,
        title="性格特性との関連（主結果）",
        images=["fig_predicted_vs_observed.png"],
        conclusion=(
            "5次元すべてがHolm補正後も有意（r = 0.254〜0.423）。"
            "ただしEは p_Holm = 0.049 で境界的であり、α = 10・50 では非有意になる。"
        ),
        methods_note=(
            "4教師のIPIP-NEO-120 item-level平均 → Ridge（α=100）+ 5-fold GroupKFold"
            "（cejc_person_id単位）+ 置換検定5,000回 + 5次元Holm補正。"
            "散布図の r は fold平均のPearson r。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>置換検定の結果（仮想Big5アンサンブル、19特徴量）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>次元</th><th>r<sub>obs</sub></th><th>p</th><th>p<sub>Holm</sub></th><th>判定</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>O 開放性</td><td>0.337</td><td>0.0062</td><td>0.0124</td><td>*</td></tr>"
            "<tr><td>C 誠実性</td><td>0.423</td><td>0.0008</td><td>0.0040</td><td>*</td></tr>"
            "<tr><td>E 外向性</td><td>0.254</td><td>0.0490</td><td>0.0490</td><td>*（境界的）</td></tr>"
            "<tr><td>A 協調性</td><td>0.397</td><td>0.0022</td><td>0.0072</td><td>*</td></tr>"
            "<tr><td>N 神経症傾向</td><td>0.410</td><td>0.0018</td><td>0.0072</td><td>*</td></tr>"
            "</tbody></table>"
            '<div class="note-box">'
            "この値は subject-wise split（GroupKFold）によるもの。"
            "通常KFoldではO 0.410 / C 0.432 / E 0.234 / A 0.449 / N 0.317 となり、"
            "有意性の判定が変わるのはEのみである（付録に併記）。"
            "</div>"
            "</div>"
        ),
    ),
    Slide(
        number=6,
        title="各性格次元に寄与する相互行為特徴量",
        images=["fig_coef_all5.png"],
        conclusion=(
            "一致特徴量は5次元すべてで同定され計23個。うち9個が本研究で新たに定義した"
            "連鎖組織系（IX）・応答型系（RESP）であり、O・C・A・Nの4次元で寄与した。"
        ),
        methods_note=(
            "19特徴量×5次元の回帰係数（α=100、全データfit）。"
            "係数ごとの置換検定5,000回で p<0.05、かつ Bootstrap 500回の95%CIがゼロを除外する"
            "特徴量を「一致特徴量」と定義する（両方を満たすものだけを寄与とみなす）。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>一致特徴量の内訳</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>次元</th><th>O</th><th>C</th><th>E</th><th>A</th><th>N</th><th>計</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>一致特徴量数</td><td>6</td><td>5</td><td>3</td><td>7</td><td>2</td><td>23</td></tr>"
            "<tr><td>うち新規（Novel）</td><td>1</td><td>3</td><td>0</td><td>3</td><td>2</td><td>9</td></tr>"
            "</tbody></table>"
            "<h3>読み取り</h3>"
            "<ul>"
            "<li>タイミング系（PG）は5次元すべてに寄与し、沈黙系はO・E・Aで一貫して負</li>"
            "<li>Nは一致特徴量2個がいずれも新規特徴量（質問直後修復開始率・YES/NO応答率）で、"
            "しかもCとは逆符号</li>"
            "<li>Eは沈黙系3指標のみで、新規特徴量の寄与がない</li>"
            "<li>修復開始率（IX_oirmarker_rate）と沈黙長の変動係数（PG_pause_variability）は"
            "5次元のいずれでも一致特徴量にならなかった</li>"
            "</ul>"
            '<div class="note-box">'
            "「19の特徴量が本当に効いているのか」という問いに対しては、"
            "特徴量群を足し込む3段階Ridgeよりも、この係数レベルの結果が直接的な答えになる。"
            "</div>"
            "</div>"
        ),
    ),
    Slide(
        number=7,
        title="交絡変数の統制",
        images=[],
        conclusion=(
            "性別・年齢を説明変数に加えてもrは低下せず（Δr = −0.004〜+0.120）、"
            "特徴量とBig5の関連は人口統計の交絡では説明されない。"
        ),
        methods_note=(
            "Model A（19特徴量）と Model B（+性別・年齢）を同一の GroupKFold 設計で比較。"
            "置換1,000回。問うているのは「関連が残るか」であり「予測精度が上がるか」ではない。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>Model A（19特徴量）vs Model B（+性別・年齢）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>次元</th><th>Model A r</th><th>Model A p</th>"
            "<th>Model B r</th><th>Model B p</th><th>Δr</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>O</td><td>0.337</td><td>0.0070</td><td>0.401</td><td>0.0020</td><td>+0.064</td></tr>"
            "<tr><td>C</td><td>0.423</td><td>0.0020</td><td>0.447</td><td>0.0010</td><td>+0.024</td></tr>"
            "<tr><td>E</td><td>0.254</td><td>0.0559</td><td>0.254</td><td>0.0509</td><td>+0.000</td></tr>"
            "<tr><td>A</td><td>0.397</td><td>0.0030</td><td>0.517</td><td>0.0010</td><td>+0.120</td></tr>"
            "<tr><td>N</td><td>0.410</td><td>0.0030</td><td>0.406</td><td>0.0010</td><td>−0.004</td></tr>"
            "</tbody></table>"
            '<div class="note-box">'
            "特徴量を基準にして人口統計を足す向きで検証している。"
            "特徴量提案論文としては、人口統計から出発して特徴量を足す3段階Ridgeより"
            "素直な立て付けになる。"
            "</div>"
            "</div>"
        ),
    ),
    # --- 付録スライド (8-9) ---
    Slide(
        number=8,
        title="付録: コーパス基本情報との関連",
        images=["fig_metadata_gender.png", "fig_metadata_age.png"],
        conclusion=(
            "一部の特徴量に性別・年齢との有意な関連が認められる。"
            "この点はSlide 7の交絡統制で扱っている。"
        ),
        methods_note=(
            "性別: Mann-Whitney U検定（レコード単位で女性66件・男性54件）。"
            "年齢: Spearman ρ。欠損はペアワイズ削除。有意性はアスタリスクのみで示す。"
        ),
    ),
    Slide(
        number=9,
        title="付録: ベースライン検証（3条件比較）",
        images=[],
        conclusion=(
            "条件3（ランダムテキスト）で関連が消失 → LLMはテキストと話者の対応を読んでいる。"
            "条件2（要約のみ）はrが高いがCronbach's αが0.3以下に崩れており、"
            "「推定精度が高い」のではなく入力が乏しいときLLMが表層量へ依存することを示す。"
        ),
        methods_note=(
            "条件1: テキスト全文（本文の主結果と同一）。条件2: 4統計量のみ"
            "（発話数・平均発話長・会話長・フィラー数）。条件3: 別話者テキスト"
            "（derangement, seed=42）。3条件すべて本文と同一設定"
            "（19特徴量、α=100、5-fold GroupKFold、置換5,000回、Holm補正）で再計算。"
        ),
        html_content=(
            '<div class="methods-text">'
            "<h3>3条件の r<sub>obs</sub>（* は p<sub>Holm</sub> &lt; 0.05）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>次元</th><th>条件1<br/>テキスト</th><th>条件2<br/>要約のみ</th>"
            "<th>条件3<br/>ランダム</th><th>判定</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>O</td><td>0.337 *</td><td>0.546 *</td><td>0.021</td><td>条件3で消失</td></tr>"
            "<tr><td>C</td><td>0.423 *</td><td>0.649 *</td><td>−0.090</td><td>条件3で消失</td></tr>"
            "<tr><td>E</td><td>0.254 *</td><td>0.296 *</td><td>−0.032</td><td>条件3で消失</td></tr>"
            "<tr><td>A</td><td>0.397 *</td><td>0.479 *</td><td>0.201</td><td>条件3で消失</td></tr>"
            "<tr><td>N</td><td>0.410 *</td><td>0.748 *</td><td>−0.179</td><td>条件3で消失</td></tr>"
            "</tbody></table>"
            "<h3>Cronbach's α（回答の内的一貫性、4モデル平均）</h3>"
            '<table class="feature-table">'
            "<thead><tr>"
            "<th>次元</th><th>条件1</th><th>条件2</th><th>判定</th>"
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
            "<li>条件3で全次元が非有意（p<sub>Holm</sub> ≥ 0.587）→ 陰性対照として機能。"
            "プロンプトの構造や回答バイアスだけで関連が出ているのではない</li>"
            "<li>条件2はrが高いがαが0.3以下 → 項目間で一貫した性格像を作れていないのに"
            "相関だけが立っている。入力が4数値に限られるとLLMの出力が入力の単調関数に近づくため</li>"
            "<li>本文の一致特徴量23個のうち9個はIX系・RESP系で、条件2の4統計量からは"
            "算出できない → 提案特徴量の寄与は表層統計量に還元されない</li>"
            "<li>この3条件比較はLLM採点の性質を調べるもので、"
            "提案特徴量そのものの妥当性を直接評価するものではない</li>"
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
.note-box {
  font-size: 15px;
  line-height: 1.7;
  color: #444;
  background: #f4f7fb;
  border-left: 3px solid #2166ac;
  padding: 10px 16px;
  margin: 14px 0 4px;
  border-radius: 0 4px 4px 0;
}
/* 図と表を同じスライドに載せる場合は図をやや小さく抑える */
.slide.with-table .slide-images img {
  max-height: 320px;
}
.slide.with-table .slide-images.multi img {
  max-height: 280px;
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
    """9枚のスライド（Methods 2枚 + 本文Results 5枚 + 付録2枚）を含む自己完結型HTMLを生成する"""
    total = len(SLIDES)
    slide_blocks: list[str] = []

    for slide in SLIDES:
        # コンテンツセクション: 画像とhtml_contentの両方を持つスライドがあるため、
        # 画像を上、表・解説を下に積む（どちらか一方だけのスライドもそのまま動く）。
        parts: list[str] = []
        if slide.images:
            multi_class = " multi" if len(slide.images) > 1 else ""
            images_html = "\n        ".join(
                _image_html(img, out_dir) for img in slide.images
            )
            parts.append(
                f'      <div class="slide-images{multi_class}">\n'
                f'        {images_html}\n'
                f'      </div>'
            )
        if slide.html_content:
            parts.append(
                f'      <div class="slide-content">\n'
                f'        {slide.html_content}\n'
                f'      </div>'
            )
        content_html = "\n".join(parts)

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

        slide_class = "slide with-table" if (slide.images and slide.html_content) else "slide"
        block = f"""\
    <div class="{slide_class}" id="slide-{slide.number}">
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
