#!/usr/bin/env python3
"""テキスト操作モジュール（純粋関数群）.

会話テキスト中の特定特徴量（フィラー、YES/NO応答、OIRマーカー）を
3段階（×0: 全削除, ×1: 元データ, ×3: 3倍）で操作する。

全関数は副作用なし。テスト容易性と再現性を確保。

Requirements: 1.1-1.10, 2.1-2.7, 3.1-3.4, 9.4
"""
from __future__ import annotations

import random
import re
import warnings
from typing import Any

# ── 定数: Dose Level ──
VALID_DOSE_LEVELS: list[int] = [0, 1, 3]

# ── CEJC注釈保護 ──
# (L), (F ...), (D ...), (W ...|...), (G ...|...) および (0.983) 等のポーズ長注釈
ANNOTATION_RE = re.compile(
    r"\([LFDWG][^)]*\)"  # (L), (F えっと), (D ン), (W ...|...), (G ...|...)
    r"|\(\d+\.?\d*\)"     # (0.983) 等のポーズ長注釈
)

# プレースホルダーのベース文字列（テキスト中に出現しない想定）
_PLACEHOLDER_BASE = "\x00ANNOT"


def protect_annotations(text: str) -> tuple[str, list[tuple[str, str]]]:
    """注釈をプレースホルダーに置換し、マッピング情報を返す.

    Args:
        text: 元テキスト

    Returns:
        (プレースホルダー置換済みテキスト, [(プレースホルダー, 元注釈), ...])
    """
    annotations: list[tuple[str, str]] = []
    protected = text
    for i, m in enumerate(ANNOTATION_RE.finditer(text)):
        placeholder = f"{_PLACEHOLDER_BASE}{i:04d}\x00"
        annotations.append((placeholder, m.group()))
    # 後ろから置換して位置ずれを防ぐ
    for i, m in reversed(list(enumerate(ANNOTATION_RE.finditer(text)))):
        placeholder = annotations[i][0]
        protected = protected[:m.start()] + placeholder + protected[m.end():]
    return protected, annotations


def restore_annotations(text: str, annotations: list[tuple[str, str]]) -> str:
    """プレースホルダーを元の注釈に復元する.

    Args:
        text: プレースホルダー含みテキスト
        annotations: protect_annotations() が返したマッピングリスト

    Returns:
        注釈復元済みテキスト
    """
    result = text
    for placeholder, original in annotations:
        result = result.replace(placeholder, original)
    return result


# ── フィラー検出パターン（extract_interaction_features_min.py と同一） ──
RE_ETO = re.compile(r"(えっと|えと)")
RE_E = re.compile(r"(えー+|えぇ+|え〜+)")
RE_ANO = re.compile(r"(あの)")

# 全フィラーを一括検出する統合パターン（位置情報取得用）
_RE_ALL_FILLERS = re.compile(r"えっと|えと|えー+|えぇ+|え〜+|あの")

# ── フィラー等長置換マップ ──
# えっと(3文字)→それで(3文字), えと(2文字)→それ(2文字), あの(2文字)→その(2文字)
# えー系は長さに応じて「でー」+パディングで等長置換
NEUTRAL_REPLACEMENTS: dict[str, str] = {
    "えっと": "それで",
    "えと": "それ",
    "あの": "その",
}

# ×3 挿入用フィラー候補
FILLER_INSERTIONS: list[str] = ["えっと", "えー", "あの"]


def _replace_e_series(match: re.Match) -> str:
    """えー系フィラーを等長の中立テキストで置換する.

    えー → でー, えーー → でーー, えぇ → でぇ, え〜 → で〜 等。
    先頭の「え」を「で」に置換するだけで等長が保たれる。
    """
    original = match.group()
    return "で" + original[1:]


def _count_fillers_raw(text: str) -> dict[str, int]:
    """テキスト中のフィラーを種類別にカウントする（注釈考慮なし）.

    extract_interaction_features_min.py の count_fillers() と同一ロジック:
    1. RE_ETO でえっと/えとをカウント
    2. RE_ETO を除去した後に RE_E でえー系をカウント
    3. RE_ANO であのをカウント

    Returns:
        {"eto": int, "e": int, "ano": int, "total": int}
    """
    eto = len(RE_ETO.findall(text))
    text_no_eto = RE_ETO.sub("", text)
    e = len(RE_E.findall(text_no_eto))
    ano = len(RE_ANO.findall(text))
    total = eto + e + ano
    return {"eto": eto, "e": e, "ano": ano, "total": total}


def count_fillers_in_text(text: str) -> int:
    """テキスト中のフィラー数をカウント（注釈内は除外）.

    CEJC注釈 (F えっと) 等の内部にあるフィラーは操作対象外のため除外する。

    Args:
        text: 会話テキスト

    Returns:
        フィラー総数（注釈内を除外）
    """
    if not text:
        return 0
    protected, _annotations = protect_annotations(text)
    return _count_fillers_raw(protected)["total"]


def _validate_dose_level(dose_level: int) -> None:
    """dose_level が有効値かチェックし、不正なら ValueError を送出する."""
    if dose_level not in VALID_DOSE_LEVELS:
        raise ValueError(
            f"Invalid dose_level={dose_level}. "
            f"Must be one of {VALID_DOSE_LEVELS}"
        )


def _make_log_dict(
    original_count: int,
    new_count: int,
    inserted: int,
    deleted: int,
    original_chars: int,
    new_chars: int,
) -> dict[str, Any]:
    """操作ログ dict を生成する."""
    return {
        "original_count": original_count,
        "new_count": new_count,
        "inserted": inserted,
        "deleted": deleted,
        "original_chars": original_chars,
        "new_chars": new_chars,
    }


# ---------------------------------------------------------------------------
# フィラー操作
# ---------------------------------------------------------------------------

def _replace_all_fillers(text: str) -> str:
    """テキスト中の全フィラーを等長の中立テキストで置換する（×0用）.

    置換順序:
    1. RE_ETO (えっと/えと) → それで/それ
    2. RE_E (えー系) → でー系（先頭「え」→「で」）
    3. RE_ANO (あの) → その
    """
    # えっと/えと → それで/それ
    def _replace_eto(m: re.Match) -> str:
        return NEUTRAL_REPLACEMENTS[m.group()]

    result = RE_ETO.sub(_replace_eto, text)
    # えー系 → でー系
    result = RE_E.sub(_replace_e_series, result)
    # あの → その
    result = RE_ANO.sub(lambda m: NEUTRAL_REPLACEMENTS["あの"], result)
    return result


def _find_insertion_positions(line: str) -> list[int]:
    """1行（発話）内のフィラー挿入可能位置を特定する.

    挿入可能位置:
    - 行頭（位置0）
    - 読点（、）の直後
    - 注釈プレースホルダーの直後

    Returns:
        挿入可能な文字位置のリスト（昇順）
    """
    positions: list[int] = [0]  # 行頭は常に挿入可能

    # 読点の直後
    for m in re.finditer(r"、", line):
        pos = m.end()
        if pos < len(line):
            positions.append(pos)

    # 注釈プレースホルダーの直後
    for m in re.finditer(re.escape(_PLACEHOLDER_BASE) + r"\d{4}\x00", line):
        pos = m.end()
        if pos < len(line):
            positions.append(pos)

    return sorted(set(positions))


def _insert_fillers_in_lines(
    lines: list[str],
    n_insert: int,
    rng: random.Random,
) -> list[str]:
    """複数行にフィラーを挿入する（×3用）.

    各行の挿入可能位置を収集し、RNGで選択してフィラーを挿入する。
    テキスト長保存のため、挿入した文字数分だけ行末の空白や
    繰り返し文字を削減する。

    Args:
        lines: 行リスト（プレースホルダー置換済み）
        n_insert: 挿入するフィラー数
        rng: 乱数生成器

    Returns:
        フィラー挿入済みの行リスト
    """
    if n_insert <= 0:
        return lines

    # 全行の挿入可能位置を収集: (line_idx, char_pos)
    all_positions: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        positions = _find_insertion_positions(line)
        for pos in positions:
            all_positions.append((i, pos))

    if not all_positions:
        return lines

    result = list(lines)
    total_inserted_chars = 0

    # 挿入位置をランダムに選択（重複許可、同一位置に複数挿入可能）
    chosen = [rng.choice(all_positions) for _ in range(n_insert)]
    # 同一行内で後ろから挿入するためソート
    chosen.sort(key=lambda x: (x[0], -x[1]))

    # 行ごとにグループ化して挿入
    from collections import defaultdict
    by_line: dict[int, list[int]] = defaultdict(list)
    for line_idx, char_pos in chosen:
        by_line[line_idx].append(char_pos)

    for line_idx in by_line:
        positions_in_line = sorted(by_line[line_idx], reverse=True)
        line = result[line_idx]
        for pos in positions_in_line:
            filler = rng.choice(FILLER_INSERTIONS)
            line = line[:pos] + filler + line[pos:]
            total_inserted_chars += len(filler)
        result[line_idx] = line

    # テキスト長保存: 挿入分を各行末から削減
    if total_inserted_chars > 0:
        result = _trim_lines_to_budget(result, total_inserted_chars, rng)

    return result


def _trim_lines_to_budget(
    lines: list[str],
    chars_to_remove: int,
    rng: random.Random,
) -> list[str]:
    """行末から文字を削減してテキスト長を調整する.

    削減対象: 行末の空白、句読点の繰り返し、一般文字。
    フィラーやプレースホルダーは削減しない。

    Args:
        lines: 行リスト
        chars_to_remove: 削減する文字数
        rng: 乱数生成器

    Returns:
        削減済みの行リスト
    """
    result = list(lines)
    remaining = chars_to_remove

    # 長い行から優先的に削減（短い行を壊さないため）
    line_lengths = [(len(line), i) for i, line in enumerate(result) if line.strip()]
    line_lengths.sort(reverse=True)

    for _, idx in line_lengths:
        if remaining <= 0:
            break
        line = result[idx]
        # 行末から安全に削減できる文字数を計算
        # プレースホルダーを含まない末尾部分のみ対象
        safe_end = line
        # プレースホルダーの最後の位置を見つける
        last_ph_end = 0
        for m in re.finditer(re.escape(_PLACEHOLDER_BASE) + r"\d{4}\x00", line):
            last_ph_end = max(last_ph_end, m.end())

        # フィラーの最後の位置も保護
        for m in _RE_ALL_FILLERS.finditer(line):
            last_ph_end = max(last_ph_end, m.end())

        # 削減可能な末尾の長さ
        trimmable = len(line) - last_ph_end
        # 最低2文字は残す
        max_trim = max(0, trimmable - 2)
        trim_amount = min(remaining, max_trim)

        if trim_amount > 0:
            result[idx] = line[:len(line) - trim_amount]
            remaining -= trim_amount

    return result


def manipulate_fillers(
    text: str,
    dose_level: int,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    """フィラー率を指定Dose Levelで操作する.

    Args:
        text: 元の会話テキスト（複数行、1行=1発話）
        dose_level: 操作倍率（0, 1, 3）
        seed: 乱数シード

    Returns:
        (操作後テキスト, 操作ログdict)
        操作ログ: {original_count, new_count, inserted, deleted,
                   original_chars, new_chars}

    Raises:
        ValueError: dose_level が 0, 1, 3 以外の場合
    """
    _validate_dose_level(dose_level)

    original_chars = len(text)

    if not text:
        return text, _make_log_dict(0, 0, 0, 0, 0, 0)

    # ×1: 恒等操作
    if dose_level == 1:
        count = count_fillers_in_text(text)
        return text, _make_log_dict(count, count, 0, 0, original_chars, original_chars)

    # 注釈を保護
    protected, annotations = protect_annotations(text)
    original_count = _count_fillers_raw(protected)["total"]

    if dose_level == 0:
        # ×0: 全フィラーを等長中立テキストで置換
        if original_count == 0:
            return text, _make_log_dict(0, 0, 0, 0, original_chars, original_chars)
        replaced = _replace_all_fillers(protected)
        result = restore_annotations(replaced, annotations)
        new_count = count_fillers_in_text(result)
        new_chars = len(result)
        return result, _make_log_dict(
            original_count, new_count, 0, original_count,
            original_chars, new_chars,
        )

    # dose_level == 3: フィラーを約3倍に増加
    rng = random.Random(seed)
    lines = protected.split("\n")

    if original_count == 0:
        # フィラー0個の場合: 各行の文頭にフィラーを挿入
        # 空行でない行にのみ挿入
        non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
        n_insert = min(len(non_empty_indices), 3)  # 最大3個挿入
        chosen_indices = rng.sample(non_empty_indices, n_insert) if non_empty_indices else []
        total_inserted_chars = 0
        for idx in chosen_indices:
            filler = rng.choice(FILLER_INSERTIONS)
            lines[idx] = filler + lines[idx]
            total_inserted_chars += len(filler)
        if total_inserted_chars > 0:
            lines = _trim_lines_to_budget(lines, total_inserted_chars, rng)
        manipulated = "\n".join(lines)
        result = restore_annotations(manipulated, annotations)
        new_count = count_fillers_in_text(result)
        new_chars = len(result)
        char_change_pct = abs(new_chars - original_chars) / original_chars * 100 if original_chars > 0 else 0
        if char_change_pct > 5:
            warnings.warn(
                f"Text length change {char_change_pct:.1f}% exceeds ±5% limit "
                f"(original={original_chars}, new={new_chars})"
            )
        return result, _make_log_dict(
            original_count, new_count, new_count, 0,
            original_chars, new_chars,
        )

    # フィラーが1個以上ある場合: 目標 = 元の3倍
    target_count = original_count * 3
    n_insert = target_count - original_count  # = original_count * 2

    lines = _insert_fillers_in_lines(lines, n_insert, rng)
    manipulated = "\n".join(lines)
    result = restore_annotations(manipulated, annotations)
    new_count = count_fillers_in_text(result)
    new_chars = len(result)

    char_change_pct = abs(new_chars - original_chars) / original_chars * 100 if original_chars > 0 else 0
    if char_change_pct > 5:
        warnings.warn(
            f"Text length change {char_change_pct:.1f}% exceeds ±5% limit "
            f"(original={original_chars}, new={new_chars})"
        )

    return result, _make_log_dict(
        original_count, new_count, n_insert, 0,
        original_chars, new_chars,
    )


# ---------------------------------------------------------------------------
# YES/NO応答操作
# ---------------------------------------------------------------------------

# YES/NO応答パターン（extract_interaction_features_min.py の YESNO_PREFIXES と同一）
YESNO_PREFIXES: list[str] = [
    "はい", "うん", "ええ", "そう", "そうです", "もちろん", "了解",
    "OK", "オーケー", "いいえ", "いや", "ううん", "違う", "ちがう",
    "だめ", "無理", "そうじゃない",
]

# 相槌パターン（extract_interaction_features_min.py の AIZUCHI_PREFIXES と同一）
AIZUCHI_PREFIXES: list[str] = [
    "はい", "うん", "ええ", "そう", "そうそう", "そっか", "なるほど",
    "へえ", "了解", "わかった", "わかりました", "OK", "オーケー",
    "あー", "ああ", "うーん",
]

# YES/NO → 中立応答の置換マップ
_YESNO_NEUTRAL_MAP: dict[str, str] = {
    "はい": "ほう",
    "うん": "ふーん",
    "ええ": "へえ",
    "そうです": "なるほど",
    "そう": "ふむ",
    "もちろん": "なるほど",
    "了解": "ふむ",
    "OK": "ふむ",
    "オーケー": "なるほど",
    "いいえ": "ふーん",
    "いや": "ふむ",
    "ううん": "ふーん",
    "違う": "ふむ",
    "ちがう": "ふーん",
    "だめ": "ふむ",
    "無理": "ふむ",
    "そうじゃない": "なるほどね",
}

# 相槌 → YES/NO変換マップ（×3用）
# 相槌のうちYESNOでないものをYES/NOに変換
_AIZUCHI_TO_YESNO_MAP: dict[str, str] = {
    "そうそう": "そうです",
    "そっか": "そうです",
    "なるほど": "はい",
    "へえ": "ええ",
    "わかった": "はい",
    "わかりました": "はい",
    "あー": "うん",
    "ああ": "うん",
    "うーん": "うん",
}


def _count_yesno_in_lines(lines: list[str]) -> int:
    """行リスト中のYES/NO応答行数をカウントする."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in YESNO_PREFIXES):
            count += 1
    return count


def _count_aizuchi_non_yesno(lines: list[str]) -> list[tuple[int, str]]:
    """相槌だがYES/NOでない行を検出する.

    Returns:
        [(行インデックス, マッチした相槌プレフィックス), ...]
    """
    results: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # YES/NOに該当する場合はスキップ
        is_yn = any(stripped.startswith(p) for p in YESNO_PREFIXES)
        if is_yn:
            continue
        # 相槌に該当するか
        for prefix in AIZUCHI_PREFIXES:
            if stripped.startswith(prefix) and prefix in _AIZUCHI_TO_YESNO_MAP:
                results.append((i, prefix))
                break
    return results


def manipulate_yesno(
    text: str,
    dose_level: int,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    """YES/NO応答率を指定Dose Levelで操作する.

    Args:
        text: 元の会話テキスト（複数行、1行=1発話）
        dose_level: 操作倍率（0, 1, 3）
        seed: 乱数シード

    Returns:
        (操作後テキスト, 操作ログdict)

    Raises:
        ValueError: dose_level が 0, 1, 3 以外の場合
    """
    _validate_dose_level(dose_level)

    original_chars = len(text)

    if not text:
        return text, _make_log_dict(0, 0, 0, 0, 0, 0)

    lines = text.split("\n")
    original_count = _count_yesno_in_lines(lines)

    # ×1: 恒等操作
    if dose_level == 1:
        return text, _make_log_dict(
            original_count, original_count, 0, 0,
            original_chars, original_chars,
        )

    if dose_level == 0:
        # ×0: YES/NO応答を中立応答で置換
        if original_count == 0:
            return text, _make_log_dict(0, 0, 0, 0, original_chars, original_chars)

        new_lines = []
        deleted = 0
        for line in lines:
            stripped = line.strip()
            replaced = False
            # 長いプレフィックスから先にマッチ（「そうです」→「そう」の順序問題回避）
            for prefix in sorted(YESNO_PREFIXES, key=len, reverse=True):
                if stripped.startswith(prefix):
                    neutral = _YESNO_NEUTRAL_MAP.get(prefix, "ふむ")
                    new_line = neutral + stripped[len(prefix):]
                    new_lines.append(new_line)
                    deleted += 1
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        result = "\n".join(new_lines)
        new_count = _count_yesno_in_lines(new_lines)
        new_chars = len(result)
        return result, _make_log_dict(
            original_count, new_count, 0, deleted,
            original_chars, new_chars,
        )

    # dose_level == 3: 相槌をYES/NO応答に変換して約3倍に
    rng = random.Random(seed)
    target_count = original_count * 3
    n_needed = target_count - original_count

    if n_needed <= 0 and original_count == 0:
        # YES/NOが0個の場合: 相槌行をYES/NOに変換
        n_needed = 3

    # 相槌（非YES/NO）行を検出
    candidates = _count_aizuchi_non_yesno(lines)
    rng.shuffle(candidates)
    n_convert = min(n_needed, len(candidates))

    new_lines = list(lines)
    inserted = 0
    convert_set = set()
    for idx, prefix in candidates[:n_convert]:
        convert_set.add(idx)

    for idx, prefix in candidates[:n_convert]:
        line = new_lines[idx]
        stripped = line.strip()
        replacement = _AIZUCHI_TO_YESNO_MAP[prefix]
        new_lines[idx] = replacement + stripped[len(prefix):]
        inserted += 1

    result = "\n".join(new_lines)
    new_count = _count_yesno_in_lines(new_lines)
    new_chars = len(result)

    char_change_pct = abs(new_chars - original_chars) / original_chars * 100 if original_chars > 0 else 0
    if char_change_pct > 5:
        warnings.warn(
            f"YESNO: Text length change {char_change_pct:.1f}% exceeds ±5% limit "
            f"(original={original_chars}, new={new_chars})"
        )

    return result, _make_log_dict(
        original_count, new_count, inserted, 0,
        original_chars, new_chars,
    )


# ---------------------------------------------------------------------------
# OIR（修復開始標識）操作
# ---------------------------------------------------------------------------

# OIRマーカーパターン（extract_interaction_features_min.py の OIR_PREFIXES と同一）
OIR_PREFIXES: list[str] = [
    "え？", "えっ", "えっ？", "ん？", "は？", "なに？", "何？",
    "もう一回", "もう一度", "どういうこと", "どういう意味",
    "聞こえ", "聞き取", "わから", "分から",
]

# OIR → 中立応答の置換マップ
_OIR_NEUTRAL_MAP: dict[str, str] = {
    "え？": "ふむ",
    "えっ": "ふむ",
    "えっ？": "ふむ？",
    "ん？": "ふむ",
    "は？": "ふむ",
    "なに？": "ふーん",
    "何？": "ふむ",
    "もう一回": "なるほど",
    "もう一度": "なるほど",
    "どういうこと": "なるほどね",
    "どういう意味": "なるほどね",
    "聞こえ": "なるほ",
    "聞き取": "なるほ",
    "わから": "なるほ",
    "分から": "なるほ",
}

# OIR挿入用マーカー候補
_OIR_INSERTIONS: list[str] = ["え？", "えっ", "ん？"]

# 中立応答 → OIR変換マップ（×3用: 相槌をOIRに変換）
_AIZUCHI_TO_OIR_MAP: dict[str, str] = {
    "そうそう": "えっ",
    "そっか": "え？",
    "なるほど": "えっ",
    "へえ": "え？",
    "あー": "ん？",
    "ああ": "ん？",
    "うーん": "ん？",
}


def _count_oir_in_lines(lines: list[str]) -> int:
    """行リスト中のOIRマーカー行数をカウントする."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in OIR_PREFIXES):
            count += 1
    return count


def _count_aizuchi_non_oir(lines: list[str]) -> list[tuple[int, str]]:
    """相槌だがOIRでない行を検出する.

    Returns:
        [(行インデックス, マッチした相槌プレフィックス), ...]
    """
    results: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # OIRに該当する場合はスキップ
        is_oir = any(stripped.startswith(p) for p in OIR_PREFIXES)
        if is_oir:
            continue
        # 相槌に該当するか
        for prefix in AIZUCHI_PREFIXES:
            if stripped.startswith(prefix) and prefix in _AIZUCHI_TO_OIR_MAP:
                results.append((i, prefix))
                break
    return results


def manipulate_oir(
    text: str,
    dose_level: int,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    """OIRマーカー率を指定Dose Levelで操作する（統制条件）.

    Args:
        text: 元の会話テキスト（複数行、1行=1発話）
        dose_level: 操作倍率（0, 1, 3）
        seed: 乱数シード

    Returns:
        (操作後テキスト, 操作ログdict)

    Raises:
        ValueError: dose_level が 0, 1, 3 以外の場合
    """
    _validate_dose_level(dose_level)

    original_chars = len(text)

    if not text:
        return text, _make_log_dict(0, 0, 0, 0, 0, 0)

    lines = text.split("\n")
    original_count = _count_oir_in_lines(lines)

    # ×1: 恒等操作
    if dose_level == 1:
        return text, _make_log_dict(
            original_count, original_count, 0, 0,
            original_chars, original_chars,
        )

    if dose_level == 0:
        # ×0: OIRマーカーを中立応答で置換
        if original_count == 0:
            return text, _make_log_dict(0, 0, 0, 0, original_chars, original_chars)

        new_lines = []
        deleted = 0
        for line in lines:
            stripped = line.strip()
            replaced = False
            for prefix in sorted(OIR_PREFIXES, key=len, reverse=True):
                if stripped.startswith(prefix):
                    neutral = _OIR_NEUTRAL_MAP.get(prefix, "ふむ")
                    new_line = neutral + stripped[len(prefix):]
                    new_lines.append(new_line)
                    deleted += 1
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        result = "\n".join(new_lines)
        new_count = _count_oir_in_lines(new_lines)
        new_chars = len(result)
        return result, _make_log_dict(
            original_count, new_count, 0, deleted,
            original_chars, new_chars,
        )

    # dose_level == 3: 相槌をOIRマーカーに変換して約3倍に
    rng = random.Random(seed)
    target_count = original_count * 3
    n_needed = target_count - original_count

    if n_needed <= 0 and original_count == 0:
        n_needed = 3

    # 相槌（非OIR）行を検出
    candidates = _count_aizuchi_non_oir(lines)
    rng.shuffle(candidates)
    n_convert = min(n_needed, len(candidates))

    new_lines = list(lines)
    inserted = 0
    for idx, prefix in candidates[:n_convert]:
        line = new_lines[idx]
        stripped = line.strip()
        replacement = _AIZUCHI_TO_OIR_MAP[prefix]
        new_lines[idx] = replacement + stripped[len(prefix):]
        inserted += 1

    result = "\n".join(new_lines)
    new_count = _count_oir_in_lines(new_lines)
    new_chars = len(result)

    char_change_pct = abs(new_chars - original_chars) / original_chars * 100 if original_chars > 0 else 0
    if char_change_pct > 5:
        warnings.warn(
            f"OIR: Text length change {char_change_pct:.1f}% exceeds ±5% limit "
            f"(original={original_chars}, new={new_chars})"
        )

    return result, _make_log_dict(
        original_count, new_count, inserted, 0,
        original_chars, new_chars,
    )
