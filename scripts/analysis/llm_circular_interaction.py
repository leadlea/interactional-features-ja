#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM circular interaction verification pipeline.

Implements the "LLM → classical features → LLM" circular framework:
1. Select a target feature based on regression coefficients
2. Generate baseline dialogue with specified personality via LLM
3. Manipulate the target feature (increase/decrease) without personality info
4. Estimate Big5 personality from manipulated dialogue via virtual teacher protocol
5. Compute ΔC between baseline and intervention
6. Generate consistency report

This script uses AWS Bedrock (ap-northeast-1) for LLM calls.
It includes a --dry-run mode that generates mock data for testing
without API access.

Usage:
    # Dry run (no API calls):
    python scripts/analysis/llm_circular_interaction.py \\
        --model_id anthropic.claude-sonnet-4-20250514-v1:0 \\
        --target_feature FILL_rate_per_100chars \\
        --seed 42 \\
        --out_dir artifacts/analysis/results/circular_interaction \\
        --dry-run

    # Real run (requires AWS Bedrock access):
    python scripts/analysis/llm_circular_interaction.py \\
        --model_id anthropic.claude-sonnet-4-20250514-v1:0 \\
        --target_feature FILL_rate_per_100chars \\
        --seed 42 \\
        --out_dir artifacts/analysis/results/circular_interaction

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.7
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────
REGION = "ap-northeast-1"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0

# Feature selection criteria (from bootstrap + permutation analyses)
# Candidate features ranked by: bootstrap topk_rate, permutation significance,
# and manipulability on dialogue text.
FEATURE_CANDIDATES = {
    "FILL_rate_per_100chars": {
        "description": "フィラー率（100文字あたり）",
        "trait": "C",
        "regression_coef_sign": "+",
        "rationale": (
            "Bootstrap topk_rate高、sign_agree_rate高。"
            "対話テキスト上でフィラー（えっと、ええ、あの）の挿入・削除が明確に操作可能。"
        ),
        "fillers": ["えっと", "ええ", "あの", "まあ", "なんか", "こう", "その"],
    },
    "IX_yesno_rate": {
        "description": "YES/NO応答率",
        "trait": "C",
        "regression_coef_sign": "+",
        "rationale": (
            "Permutation係数検定で有意。"
            "YES/NO応答の挿入・削除は対話構造上操作可能。"
        ),
        "fillers": [],
    },
    "PG_speech_ratio": {
        "description": "発話率",
        "trait": "C",
        "regression_coef_sign": "+",
        "rationale": (
            "Bootstrap安定。発話量の増減は操作可能だが、"
            "対話の自然さへの影響が大きい。"
        ),
        "fillers": [],
    },
}

# Default target feature
DEFAULT_TARGET_FEATURE = "FILL_rate_per_100chars"

# ── IPIP-NEO-120 items for C (Conscientiousness) ────────────────────
# Subset of items used for quick estimation (10 items from C domain)
# Full protocol uses 24 items; here we use a representative subset
# for the circular interaction verification.
C_ITEMS_SUBSET = [
    {"item_id": 5, "text": "I am always prepared.", "reverse": 0},
    {"item_id": 25, "text": "I pay attention to details.", "reverse": 0},
    {"item_id": 45, "text": "I make a mess of things.", "reverse": 1},
    {"item_id": 65, "text": "I get chores done right away.", "reverse": 0},
    {"item_id": 85, "text": "I often forget to put things back in their proper place.", "reverse": 1},
    {"item_id": 10, "text": "I carry out my plans.", "reverse": 0},
    {"item_id": 30, "text": "I waste my time.", "reverse": 1},
    {"item_id": 50, "text": "I find it difficult to get down to work.", "reverse": 1},
    {"item_id": 70, "text": "I do things according to a plan.", "reverse": 0},
    {"item_id": 90, "text": "I continue until everything is perfect.", "reverse": 0},
]

SCALE = [
    "Very Inaccurate",
    "Moderately Inaccurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Accurate",
    "Very Accurate",
]
S2I = {s: i for i, s in enumerate(SCALE)}  # 0..4


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class CircularInteractionResult:
    """Result of a single circular interaction verification run.

    Attributes:
        target_feature: Name of the manipulated feature (e.g. FILL_rate_per_100chars).
        target_trait: Big5 dimension being tested (e.g. "C").
        baseline_score: Big5 score estimated from the unmodified baseline dialogue.
        intervention_increase_score: Big5 score after increasing the target feature.
        intervention_decrease_score: Big5 score after decreasing the target feature.
        delta_increase: Score change from baseline when feature is increased.
        delta_decrease: Score change from baseline when feature is decreased.
        direction_consistent: Whether manipulation direction and ΔC direction
            are consistent with the regression coefficient sign.
        regression_coef_sign: Sign of the regression coefficient ("+" or "-").
        notes: Limitations and caveats.
    """

    target_feature: str
    target_trait: str
    baseline_score: float
    intervention_increase_score: float
    intervention_decrease_score: float
    delta_increase: float
    delta_decrease: float
    direction_consistent: bool
    regression_coef_sign: str
    notes: str


# ── Prompt templates ─────────────────────────────────────────────────

PROMPT_DIALOGUE_GENERATION = """あなたは日本語の日常会話を生成するアシスタントです。

以下の性格特性を持つ話者Aと、普通の性格の話者Bの日本語日常会話を生成してください。

話者Aの性格特性:
- 誠実性（Conscientiousness）: {c_level}

会話の条件:
- 場面: 自宅での日常会話（家族・親しい友人との雑談）
- 長さ: 20〜30ターン程度
- 話題: 日常的な話題（予定、買い物、最近の出来事など）
- 形式: 各ターンは「話者A:」または「話者B:」で始める
- 自然な日本語の口語体で生成すること
- フィラー（えっと、ええ、あの、まあ等）や相槌（うん、そうだね等）を自然に含めること

話者Aの誠実性が{c_level_desc}ことが会話スタイルに反映されるようにしてください。

会話を生成してください:"""

PROMPT_FEATURE_MANIPULATION_INCREASE = """以下の日本語日常会話テキストについて、フィラー（えっと、ええ、あの、まあ、なんか、こう、その）の出現頻度を増加させてください。

重要な制約:
- 性格に関する情報は一切考慮しないでください
- 会話の内容や話題は変更しないでください
- フィラーの挿入のみを行い、発話の意味を変えないでください
- 各発話の冒頭や文中の自然な位置にフィラーを追加してください
- 元の会話形式（話者A: / 話者B:）を維持してください

元の会話:
{dialogue}

フィラーを増加させた会話を生成してください:"""

PROMPT_FEATURE_MANIPULATION_DECREASE = """以下の日本語日常会話テキストについて、フィラー（えっと、ええ、あの、まあ、なんか、こう、その）の出現頻度を減少させてください。

重要な制約:
- 性格に関する情報は一切考慮しないでください
- 会話の内容や話題は変更しないでください
- フィラーの削除のみを行い、発話の意味を変えないでください
- 可能な限り多くのフィラーを削除してください
- 元の会話形式（話者A: / 話者B:）を維持してください

元の会話:
{dialogue}

フィラーを減少させた会話を生成してください:"""

PROMPT_PERSONALITY_ESTIMATION = """あなたのタスクは、以下に示す参加者の日常会話テキストを根拠として、IPIP-NEO-120 の質問項目に回答することです。
あなた自身がこの会話の話者Aになりきって、その人の性格特性が反映されるように答えてください。
推論される性格特性に基づいて判断し、会話内容が示す傾向や行動をよく考えてください。

質問ごとに、最も適切な選択肢を次から1つだけ選んでください（必ず下の英語の文言をそのまま返してください）:

IPIP-NEO-120 question to answer:
{statement}

Participant's conversation:
{transcript}

Your response must be exactly one of:
Very Inaccurate
Moderately Inaccurate
Neither Accurate Nor Inaccurate
Moderately Accurate
Very Accurate

Do not include any explanation, punctuation, or additional text. Return only the exact phrase from the list above."""


# ── Bedrock API helpers ──────────────────────────────────────────────

def _extract_text_blocks(converse_response: dict) -> str:
    """Extract text content from Bedrock converse API response."""
    blocks = (
        converse_response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    texts = []
    for b in blocks:
        if isinstance(b, dict) and "text" in b:
            t = str(b.get("text") or "").strip()
            if t:
                texts.append(t)
    return "\n".join(texts).strip()


def _normalize_choice(text: str) -> str | None:
    """Normalize LLM response to one of the 5 IPIP-NEO scale choices."""
    if text is None:
        return None
    t = " ".join(str(text).strip().split())
    if t in S2I:
        return t
    tl = t.lower()
    for s in SCALE:
        if s.lower() in tl:
            return s
    return None


def _resolve_inference_profile(model_id: str, region: str) -> str:
    """Resolve a bare model ID to an inference profile ID if needed.

    Newer Bedrock models (e.g. Claude Sonnet 4, Claude Opus 4.5) require
    an inference profile ID for on-demand throughput.  If the caller passes
    a bare model ID (e.g. ``anthropic.claude-sonnet-4-20250514-v1:0``),
    this function prepends the appropriate cross-region prefix.

    Mapping:
        ap-northeast-1  → ``apac.``
        us-*            → ``us.``
        eu-*            → ``eu.``
        fallback        → ``global.``

    If the model_id already contains a prefix (has a dot before the
    provider name), it is returned unchanged.
    """
    # Already has a profile prefix (e.g. "apac.anthropic..." or "global.anthropic...")
    if "." in model_id.split(".")[0] or model_id.startswith(("apac.", "us.", "eu.", "global.")):
        return model_id

    # Determine prefix from region
    if region.startswith("ap-"):
        prefix = "apac"
    elif region.startswith("us-"):
        prefix = "us"
    elif region.startswith("eu-"):
        prefix = "eu"
    else:
        prefix = "global"

    resolved = f"{prefix}.{model_id}"
    logger.info("  Resolved model ID to inference profile: %s → %s", model_id, resolved)
    return resolved


def call_bedrock(
    client,
    model_id: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    seed: int | None = None,
    region: str = REGION,
) -> str:
    """Call Bedrock converse API with retry logic.

    Args:
        client: boto3 bedrock-runtime client.
        model_id: Bedrock model identifier (bare or inference profile).
        prompt: User prompt text.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility (model-dependent).
        region: AWS region (used to resolve inference profile prefix).

    Returns:
        Generated text response.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    resolved_model_id = _resolve_inference_profile(model_id, region)

    # Some models (e.g. Claude Sonnet 4) do not support the seed parameter
    # in additionalModelRequestFields.  Only pass seed for models known to
    # accept it (currently none of the Anthropic models do).
    _seed_unsupported_prefixes = ("anthropic.", "apac.anthropic.", "us.anthropic.",
                                  "eu.anthropic.", "global.anthropic.")
    seed_supported = not resolved_model_id.lower().startswith(_seed_unsupported_prefixes)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {}
            if seed_supported and seed is not None and int(seed) != 0:
                kwargs["additionalModelRequestFields"] = {"seed": int(seed)}

            response = client.converse(
                modelId=resolved_model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": int(max_tokens),
                    "temperature": float(temperature),
                    "topP": 1.0,
                },
                **kwargs,
            )

            out_text = _extract_text_blocks(response)
            if out_text:
                return out_text

            logger.warning(
                "Empty response on attempt %d/%d", attempt, MAX_RETRIES
            )
        except Exception as e:
            last_error = e
            logger.warning(
                "Bedrock API error on attempt %d/%d: %s",
                attempt,
                MAX_RETRIES,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)

    raise RuntimeError(
        f"Bedrock API call failed after {MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


# ── Mock data for dry-run mode ───────────────────────────────────────

def _generate_mock_dialogue(c_level: str, seed: int) -> str:
    """Generate a mock dialogue for dry-run testing."""
    rng = np.random.default_rng(seed)

    if c_level == "高":
        dialogue = (
            "話者A: えっと、今日の予定なんだけど、まず午前中に買い物に行こうと思ってるんだ。\n"
            "話者B: うん、いいね。何を買うの？\n"
            "話者A: あの、冷蔵庫の中を確認してリストを作ったんだけど、野菜と調味料が足りないみたい。\n"
            "話者B: ちゃんとリスト作ったんだ。偉いね。\n"
            "話者A: まあ、いつも計画的にやらないと気が済まないタイプだから。ええ、それで午後は掃除もしようかなと。\n"
            "話者B: 掃除も？今日忙しいね。\n"
            "話者A: そうだね。でも、えっと、やることは早めに片付けたい方なんだよね。\n"
            "話者B: なるほどね。じゃあ私も手伝うよ。\n"
            "話者A: ありがとう。あの、じゃあ分担を決めようか。キッチンは私がやるから、リビングをお願いしてもいい？\n"
            "話者B: うん、了解。\n"
            "話者A: えっと、あと洗濯物も取り込まないと。天気予報見たら午後から曇りだって。\n"
            "話者B: そうなんだ。じゃあ早めに取り込んだ方がいいね。\n"
            "話者A: うん。まあ、計画通りに進めば夕方には全部終わるはず。\n"
            "話者B: さすがだね。いつも段取りがいいよね。\n"
            "話者A: えっと、ありがとう。なんか、こういうの好きなんだよね、予定立てるの。\n"
            "話者B: わかるわかる。\n"
            "話者A: あの、それで夕飯は何にしようか。買い物のついでに食材も見てこようと思って。\n"
            "話者B: カレーとかどう？\n"
            "話者A: いいね。じゃあ、ええ、カレーの材料もリストに追加しておくね。\n"
            "話者B: お願い。\n"
        )
    else:
        dialogue = (
            "話者A: あー、今日何しよっか。\n"
            "話者B: うーん、買い物とか？\n"
            "話者A: 買い物かー。何買うんだっけ。冷蔵庫見てないや。\n"
            "話者B: 見てないの？じゃあ見てきたら？\n"
            "話者A: まあ、あとでいいや。なんか面倒くさい。\n"
            "話者B: いつもそうだよね。\n"
            "話者A: えっと、まあ、適当に行って適当に買えばいいかなって。\n"
            "話者B: リストとか作らないの？\n"
            "話者A: リスト？作んないよ、そんなの。行けばわかるし。\n"
            "話者B: 忘れ物しそうだけど。\n"
            "話者A: まあ、忘れたらまた行けばいいし。あの、それより今日天気いいから散歩でも行かない？\n"
            "話者B: 掃除は？\n"
            "話者A: 掃除？あー、明日でいいや。\n"
            "話者B: いつも明日って言うよね。\n"
            "話者A: なんか、今やる気出ないんだよね。ええ、まあそのうちやるよ。\n"
            "話者B: 本当に？\n"
            "話者A: たぶん。あの、とりあえず散歩行こうよ。\n"
            "話者B: はいはい。\n"
            "話者A: えっと、あ、洗濯物干してたっけ。\n"
            "話者B: 干してないでしょ。\n"
        )
    return dialogue


def _generate_mock_manipulated(dialogue: str, direction: str, seed: int) -> str:
    """Generate mock manipulated dialogue for dry-run testing."""
    lines = dialogue.strip().split("\n")
    fillers = ["えっと、", "ええ、", "あの、", "まあ、", "なんか、", "こう、", "その、"]
    rng = np.random.default_rng(seed)

    result = []
    for line in lines:
        if direction == "increase":
            # Insert fillers at natural positions
            if ":" in line:
                prefix, content = line.split(":", 1)
                filler = rng.choice(fillers)
                content = f" {filler}" + content
                result.append(f"{prefix}:{content}")
            else:
                result.append(line)
        else:  # decrease
            # Remove fillers
            modified = line
            for f in fillers:
                modified = modified.replace(f, "")
            # Clean up double spaces
            while "  " in modified:
                modified = modified.replace("  ", " ")
            result.append(modified)
    return "\n".join(result)


def _generate_mock_score(c_level: str, direction: str, seed: int) -> float:
    """Generate a mock Big5 C score for dry-run testing.

    Simulates plausible scores:
    - High C baseline: ~3.2
    - Low C baseline: ~1.8
    - Feature increase shifts score up (for positive coef features)
    - Feature decrease shifts score down
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.1)

    if c_level == "高":
        base = 3.2
    else:
        base = 1.8

    if direction == "baseline":
        return round(base + noise, 3)
    elif direction == "increase":
        return round(base + 0.15 + noise, 3)
    else:  # decrease
        return round(base - 0.15 + noise, 3)


# ── CircularInteractionPipeline ──────────────────────────────────────

class CircularInteractionPipeline:
    """LLM circular interaction verification pipeline.

    Steps:
    1. select_target_feature() — 回帰係数に基づく特徴量選定
    2. generate_baseline_dialogue() — LLMで指定性格の対話生成
    3. manipulate_feature() — 特徴量の増減操作（性格情報なし）
    4. estimate_personality() — 仮想教師プロトコルでBig5推定
    5. compute_delta() — ベースラインと介入後のΔC算出
    6. report_results() — 整合性レポート生成

    Args:
        model_id: Bedrock model identifier.
        target_feature: Feature to manipulate (default: FILL_rate_per_100chars).
        seed: Random seed for reproducibility.
        out_dir: Output directory for results.
        dry_run: If True, use mock data instead of API calls.
        region: AWS region for Bedrock (default: ap-northeast-1).
    """

    def __init__(
        self,
        model_id: str,
        target_feature: str = DEFAULT_TARGET_FEATURE,
        seed: int = 42,
        out_dir: str = "artifacts/analysis/results/circular_interaction",
        dry_run: bool = False,
        region: str = REGION,
    ):
        self.model_id = model_id
        self.target_feature = target_feature
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.dry_run = dry_run
        self.region = region

        # Validate target feature
        if target_feature not in FEATURE_CANDIDATES:
            raise ValueError(
                f"Unknown target feature: {target_feature}. "
                f"Available: {list(FEATURE_CANDIDATES.keys())}"
            )

        self.feature_info = FEATURE_CANDIDATES[target_feature]
        self.target_trait = self.feature_info["trait"]
        self.regression_coef_sign = self.feature_info["regression_coef_sign"]

        # State: populated during pipeline execution
        self._client = None
        self.baseline_dialogue_high: str | None = None
        self.baseline_dialogue_low: str | None = None
        self.manipulated_increase_high: str | None = None
        self.manipulated_decrease_high: str | None = None
        self.manipulated_increase_low: str | None = None
        self.manipulated_decrease_low: str | None = None
        self.scores: dict[str, float] = {}

    def _get_client(self):
        """Lazily initialize Bedrock client."""
        if self._client is None:
            if self.dry_run:
                logger.info("[DRY-RUN] Skipping Bedrock client initialization")
                return None
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    connect_timeout=10,
                    read_timeout=120,
                    retries={"max_attempts": 8, "mode": "adaptive"},
                ),
            )
        return self._client

    # ── Step 1: Feature selection ────────────────────────────────────

    def select_target_feature(self) -> dict:
        """Select target feature based on regression coefficient criteria.

        Selection criteria:
        1. Bootstrap topk_rate高（安定的に重要）
        2. Permutation係数検定で有意
        3. 操作が対話テキスト上で実現可能

        Returns:
            dict with feature name, description, trait, rationale.
        """
        logger.info("=" * 60)
        logger.info("Step 1: Feature selection")
        logger.info("  Target feature: %s", self.target_feature)
        logger.info("  Description: %s", self.feature_info["description"])
        logger.info("  Target trait: %s", self.target_trait)
        logger.info("  Regression coef sign: %s", self.regression_coef_sign)
        logger.info("  Rationale: %s", self.feature_info["rationale"])
        logger.info("=" * 60)

        return {
            "feature": self.target_feature,
            "description": self.feature_info["description"],
            "trait": self.target_trait,
            "regression_coef_sign": self.regression_coef_sign,
            "rationale": self.feature_info["rationale"],
        }

    # ── Step 2: Baseline dialogue generation ─────────────────────────

    def generate_baseline_dialogue(self) -> dict[str, str]:
        """Generate baseline dialogues for high-C and low-C speakers.

        Uses LLM to generate natural Japanese daily conversation with
        specified personality traits.

        Returns:
            dict with keys "high" and "low", each containing dialogue text.
        """
        logger.info("Step 2: Generating baseline dialogues")

        dialogues = {}
        for c_level, c_level_desc in [("高", "高い"), ("低", "低い")]:
            if self.dry_run:
                logger.info("  [DRY-RUN] Generating mock dialogue (C=%s)", c_level)
                dialogue = _generate_mock_dialogue(c_level, self.seed)
            else:
                logger.info("  Generating dialogue (C=%s) via Bedrock", c_level)
                prompt = PROMPT_DIALOGUE_GENERATION.format(
                    c_level=c_level,
                    c_level_desc=c_level_desc,
                )
                client = self._get_client()
                dialogue = call_bedrock(
                    client,
                    self.model_id,
                    prompt,
                    max_tokens=4096,
                    temperature=0.7,
                    seed=self.seed,
                    region=self.region,
                )

            dialogues[c_level] = dialogue
            logger.info(
                "  Generated dialogue (C=%s): %d chars, %d lines",
                c_level,
                len(dialogue),
                dialogue.count("\n") + 1,
            )

        self.baseline_dialogue_high = dialogues["高"]
        self.baseline_dialogue_low = dialogues["低"]
        return {"high": dialogues["高"], "low": dialogues["低"]}

    # ── Step 3: Feature manipulation ─────────────────────────────────

    def manipulate_feature(self) -> dict[str, dict[str, str]]:
        """Manipulate target feature (increase/decrease) without personality info.

        For each baseline dialogue (high-C and low-C), generate two variants:
        - increase: target feature value increased
        - decrease: target feature value decreased

        The LLM is instructed to modify ONLY the target feature,
        without any personality-related information.

        Returns:
            dict with keys "high" and "low", each containing
            {"increase": str, "decrease": str}.
        """
        logger.info("Step 3: Manipulating feature '%s'", self.target_feature)

        if self.baseline_dialogue_high is None or self.baseline_dialogue_low is None:
            raise RuntimeError(
                "Baseline dialogues not generated. "
                "Call generate_baseline_dialogue() first."
            )

        results = {}
        for label, dialogue in [
            ("high", self.baseline_dialogue_high),
            ("low", self.baseline_dialogue_low),
        ]:
            manipulated = {}
            for direction, prompt_tmpl in [
                ("increase", PROMPT_FEATURE_MANIPULATION_INCREASE),
                ("decrease", PROMPT_FEATURE_MANIPULATION_DECREASE),
            ]:
                if self.dry_run:
                    logger.info(
                        "  [DRY-RUN] Mock manipulation (%s, %s)",
                        label,
                        direction,
                    )
                    text = _generate_mock_manipulated(
                        dialogue, direction, self.seed
                    )
                else:
                    logger.info(
                        "  Manipulating (%s, %s) via Bedrock",
                        label,
                        direction,
                    )
                    prompt = prompt_tmpl.format(dialogue=dialogue)
                    client = self._get_client()
                    text = call_bedrock(
                        client,
                        self.model_id,
                        prompt,
                        max_tokens=4096,
                        temperature=0.3,
                        seed=self.seed,
                        region=self.region,
                    )

                manipulated[direction] = text
                logger.info(
                    "  Manipulated (%s, %s): %d chars",
                    label,
                    direction,
                    len(text),
                )

            results[label] = manipulated

        self.manipulated_increase_high = results["high"]["increase"]
        self.manipulated_decrease_high = results["high"]["decrease"]
        self.manipulated_increase_low = results["low"]["increase"]
        self.manipulated_decrease_low = results["low"]["decrease"]

        return results

    # ── Step 4: Personality estimation ───────────────────────────────

    def _estimate_single_score(
        self, dialogue: str, label: str
    ) -> float:
        """Estimate Big5 C score for a single dialogue.

        Uses IPIP-NEO-120 subset items and the virtual teacher protocol.

        Args:
            dialogue: Conversation text to evaluate.
            label: Label for logging (e.g. "high_baseline").

        Returns:
            Mean C score (0-4 scale).
        """
        scores = []
        for item in C_ITEMS_SUBSET:
            if self.dry_run:
                # Mock: generate plausible score
                rng = np.random.default_rng(
                    self.seed + item["item_id"] + hash(label) % 10000
                )
                raw_score = float(rng.choice([0, 1, 2, 3, 4]))
            else:
                prompt = PROMPT_PERSONALITY_ESTIMATION.format(
                    statement=item["text"],
                    transcript=dialogue,
                )
                client = self._get_client()
                response = call_bedrock(
                    client,
                    self.model_id,
                    prompt,
                    max_tokens=128,
                    temperature=0.0,
                    seed=self.seed,
                    region=self.region,
                )
                choice = _normalize_choice(response)
                if choice is None:
                    logger.warning(
                        "  Could not parse response for item %d: '%s'",
                        item["item_id"],
                        response[:100],
                    )
                    raw_score = 2.0  # neutral fallback
                else:
                    raw_score = float(S2I[choice])

            # Apply reverse scoring
            if item["reverse"] == 1:
                raw_score = 4.0 - raw_score

            scores.append(raw_score)

        mean_score = float(np.mean(scores))
        logger.info(
            "  Estimated C score for %s: %.3f (from %d items)",
            label,
            mean_score,
            len(scores),
        )
        return mean_score

    def estimate_personality(self) -> dict[str, float]:
        """Estimate Big5 personality for all dialogue variants.

        Uses the virtual teacher protocol (IPIP-NEO-120) to estimate
        Conscientiousness scores for:
        - high_baseline, high_increase, high_decrease
        - low_baseline, low_increase, low_decrease

        Returns:
            dict mapping label to C score.
        """
        logger.info("Step 4: Estimating personality scores")

        if self.baseline_dialogue_high is None:
            raise RuntimeError(
                "Dialogues not generated. Run previous steps first."
            )

        dialogue_map = {
            "high_baseline": self.baseline_dialogue_high,
            "high_increase": self.manipulated_increase_high,
            "high_decrease": self.manipulated_decrease_high,
            "low_baseline": self.baseline_dialogue_low,
            "low_increase": self.manipulated_increase_low,
            "low_decrease": self.manipulated_decrease_low,
        }

        scores = {}
        for label, dialogue in dialogue_map.items():
            if dialogue is None:
                logger.warning("  Skipping %s (no dialogue)", label)
                continue
            scores[label] = self._estimate_single_score(dialogue, label)

        self.scores = scores
        return scores


    # ── Step 5: Compute delta ────────────────────────────────────────

    def compute_delta(self) -> list[CircularInteractionResult]:
        """Compute ΔC between baseline and intervention for each condition.

        For each condition (high-C, low-C):
        - delta_increase = intervention_increase_score - baseline_score
        - delta_decrease = intervention_decrease_score - baseline_score
        - direction_consistent: checks if manipulation direction aligns
          with regression coefficient sign

        For a positive regression coefficient (e.g. FILL_rate_per_100chars → C):
        - Increasing the feature should increase C → delta_increase > 0
        - Decreasing the feature should decrease C → delta_decrease < 0

        Returns:
            List of CircularInteractionResult for each condition.
        """
        logger.info("Step 5: Computing ΔC")

        if not self.scores:
            raise RuntimeError(
                "Scores not estimated. Call estimate_personality() first."
            )

        results = []
        for condition in ["high", "low"]:
            baseline_key = f"{condition}_baseline"
            increase_key = f"{condition}_increase"
            decrease_key = f"{condition}_decrease"

            if baseline_key not in self.scores:
                logger.warning("  Missing scores for %s condition", condition)
                continue

            baseline = self.scores[baseline_key]
            increase = self.scores.get(increase_key, float("nan"))
            decrease = self.scores.get(decrease_key, float("nan"))

            delta_inc = round(increase - baseline, 4)
            delta_dec = round(decrease - baseline, 4)

            # Check direction consistency
            if self.regression_coef_sign == "+":
                # Positive coef: increase feature → increase score
                consistent = (delta_inc > 0) and (delta_dec < 0)
            else:
                # Negative coef: increase feature → decrease score
                consistent = (delta_inc < 0) and (delta_dec > 0)

            result = CircularInteractionResult(
                target_feature=self.target_feature,
                target_trait=self.target_trait,
                baseline_score=baseline,
                intervention_increase_score=increase,
                intervention_decrease_score=decrease,
                delta_increase=delta_inc,
                delta_decrease=delta_dec,
                direction_consistent=consistent,
                regression_coef_sign=self.regression_coef_sign,
                notes=(
                    f"Condition: C={condition}. "
                    f"Feature: {self.feature_info['description']}. "
                    f"Model: {self.model_id}. "
                    f"Seed: {self.seed}. "
                    f"{'[DRY-RUN] Mock data used.' if self.dry_run else ''}"
                ),
            )
            results.append(result)

            logger.info(
                "  %s condition: baseline=%.3f, Δinc=%+.4f, Δdec=%+.4f, "
                "consistent=%s",
                condition,
                baseline,
                delta_inc,
                delta_dec,
                consistent,
            )

        return results

    # ── Step 6: Report results ───────────────────────────────────────

    def report_results(
        self, results: list[CircularInteractionResult]
    ) -> Path:
        """Generate consistency report and save all artifacts.

        Saves:
        - report.json: Full structured results
        - report.txt: Human-readable summary
        - dialogues/: All generated dialogues

        Args:
            results: List of CircularInteractionResult from compute_delta().

        Returns:
            Path to the output directory.
        """
        logger.info("Step 6: Generating report")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Save dialogues ───────────────────────────────────────────
        dialogue_dir = self.out_dir / "dialogues"
        dialogue_dir.mkdir(parents=True, exist_ok=True)

        dialogue_map = {
            "high_baseline.txt": self.baseline_dialogue_high,
            "high_increase.txt": self.manipulated_increase_high,
            "high_decrease.txt": self.manipulated_decrease_high,
            "low_baseline.txt": self.baseline_dialogue_low,
            "low_increase.txt": self.manipulated_increase_low,
            "low_decrease.txt": self.manipulated_decrease_low,
        }
        for fname, text in dialogue_map.items():
            if text is not None:
                (dialogue_dir / fname).write_text(text, encoding="utf-8")

        # ── Save structured results (JSON) ───────────────────────────
        report_data = {
            "pipeline_config": {
                "model_id": self.model_id,
                "target_feature": self.target_feature,
                "target_trait": self.target_trait,
                "regression_coef_sign": self.regression_coef_sign,
                "seed": self.seed,
                "dry_run": self.dry_run,
            },
            "feature_info": self.feature_info,
            "scores": self.scores,
            "results": [asdict(r) for r in results],
        }

        report_json_path = self.out_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info("  Wrote %s", report_json_path)

        # ── Generate human-readable report ───────────────────────────
        lines = [
            "=" * 70,
            "LLM Circular Interaction Verification Report",
            "=" * 70,
            "",
            f"Model:            {self.model_id}",
            f"Target feature:   {self.target_feature} ({self.feature_info['description']})",
            f"Target trait:     {self.target_trait}",
            f"Regression coef:  {self.regression_coef_sign}",
            f"Seed:             {self.seed}",
            f"Dry run:          {self.dry_run}",
            "",
            "-" * 70,
            "Scores",
            "-" * 70,
        ]
        for label, score in sorted(self.scores.items()):
            lines.append(f"  {label:25s}: {score:.3f}")

        lines.extend([
            "",
            "-" * 70,
            "Results",
            "-" * 70,
        ])
        for r in results:
            lines.extend([
                f"",
                f"  Condition: C={r.notes.split('Condition: C=')[1].split('.')[0]}",
                f"    Baseline score:     {r.baseline_score:.3f}",
                f"    After increase:     {r.intervention_increase_score:.3f} "
                f"(Δ = {r.delta_increase:+.4f})",
                f"    After decrease:     {r.intervention_decrease_score:.3f} "
                f"(Δ = {r.delta_decrease:+.4f})",
                f"    Direction consistent: {r.direction_consistent}",
            ])

        # Overall consistency
        n_consistent = sum(1 for r in results if r.direction_consistent)
        n_total = len(results)
        lines.extend([
            "",
            "-" * 70,
            "Summary",
            "-" * 70,
            f"  Consistent conditions: {n_consistent}/{n_total}",
            f"  Overall consistency:   "
            f"{'YES' if n_consistent == n_total else 'PARTIAL' if n_consistent > 0 else 'NO'}",
            "",
        ])

        # Limitations
        lines.extend([
            "-" * 70,
            "Limitations",
            "-" * 70,
            "  - 対話生成はLLMによるものであり、実際の自然会話とは異なる",
            "  - 特徴量操作が対話の自然さを損なう可能性がある",
            "  - IPIP-NEO-120のサブセット（10項目）による簡易推定",
            "  - 単一モデル・単一シードでの検証（頑健性は限定的）",
            "  - フィラー操作は文脈依存であり、機械的な挿入/削除は不自然になりうる",
            "",
            "=" * 70,
        ])

        report_txt = "\n".join(lines)
        report_txt_path = self.out_dir / "report.txt"
        report_txt_path.write_text(report_txt, encoding="utf-8")
        logger.info("  Wrote %s", report_txt_path)

        # Print to console
        print(report_txt)

        return self.out_dir

    # ── Full pipeline execution ──────────────────────────────────────

    def run(self) -> list[CircularInteractionResult]:
        """Execute the full circular interaction verification pipeline.

        Returns:
            List of CircularInteractionResult.
        """
        logger.info("Starting LLM Circular Interaction Pipeline")
        logger.info("  Model: %s", self.model_id)
        logger.info("  Feature: %s", self.target_feature)
        logger.info("  Seed: %d", self.seed)
        logger.info("  Dry run: %s", self.dry_run)
        logger.info("")

        # Step 1
        self.select_target_feature()

        # Step 2
        self.generate_baseline_dialogue()

        # Step 3
        self.manipulate_feature()

        # Step 4
        self.estimate_personality()

        # Step 5
        results = self.compute_delta()

        # Step 6
        self.report_results(results)

        logger.info("Pipeline complete. Results saved to %s", self.out_dir)
        return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    """CLI entry point for the circular interaction pipeline."""
    ap = argparse.ArgumentParser(
        description=(
            "LLM circular interaction verification pipeline. "
            "Validates the 'LLM → classical features → LLM' circular framework "
            "by manipulating a target feature and measuring the effect on "
            "personality estimation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry run (no API calls):\n"
            "  python scripts/analysis/llm_circular_interaction.py \\\n"
            "      --model_id anthropic.claude-sonnet-4-20250514-v1:0 \\\n"
            "      --target_feature FILL_rate_per_100chars \\\n"
            "      --seed 42 \\\n"
            "      --out_dir artifacts/analysis/results/circular_interaction \\\n"
            "      --dry-run\n"
            "\n"
            "  # Real run (requires AWS Bedrock access):\n"
            "  python scripts/analysis/llm_circular_interaction.py \\\n"
            "      --model_id anthropic.claude-sonnet-4-20250514-v1:0 \\\n"
            "      --target_feature FILL_rate_per_100chars \\\n"
            "      --seed 42 \\\n"
            "      --out_dir artifacts/analysis/results/circular_interaction\n"
            "\n"
            "Note: Bare model IDs (e.g. anthropic.claude-sonnet-4-20250514-v1:0)\n"
            "are automatically resolved to inference profile IDs based on --region.\n"
            "You can also pass an explicit inference profile ID\n"
            "(e.g. apac.anthropic.claude-sonnet-4-20250514-v1:0).\n"
        ),
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help=(
            "Bedrock model ID or inference profile ID. "
            "Bare model IDs are auto-resolved to inference profiles "
            "(e.g. anthropic.claude-sonnet-4-20250514-v1:0 → "
            "apac.anthropic.claude-sonnet-4-20250514-v1:0 for ap-northeast-1)"
        ),
    )
    ap.add_argument(
        "--target_feature",
        default=DEFAULT_TARGET_FEATURE,
        choices=list(FEATURE_CANDIDATES.keys()),
        help=f"Target feature to manipulate (default: {DEFAULT_TARGET_FEATURE})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    ap.add_argument(
        "--out_dir",
        default="artifacts/analysis/results/circular_interaction",
        help="Output directory (default: artifacts/analysis/results/circular_interaction)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Use mock data instead of Bedrock API calls (for testing)",
    )
    ap.add_argument(
        "--region",
        default=REGION,
        help=f"AWS region for Bedrock (default: {REGION})",
    )
    args = ap.parse_args()

    pipeline = CircularInteractionPipeline(
        model_id=args.model_id,
        target_feature=args.target_feature,
        seed=args.seed,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        region=args.region,
    )

    results = pipeline.run()

    # Exit code: 0 if at least one condition is consistent
    n_consistent = sum(1 for r in results if r.direction_consistent)
    if n_consistent == 0:
        logger.warning("No conditions showed direction consistency")


if __name__ == "__main__":
    main()
