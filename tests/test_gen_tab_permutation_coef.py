#!/usr/bin/env python3
"""Unit tests for _read_permutation_coef_tsvs and gen_tab_permutation_coef."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.paper_figs.gen_paper_figs_v2 import (
    _read_permutation_coef_tsvs,
    gen_tab_permutation_coef,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_tsv(tmp_path: Path) -> Path:
    """Create a sample permutation_coef TSV file."""
    content = textwrap.dedent("""\
        feature\tcoef_obs\tp_value\tsignificant
        PG_speech_ratio\t0.1234\t0.0020\tTrue
        PG_pause_mean\t-0.0567\t0.3200\tFalse
        FILL_has_any\t0.0890\t0.0480\tTrue
    """)
    tsv_path = tmp_path / "permutation_coef_C_ensemble.tsv"
    tsv_path.write_text(content, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# _read_permutation_coef_tsvs
# ---------------------------------------------------------------------------
class TestReadPermutationCoefTsvs:
    def test_reads_single_file(self, sample_tsv: Path) -> None:
        df = _read_permutation_coef_tsvs(sample_tsv)
        assert len(df) == 3
        assert "feature" in df.columns
        assert "coef_obs" in df.columns
        assert "p_value" in df.columns
        assert "significant" in df.columns
        assert "_source" in df.columns

    def test_reads_multiple_files(self, tmp_path: Path) -> None:
        for suffix in ["C_ensemble", "A_sonnet"]:
            content = (
                "feature\tcoef_obs\tp_value\tsignificant\n"
                f"PG_speech_ratio\t0.1\t0.01\tTrue\n"
            )
            (tmp_path / f"permutation_coef_{suffix}.tsv").write_text(
                content, encoding="utf-8"
            )
        df = _read_permutation_coef_tsvs(tmp_path)
        assert len(df) == 2
        assert df["_source"].nunique() == 2

    def test_raises_when_no_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No permutation_coef_"):
            _read_permutation_coef_tsvs(tmp_path)


# ---------------------------------------------------------------------------
# gen_tab_permutation_coef
# ---------------------------------------------------------------------------
class TestGenTabPermutationCoef:
    def test_generates_latex_file(self, sample_tsv: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(sample_tsv, out_dir)
        tex_path = out_dir / "tab_permutation_coef.tex"
        assert tex_path.exists()
        content = tex_path.read_text(encoding="utf-8")
        # Check booktabs structure
        assert "\\begin{tabular}" in content
        assert "\\toprule" in content
        assert "\\midrule" in content
        assert "\\bottomrule" in content

    def test_significance_marked_by_asterisk_not_bold(
        self, sample_tsv: Path, tmp_path: Path
    ) -> None:
        """Significance is shown with an asterisk on the p-value.

        Bold is reserved for structural emphasis (headings), so significance in
        every table is marked the same way: an asterisk superscript. Emphasising
        the feature name instead would let typography carry a claim that the
        p-value column already states.
        """
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(sample_tsv, out_dir)
        content = (out_dir / "tab_permutation_coef.tex").read_text(encoding="utf-8")

        # significant rows: asterisk on the p-value
        assert "0.0020$^{*}$" in content
        assert "0.0480$^{*}$" in content
        # non-significant row: bare p-value
        assert "0.3200 \\\\" in content
        # no bold anywhere in the table body
        assert "\\textbf" not in content

    def test_no_separate_significance_column(
        self, sample_tsv: Path, tmp_path: Path
    ) -> None:
        """The asterisk replaces a dedicated Sig. / checkmark column."""
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(sample_tsv, out_dir)
        content = (out_dir / "tab_permutation_coef.tex").read_text(encoding="utf-8")
        assert "\\checkmark" not in content
        assert "Sig." not in content

    def test_header_columns(self, sample_tsv: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(sample_tsv, out_dir)
        content = (out_dir / "tab_permutation_coef.tex").read_text(encoding="utf-8")
        assert "Feature" in content
        # subscripts are upright, not italic maths variables
        assert "$\\beta_{\\mathrm{obs}}$" in content
        assert "$p$-value" in content

    def test_graceful_skip_when_no_files(self, tmp_path: Path, capsys) -> None:
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(tmp_path, out_dir)
        # Should not create the file
        assert not (out_dir / "tab_permutation_coef.tex").exists()

    def test_prefers_ensemble_source(self, tmp_path: Path) -> None:
        # Create two files: one ensemble, one not
        for suffix, coef in [("C_sonnet", "0.1"), ("C_ensemble", "0.2")]:
            content = (
                "feature\tcoef_obs\tp_value\tsignificant\n"
                f"PG_speech_ratio\t{coef}\t0.01\tTrue\n"
            )
            (tmp_path / f"permutation_coef_{suffix}.tsv").write_text(
                content, encoding="utf-8"
            )
        out_dir = tmp_path / "output"
        gen_tab_permutation_coef(tmp_path, out_dir)
        content = (out_dir / "tab_permutation_coef.tex").read_text(encoding="utf-8")
        # Should use ensemble (coef=0.2)
        assert "0.2000" in content
