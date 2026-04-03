"""
test_phase1.py
==============
Checkpoint test for Phase 1 files.
Run from recallify/ root:

    python test_phase1.py

Tests (in order):
  1. terminology_lexicon.py — term extraction for all 3 domains
  2. domain_models.py       — enums, dataclasses, weight validation
  3. dhs_metric.py          — DHS scoring (NLI disabled for speed)
  4. dataset_loader.py      — loads 5 samples per domain from cache or HF

All tests must PASS before moving to Phase 2.
"""

import sys
import logging
logging.basicConfig(level=logging.WARNING)   # quiet during tests

PASS = "✓"
FAIL = "✗"
results = []

def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        print(f"       Error: {e}")

print("\n══════════════════════════════════════════════")
print("  TriDomRAG — Phase 1 Checkpoint Tests")
print("══════════════════════════════════════════════\n")

# ─────────────────────────────────────────────────────────────────────────────
print("── 1. terminology_lexicon.py ──")
# ─────────────────────────────────────────────────────────────────────────────

def test_lexicon_import():
    from src.terminology_lexicon import TerminologyLexicon
    lex = TerminologyLexicon()
    assert lex is not None   # just check import

def test_lexicon_sizes():
    from src.terminology_lexicon import TerminologyLexicon
    from core.domain_models import DomainType
    lex = TerminologyLexicon()
    sizes = lex.lexicon_size()
    for d in ["healthcare", "legal", "finance"]:
        assert sizes[d] >= 30, f"{d} lexicon too small: {sizes[d]}"

def test_lexicon_extraction_healthcare():
    from src.terminology_lexicon import TerminologyLexicon
    from core.domain_models import DomainType
    lex = TerminologyLexicon()
    terms = lex.extract_terms("Patient with hypertension and ckd", DomainType.HEALTHCARE)
    assert "hypertension" in terms, f"Expected 'hypertension', got {terms}"

def test_lexicon_extraction_finance():
    from src.terminology_lexicon import TerminologyLexicon
    from core.domain_models import DomainType
    lex = TerminologyLexicon()
    terms = lex.extract_terms("EBITDA and free cash flow were strong", DomainType.FINANCE)
    assert "ebitda" in terms or "free cash flow" in terms

def test_lexicon_extraction_legal():
    from src.terminology_lexicon import TerminologyLexicon
    from core.domain_models import DomainType
    lex = TerminologyLexicon()
    terms = lex.extract_terms("Breach of contract with liquidated damages", DomainType.LEGAL)
    assert "breach of contract" in terms

test("import TerminologyLexicon",          test_lexicon_import)
test("lexicon sizes ≥ 30 per domain",      test_lexicon_sizes)
test("healthcare term extraction",         test_lexicon_extraction_healthcare)
test("finance term extraction",            test_lexicon_extraction_finance)
test("legal term extraction",              test_lexicon_extraction_legal)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. domain_models.py ──")
# ─────────────────────────────────────────────────────────────────────────────

def test_enum_values():
    from core.domain_models import DomainType, AnswerType, Difficulty
    assert DomainType.FINANCE.value == "finance"
    assert AnswerType.NUMERICAL.value == "numerical"
    assert Difficulty.EXPERT.value == "expert"

def test_weights_sum_to_one():
    from core.domain_models import DOMAIN_WEIGHTS, DomainType
    for d in DomainType:
        w = DOMAIN_WEIGHTS[d]
        total = w.alpha + w.beta + w.gamma
        assert abs(total - 1.0) < 1e-6, f"{d} weights sum={total}"

def test_benchmark_sample_from_dict():
    from core.domain_models import BenchmarkSample, AnswerType, DomainType
    s = BenchmarkSample.from_dict({
        "id": "fn_0001", "domain": "finance",
        "question": "Revenue?", "answer": "$1B", "context": "...",
        "answer_type": "numerical", "difficulty": "basic", "source_doc": "10K",
    })
    assert s.answer_type == AnswerType.NUMERICAL
    assert s.domain == DomainType.FINANCE

def test_dhs_result_to_dict():
    from core.domain_models import (DHSResult, DHSComponents, DomainType,
                                     AnswerType, DOMAIN_WEIGHTS)
    r = DHSResult(
        sample_id="t001", domain=DomainType.HEALTHCARE,
        question="Q?", generated_answer="A", gold_answer="A",
        answer_type=AnswerType.TEXTUAL,
        components=DHSComponents(0.8, 0.7, 0.9),
        weights=DOMAIN_WEIGHTS[DomainType.HEALTHCARE],
        dhs_score=0.8,
    )
    d = r.to_dict()
    assert "dhs_score" in d and d["passed"] is True

def test_experiment_result_avg():
    from core.domain_models import (ExperimentResult, ExperimentConfig,
                                     DHSResult, DHSComponents, DomainType,
                                     AnswerType, DOMAIN_WEIGHTS,
                                     RetrievalStrategy, Granularity)
    config = ExperimentConfig(RetrievalStrategy.HYBRID, Granularity.DOCUMENT)
    er = ExperimentResult(config=config, domain=DomainType.FINANCE)
    w  = DOMAIN_WEIGHTS[DomainType.FINANCE]
    for i in range(5):
        er.dhs_results.append(DHSResult(
            sample_id=f"fn_{i}", domain=DomainType.FINANCE,
            question="Q", generated_answer="A", gold_answer="A",
            answer_type=AnswerType.NUMERICAL,
            components=DHSComponents(0.8, 0.7, 0.9),
            weights=w, dhs_score=0.8,
        ))
    assert abs(er.avg_dhs - 0.8) < 1e-6

test("enum values correct",                test_enum_values)
test("domain weights sum to 1.0",          test_weights_sum_to_one)
test("BenchmarkSample.from_dict()",        test_benchmark_sample_from_dict)
test("DHSResult.to_dict()",                test_dhs_result_to_dict)
test("ExperimentResult averages",          test_experiment_result_avg)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. dhs_metric.py ──")
# ─────────────────────────────────────────────────────────────────────────────

def test_dhs_import():
    from src.dhs_metric import DHSMetric
    m = DHSMetric(use_nli=False)
    assert m is not None

def test_dhs_numerical_perfect():
    from src.dhs_metric import DHSMetric
    from core.domain_models import DomainType, AnswerType
    m = DHSMetric(use_nli=False)
    r = m.score("t1", DomainType.FINANCE, "Revenue?",
                 "$19.88 billion", "$19.88 billion",
                 ["Apple reported $19.88 billion net income."],
                 AnswerType.NUMERICAL)
    assert r.dhs_score > 0.7, f"Perfect match scored {r.dhs_score}"

def test_dhs_numerical_wrong():
    from src.dhs_metric import DHSMetric
    from core.domain_models import DomainType, AnswerType
    m = DHSMetric(use_nli=False)
    r = m.score("t2", DomainType.FINANCE, "Revenue?",
                 "$999 trillion", "$19.88 billion",
                 ["Unrelated context about marketing."],
                 AnswerType.NUMERICAL)
    assert r.dhs_score < 0.6, f"Wrong answer should score low, got {r.dhs_score}"

def test_dhs_failure_mode_tagged():
    from src.dhs_metric import DHSMetric
    from core.domain_models import DomainType, AnswerType
    m = DHSMetric(use_nli=False)
    r = m.score("t3", DomainType.FINANCE, "Revenue?",
                 "$999 trillion", "$19.88 billion",
                 ["Unrelated context."], AnswerType.NUMERICAL)
    assert r.failure_mode is not None, "Failure mode should be tagged for low DHS"

def test_dhs_textual_token_fallback():
    from src.dhs_metric import DHSMetric
    from core.domain_models import DomainType, AnswerType
    m = DHSMetric(use_nli=False)  # uses token overlap
    r = m.score("t4", DomainType.HEALTHCARE, "Treatment?",
                 "Lifestyle modification and medication",
                 "Antihypertensive medication and lifestyle",
                 ["Lifestyle modification and antihypertensive medication."],
                 AnswerType.TEXTUAL)
    assert 0.0 <= r.dhs_score <= 1.0

def test_extract_number():
    from src.dhs_metric import DHSMetric
    m = DHSMetric(use_nli=False)
    assert abs(m._extract_number("$19.88 billion") - 19.88e9) < 1e3
    assert abs(m._extract_number("83.5%") - 83.5) < 0.01
    assert m._extract_number("No numbers here") is None

test("import DHSMetric",                        test_dhs_import)
test("numerical DHS — perfect match > 0.7",     test_dhs_numerical_perfect)
test("numerical DHS — wrong answer < 0.6",      test_dhs_numerical_wrong)
test("failure mode tagged when DHS < 0.5",      test_dhs_failure_mode_tagged)
test("textual DHS — token fallback [0,1]",      test_dhs_textual_token_fallback)
test("_extract_number() parses billions/%",     test_extract_number)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. dataset_loader.py ──")
# ─────────────────────────────────────────────────────────────────────────────

def test_loader_import():
    from src.dataset_loader import DatasetLoader
    assert DatasetLoader is not None

def test_loader_infer_type():
    from src.dataset_loader import DatasetLoader, TYPE_N, TYPE_T, TYPE_E
    loader = DatasetLoader()
    assert loader._infer_type("$19.88 billion") == TYPE_N
    assert loader._infer_type("Yes, the clause restricts sublicensing according to...") == TYPE_T

def test_loader_infer_difficulty():
    from src.dataset_loader import DatasetLoader
    loader = DatasetLoader()
    assert loader._infer_difficulty("What is hypertension?") == "basic"
    assert loader._infer_difficulty("Analyse and evaluate the complex relationship between...") == "expert"

def test_loader_cache_roundtrip(tmp_path=None):
    from src.dataset_loader import DatasetLoader
    import tempfile, pathlib, json
    with tempfile.TemporaryDirectory() as td:
        loader = DatasetLoader(data_dir=pathlib.Path(td))
        sample_data = [{"id": "x1", "domain": "finance", "question": "Q?",
                        "answer": "A", "context": "C", "answer_type": "textual",
                        "difficulty": "basic", "source_doc": "doc1"}]
        cache = pathlib.Path(td) / "finance.json"
        loader._write_cache(cache, sample_data)
        loaded = loader._read_cache(cache)
        assert loaded[0]["id"] == "x1"

test("import DatasetLoader",               test_loader_import)
test("_infer_type() numerical/textual",    test_loader_infer_type)
test("_infer_difficulty() basic/expert",   test_loader_infer_difficulty)
test("cache write + read roundtrip",       test_loader_cache_roundtrip)

# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)
print(f"  Results: {passed}/{total} passed   {failed} failed")
print("══════════════════════════════════════════════\n")
if failed == 0:
    print("  🟢 Phase 1 COMPLETE — safe to start Phase 2\n")
else:
    print("  🔴 Fix failing tests before Phase 2\n")
    sys.exit(1)