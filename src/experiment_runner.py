"""
experiment_runner.py
====================
Runs all 6 RAG configurations across all 3 domains and saves results.
This generates the numbers that go directly into the paper tables.

Usage:
  # Run all experiments (full paper run)
  python src/experiment_runner.py --mode full

  # Quick dev run (10 samples per domain)
  python src/experiment_runner.py --mode dev

  # Single config test
  python src/experiment_runner.py --mode dev --config bm25_chunk

Results saved to:
  results/experiment_results.json   ← all raw DHSResult objects
  results/summary_table.json        ← paper Table 2 numbers
  results/failure_analysis.json     ← paper Table 6 numbers
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domain_models import (
    DomainType, BenchmarkSample, ExperimentResult,
    RetrievalStrategy, Granularity,
)
from src.dataset_loader    import DatasetLoader
from src.retrieval_configs import (
    RetrievalConfig, get_all_configs,
    build_retriever, prepare_corpus,
)
from src.dhs_metric        import DHSMetric
from dotenv import load_dotenv
load_dotenv()
from src.llm_config        import get_llm

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Orchestrates the full benchmark evaluation.

    Flow for each (config × domain):
      1. Load benchmark samples for the domain
      2. Build retrieval corpus (chunk or document level)
      3. Index corpus with the retriever (BM25/Dense/Hybrid)
      4. For each sample: retrieve top-k → generate answer → score with DHS
      5. Save results
    """

    def __init__(
        self,
        configs      : Optional[List[RetrievalConfig]] = None,
        sample_size  : int  = 400,        # 400 per domain = full paper run
        use_nli      : bool = False,      # True only at college lab with GPU
        llm_provider : str  = "local",
    ):
        self.configs      = configs or get_all_configs()
        self.sample_size  = sample_size
        self.metric       = DHSMetric(use_nli=use_nli)
        self.llm          = get_llm(provider=llm_provider)
        self.loader       = DatasetLoader(sample_size=sample_size)
        RESULTS_DIR.mkdir(exist_ok=True)

        logger.info(f"ExperimentRunner ready")
        logger.info(f"  Configs      : {[c.name for c in self.configs]}")
        logger.info(f"  Sample size  : {sample_size} per domain")
        logger.info(f"  LLM          : {type(self.llm).__name__}")
        logger.info(f"  NLI scoring  : {use_nli}")

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self) -> Dict:
        """Run all configs × all domains. Returns full results dict."""
        logger.info("=" * 60)
        logger.info("Starting full experiment run")
        logger.info("=" * 60)

        all_results  = {}
        summary_rows = []
        start_time   = datetime.now()

        # Load all datasets once
        logger.info("Loading datasets...")
        all_data = self.loader.load_all()

        for config in self.configs:
            all_results[config.name] = {}
            for domain in DomainType:
                logger.info(f"\n── {config.name} × {domain.value} ──")

                raw_samples = all_data[domain.value]
                samples     = [BenchmarkSample.from_dict(s) for s in raw_samples]

                exp_result  = self._run_single(config, domain, samples, raw_samples)
                all_results[config.name][domain.value] = exp_result.summary()

                row = {
                    "config"    : config.name,
                    "domain"    : domain.value,
                    **exp_result.summary(),
                }
                summary_rows.append(row)
                logger.info(f"  avg_dhs={exp_result.avg_dhs:.3f}  "
                             f"failures={exp_result.failure_count}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ All experiments done in {elapsed:.0f}s")

        # Save results
        self._save_results(all_results, summary_rows)
        return all_results

    def run_single_config(self, config_name: str) -> Dict:
        """Run one named config across all domains. Useful for debugging."""
        config = next((c for c in self.configs if c.name == config_name), None)
        if config is None:
            raise ValueError(f"Config '{config_name}' not found. "
                             f"Available: {[c.name for c in self.configs]}")

        all_data    = self.loader.load_all()
        all_results = {}

        for domain in DomainType:
            raw_samples = all_data[domain.value]
            samples     = [BenchmarkSample.from_dict(s) for s in raw_samples]
            exp_result  = self._run_single(config, domain, samples, raw_samples)
            all_results[domain.value] = exp_result.summary()

        return all_results

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNAL
    # ─────────────────────────────────────────────────────────────────────────

    def _run_single(
        self,
        config      : RetrievalConfig,
        domain      : DomainType,
        samples     : List[BenchmarkSample],
        raw_samples : List[Dict],
    ) -> ExperimentResult:
        """Run one (config, domain) pair and return ExperimentResult."""

        exp_result = ExperimentResult(config=config, domain=domain)  # type: ignore

        # 1. Build corpus
        corpus_texts, corpus_meta = prepare_corpus(
            raw_samples,
            granularity  = config.granularity,
            chunk_size   = config.chunk_size,
            overlap      = config.chunk_overlap,
        )

        # 2. Index
        retriever = build_retriever(config)
        retriever.index(corpus_texts, corpus_meta)

        # 3. Run per sample
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"    [{domain.value}] {i}/{len(samples)} samples...")

            try:
                dhs_result = self._evaluate_sample(
                    sample, retriever, config.top_k, domain
                )
                exp_result.dhs_results.append(dhs_result)
            except Exception as e:
                logger.error(f"    Sample {sample.id} failed: {e}")

        return exp_result

    def _evaluate_sample(self, sample, retriever, top_k, domain):
        """Retrieve → generate → score one sample."""

        # Retrieve
        retrieved = retriever.retrieve(sample.question, top_k=top_k)
        context_chunks = [text for text, score, meta in retrieved]
        context_text   = "\n\n".join(context_chunks)

        # Check source availability
        source_available = any(
            meta.get("source_doc", "") for _, _, meta in retrieved
        )

        # Generate answer
        generated_answer = self.llm.generate(
            question = sample.question,
            context  = context_text,
            domain   = domain.value,
        )

        # Score with DHS
        dhs_result = self.metric.score(
            sample_id        = sample.id,
            domain           = domain,
            question         = sample.question,
            generated_answer = generated_answer,
            gold_answer      = sample.answer,
            context_chunks   = context_chunks,
            answer_type      = sample.answer_type,
            source_available = source_available,
        )

        return dhs_result

    def _save_results(self, all_results: Dict, summary_rows: List[Dict]) -> None:
        """Save all results to results/ directory."""

        # Full results
        results_path = RESULTS_DIR / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved → {results_path}")

        # Summary table (paper Table 2)
        summary_path = RESULTS_DIR / "summary_table.json"
        with open(summary_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        logger.info(f"Summary saved → {summary_path}")

        # Print paper-ready table
        self._print_table(summary_rows)

    def _print_table(self, rows: List[Dict]) -> None:
        """Print results in a format matching paper Table 2."""
        print("\n" + "=" * 80)
        print("PAPER TABLE 2 — DHS Results")
        print("=" * 80)
        print(f"{'Config':<25} {'Domain':<15} {'DHS':>8} {'F':>8} {'A':>8} {'T':>8} {'Fails':>6}")
        print("-" * 80)
        for r in rows:
            print(
                f"{r['config']:<25} "
                f"{r['domain']:<15} "
                f"{r.get('avg_dhs',0):.3f}    "
                f"{r.get('avg_faithfulness',0):.3f}    "
                f"{r.get('avg_attribution',0):.3f}    "
                f"{r.get('avg_terminology',0):.3f}    "
                f"{r.get('failures',0):>5}"
            )
        print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="TriDomRAG Experiment Runner")
    parser.add_argument("--mode",   default="dev",
                        choices=["dev", "full"],
                        help="dev=10 samples, full=400 samples")
    parser.add_argument("--config", default=None,
                        help="Run single config e.g. bm25_chunk")
    parser.add_argument("--llm",    default="local",
                        choices=["local", "openai", "groq"],
                        help="LLM backend to use")
    parser.add_argument("--nli",    action="store_true",
                        help="Enable NLI scoring (needs GPU)")
    args = parser.parse_args()

    sample_size = 10 if args.mode == "dev" else 400

    runner = ExperimentRunner(
        sample_size  = sample_size,
        use_nli      = args.nli,
        llm_provider = args.llm,
    )

    if args.config:
        print(f"\nRunning single config: {args.config}")
        result = runner.run_single_config(args.config)
        print(json.dumps(result, indent=2))
    else:
        print(f"\nRunning {'DEV' if args.mode == 'dev' else 'FULL'} experiment...")
        runner.run_all()