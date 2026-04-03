"""
terminology_lexicon.py
======================
Domain-specific terminology sets used by the T-score in DHS.

Sources:
  Healthcare : SNOMED-CT common terms + clinical abbreviations
  Legal      : Black's Law Dictionary core terms
  Finance    : XBRL taxonomy + common financial jargon

Usage:
  from src.terminology_lexicon import TerminologyLexicon
  lex   = TerminologyLexicon()
  terms = lex.extract_terms("Patient shows signs of hypertension", DomainType.HEALTHCARE)
  # → {"hypertension"}
"""

import re
import logging
from typing import Set
from core.domain_models import DomainType

logger = logging.getLogger(__name__)


class TerminologyLexicon:
    """
    Lightweight static lexicon for domain-term extraction.
    No external downloads required — fully self-contained.
    These term sets can be expanded as the project grows.
    """

    def __init__(self):
        self._lexicons = {
            DomainType.HEALTHCARE : self._build_healthcare_lexicon(),
            DomainType.LEGAL      : self._build_legal_lexicon(),
            DomainType.FINANCE    : self._build_finance_lexicon(),
        }
        logger.info("TerminologyLexicon initialised for all 3 domains.")

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def extract_terms(self, text: str, domain: DomainType) -> Set[str]:
        """
        Return the set of domain terms found in `text`.
        Matching is case-insensitive, whole-word.
        """
        lexicon = self._lexicons.get(domain, set())
        text_lower = text.lower()
        found: Set[str] = set()

        for term in lexicon:
            # whole-word match (handles multi-word terms too)
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text_lower):
                found.add(term)

        return found

    def get_lexicon(self, domain: DomainType) -> Set[str]:
        """Return the full lexicon for a domain."""
        return self._lexicons.get(domain, set())

    def lexicon_size(self) -> dict:
        return {d.value: len(terms) for d, terms in self._lexicons.items()}

    # ─────────────────────────────────────────────────────────────────────────
    #  LEXICON BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_healthcare_lexicon(self) -> Set[str]:
        """
        Clinical terminology — SNOMED-CT common subset + medical abbreviations.
        Covers: diseases, procedures, drugs, lab values, anatomy.
        """
        return {
            # Cardiovascular
            "hypertension", "htn", "myocardial infarction", "mi", "heart failure",
            "atrial fibrillation", "afib", "coronary artery disease", "cad",
            "stroke", "tia", "transient ischemic attack", "angina", "arrhythmia",
            "systolic", "diastolic", "ejection fraction",
            # Metabolic
            "diabetes mellitus", "type 2 diabetes", "t2dm", "insulin resistance",
            "hyperglycemia", "hypoglycemia", "hba1c", "glycated hemoglobin",
            "obesity", "bmi", "dyslipidemia", "hyperlipidemia",
            # Renal
            "chronic kidney disease", "ckd", "glomerular filtration rate", "gfr",
            "creatinine", "proteinuria", "renal failure", "dialysis",
            # Respiratory
            "chronic obstructive pulmonary disease", "copd", "asthma", "dyspnea",
            "pneumonia", "pulmonary fibrosis", "spirometry", "fev1",
            # Oncology
            "carcinoma", "malignancy", "metastasis", "chemotherapy", "radiotherapy",
            "immunotherapy", "tumor", "biopsy", "lymphoma", "leukemia",
            # Drugs / treatments
            "antihypertensive", "beta blocker", "ace inhibitor", "statin",
            "anticoagulant", "warfarin", "heparin", "aspirin", "metformin",
            "insulin", "corticosteroid", "antibiotic", "analgesic",
            # Procedures
            "percutaneous coronary intervention", "pci", "coronary artery bypass graft",
            "cabg", "endoscopy", "colonoscopy", "mri", "ct scan", "echocardiogram",
            # Lab values
            "hemoglobin", "hematocrit", "white blood cell", "wbc", "platelet",
            "sodium", "potassium", "creatinine", "bun", "troponin", "bnp",
            # Clinical terms
            "prognosis", "diagnosis", "etiology", "pathogenesis", "remission",
            "comorbidity", "contraindication", "indication", "efficacy",
            "randomized controlled trial", "rct", "meta-analysis", "systematic review",
            "confidence interval", "odds ratio", "relative risk", "p-value",
        }

    def _build_legal_lexicon(self) -> Set[str]:
        """
        Legal terminology — Black's Law Dictionary core + common contract law.
        Covers: contract law, statutory, case law, property, tort, IP.
        """
        return {
            # Contract law
            "consideration", "breach of contract", "indemnification", "indemnity",
            "liquidated damages", "force majeure", "arbitration", "mediation",
            "jurisdiction", "governing law", "choice of law", "venue",
            "sublicense", "sublicensing", "assignment", "novation",
            "termination", "termination for cause", "termination for convenience",
            "cure period", "notice period", "warranty", "representation",
            "covenant", "condition precedent", "condition subsequent",
            # IP law
            "intellectual property", "copyright", "trademark", "patent",
            "trade secret", "infringement", "fair use", "work for hire",
            "license", "exclusive license", "non-exclusive license", "royalty",
            # Corporate / statutory
            "fiduciary duty", "duty of care", "duty of loyalty",
            "shareholder", "stockholder", "board of directors", "quorum",
            "merger", "acquisition", "due diligence", "material adverse change",
            "securities", "sec", "disclosure", "prospectus",
            # Tort
            "negligence", "strict liability", "proximate cause", "causation",
            "damages", "compensatory damages", "punitive damages",
            "plaintiff", "defendant", "claimant", "respondent",
            # Procedure
            "discovery", "deposition", "interrogatories", "subpoena",
            "summary judgment", "motion to dismiss", "injunction",
            "preliminary injunction", "temporary restraining order", "tro",
            "class action", "settlement", "consent decree",
            # Statutory
            "statute of limitations", "standing", "mootness", "ripeness",
            "due process", "equal protection", "fifth amendment", "first amendment",
            "common law", "equity", "precedent", "stare decisis",
        }

    def _build_finance_lexicon(self) -> Set[str]:
        """
        Financial terminology — XBRL taxonomy + common financial analysis terms.
        Covers: income statement, balance sheet, cash flow, ratios, risk.
        """
        return {
            # Income statement
            "revenue", "net revenue", "gross revenue", "net sales",
            "cost of goods sold", "cogs", "gross profit", "gross margin",
            "operating income", "ebit", "ebitda", "net income", "earnings",
            "earnings per share", "eps", "diluted eps", "basic eps",
            "operating expenses", "opex", "capex", "capital expenditure",
            "depreciation", "amortization", "d&a",
            # Balance sheet
            "total assets", "total liabilities", "shareholders equity",
            "stockholders equity", "retained earnings", "goodwill",
            "intangible assets", "property plant and equipment", "ppe",
            "accounts receivable", "inventory", "accounts payable",
            "long term debt", "short term debt", "cash and equivalents",
            "working capital", "current assets", "current liabilities",
            # Cash flow
            "operating cash flow", "free cash flow", "fcf",
            "capital allocation", "share buyback", "dividend",
            "cash from operations", "cash from investing", "cash from financing",
            # Ratios & metrics
            "price to earnings", "p/e ratio", "price to book", "p/b",
            "return on equity", "roe", "return on assets", "roa",
            "return on invested capital", "roic", "debt to equity", "leverage",
            "interest coverage ratio", "current ratio", "quick ratio",
            # Risk
            "credit risk", "market risk", "liquidity risk", "operational risk",
            "value at risk", "var", "conditional value at risk", "cvar",
            "interest rate risk", "irrbb", "foreign exchange risk", "fx risk",
            "beta", "volatility", "standard deviation",
            # Regulatory / reporting
            "gaap", "ifrs", "10-k", "10-q", "8-k", "sec filing",
            "annual report", "quarterly report", "earnings release",
            "guidance", "forward looking statement", "fiscal year",
            "quarter", "year over year", "yoy", "quarter over quarter", "qoq",
            # Market terms
            "market capitalization", "market cap", "enterprise value", "ev",
            "initial public offering", "ipo", "secondary offering",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test — python src/terminology_lexicon.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)

    lex = TerminologyLexicon()
    print("\n── Lexicon sizes ──")
    pprint.pprint(lex.lexicon_size())

    tests = [
        (DomainType.HEALTHCARE,
         "The patient was diagnosed with hypertension and type 2 diabetes. HbA1c was 8.2.",
         {"hypertension", "type 2 diabetes", "hba1c"}),
        (DomainType.LEGAL,
         "The breach of contract resulted in liquidated damages under the governing law of Delaware.",
         {"breach of contract", "liquidated damages", "governing law"}),
        (DomainType.FINANCE,
         "Apple reported EBITDA of $29B with EPS of $2.18 and strong free cash flow.",
         {"ebitda", "eps", "free cash flow"}),
    ]

    print("\n── Term extraction tests ──")
    all_pass = True
    for domain, text, expected in tests:
        found = lex.extract_terms(text, domain)
        missing = expected - found
        status = "✓" if not missing else f"✗ missing: {missing}"
        print(f"  [{domain.value}] {status}")
        print(f"    Found : {found}")
        if missing:
            all_pass = False

    if all_pass:
        print("\n✓ All terminology_lexicon.py tests passed.")
    else:
        print("\n⚠ Some tests failed — check lexicon terms above.")