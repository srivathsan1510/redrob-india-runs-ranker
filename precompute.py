#!/usr/bin/env python3
"""
Phase A: offline precomputation.

Reads candidates.jsonl (or a small JSON sample) and produces:
  - artifacts/features.parquet      structured per-candidate features
  - artifacts/candidate_text.parquet candidate_id -> combined text (for embedding)
  - artifacts/embeddings.npy         text embedding matrix, row order = features.parquet order
  - artifacts/jd_embedding.npy       single embedding vector for the JD text

No time limit here — this can take as long as it needs (minutes for embeddings
on 100K rows). The output of this script is what rank.py loads; rank.py itself
must finish in 5 minutes.

Embedding backend is swappable via --embedder {tfidf,sentence-transformers}.
tfidf has no network/model-download dependency and runs anywhere.
sentence-transformers gives better semantic matching but requires one-time
network access to download the model (after which it's fully offline).
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_candidates(path: str):
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    opener = gzip.open if path.suffix == ".gz" else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


# ---------------------------------------------------------------------------
# Text assembly (for embeddings)
# ---------------------------------------------------------------------------

def build_candidate_text(c: dict) -> str:
    """Combine the parts of a profile that carry semantic JD-fit signal."""
    parts = []
    profile = c.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))

    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))

    skill_names = [s.get("name", "") for s in c.get("skills", [])]
    parts.append(" ".join(skill_names))

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Structured feature extraction
# ---------------------------------------------------------------------------

def months_since(date_str, reference_date):
    if not date_str:
        return None
    try:
        d = pd.to_datetime(date_str)
        ref = pd.to_datetime(reference_date)
        return (ref - d).days / 30.44
    except Exception:
        return None


def extract_features(c: dict, reference_date: str) -> dict:
    cid = c.get("candidate_id")
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    education = c.get("education", [])
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})

    skill_names_lower = [s.get("name", "").lower() for s in skills]
    all_text = build_candidate_text(c).lower()

    feat = {
        "candidate_id": cid,
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        "num_skills": len(skills),
        "skill_names": skill_names_lower,
        "num_career_entries": len(career),
        "num_education_entries": len(education),
    }

    # --- career history summaries ---
    durations = [j.get("duration_months") or 0 for j in career]
    feat["sum_career_duration_months"] = sum(durations)
    feat["career_titles"] = [j.get("title", "") for j in career]
    feat["career_companies_lower"] = [j.get("company", "").lower() for j in career]
    feat["career_descriptions"] = [j.get("description", "") for j in career]

    # months since most recent role start that looks like an IC/coding role
    # (used by the architect_no_code disqualifier)
    ic_keywords = ["engineer", "developer", "scientist", "programmer", "researcher"]
    months_since_ic = None
    for j in career:
        title_l = j.get("title", "").lower()
        if any(k in title_l for k in ic_keywords):
            m = months_since(j.get("start_date"), reference_date)
            if m is not None and (months_since_ic is None or m < months_since_ic):
                months_since_ic = m
    feat["months_since_ic_role"] = months_since_ic

    # months in "recent AI" roles (title or description mentions AI/ML/LLM work)
    ai_keywords = ["machine learning", "ml engineer", "ai engineer", "nlp", "llm",
                   "ranking", "retrieval", "recommendation", "embeddings", "deep learning"]
    recent_ai_months = 0
    for j in career:
        text_l = (j.get("title", "") + " " + j.get("description", "")).lower()
        if any(k in text_l for k in ai_keywords):
            recent_ai_months += j.get("duration_months") or 0
    feat["recent_ai_months"] = recent_ai_months

    # --- duplicate / inconsistent career description detection ---
    # Reusing identical description text across roles is only a red flag when
    # the titles it's attached to are NOT topically related (e.g. same text
    # under "Frontend Engineer" and "DevOps Engineer" — unrelated domains).
    # Reuse across genuinely related titles (e.g. "Search Engineer" and "NLP
    # Engineer" sharing a ranking-systems description) is not inconsistent.
    descs = [j.get("description", "") for j in career if j.get("description")]
    titles_for_desc = {}
    for j in career:
        d = j.get("description", "")
        if d:
            titles_for_desc.setdefault(d, []).append(j.get("title", "").lower())

    # domain buckets — generic words like "engineer"/"developer" deliberately
    # excluded so they don't make every pair of titles look "related"
    domain_buckets = {
        "ml_ai": ["ml", "machine learning", "ai", "nlp", "recommendation", "search",
                  "ranking", "retrieval", "data science", "applied"],
        "frontend": ["frontend", "front-end", "ui", "react", "angular", "mobile"],
        "backend": ["backend", "back-end", "api"],
        "devops_cloud": ["devops", "cloud", "infrastructure", "sre", "platform"],
        "data_eng": ["data engineer", "analytics", "etl", "pipeline"],
        "qa": ["qa", "quality", "test"],
        "management": ["manager", "lead", "director", "head"],
        "other_nontech": ["accountant", "hr", "marketing", "sales", "graphic", "civil", "mechanical"],
    }

    def title_domains(t):
        t = t.lower()
        return {b for b, kws in domain_buckets.items() if any(k in t for k in kws)}

    suspicious_duplicate = False
    for desc, titles in titles_for_desc.items():
        if len(titles) < 2:
            continue
        domain_sets = [title_domains(t) for t in titles]
        any_unrelated_pair = False
        for i in range(len(domain_sets)):
            for j_ in range(i + 1, len(domain_sets)):
                d1, d2 = domain_sets[i], domain_sets[j_]
                # unrelated if neither has overlapping domain buckets
                # (titles with no recognized bucket at all are treated as
                # ambiguous, not automatically flagged)
                if d1 and d2 and not (d1 & d2):
                    any_unrelated_pair = True
        if any_unrelated_pair:
            suspicious_duplicate = True

    feat["has_duplicate_career_descriptions"] = suspicious_duplicate

    # --- title/description coherence (independent of duplication) ---
    # Flags e.g. a "Cloud Engineer" entry whose description is actually
    # about QA/test-automation work, with no domain overlap at all.
    mismatched_title_desc = False
    for j in career:
        title_l = j.get("title", "").lower()
        desc_l = j.get("description", "").lower()
        if not title_l or not desc_l:
            continue
        t_domains = title_domains(title_l)
        d_domains = {b for b, kws in domain_buckets.items() if any(k in desc_l for k in kws)}
        if t_domains and d_domains and not (t_domains & d_domains):
            mismatched_title_desc = True
    feat["has_title_description_mismatch"] = mismatched_title_desc

    # --- education tier ---
    tiers = [e.get("tier", "unknown") for e in education]
    feat["education_tiers"] = tiers

    # --- skills detail for honeypot check ---
    expert_low_duration = False
    for s in skills:
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", None)
        if dur is None:
            continue
        if prof == "expert" and dur <= 6:
            expert_low_duration = True
        if prof == "advanced" and dur <= 3:
            expert_low_duration = True
    feat["honeypot_expert_low_duration"] = expert_low_duration

    # endorsements vs connections
    endorsements = signals.get("endorsements_received", 0) or 0
    connections = signals.get("connection_count", 0) or 0
    feat["honeypot_endorsement_ratio_flag"] = (
        connections > 0 and (endorsements / connections) > 5.0
    ) or (connections == 0 and endorsements > 20)

    # jack-of-all-trades assessment scores
    assess = signals.get("skill_assessment_scores", {}) or {}
    high_scores = sum(1 for v in assess.values() if v is not None and v >= 90)
    feat["honeypot_jack_of_all_trades"] = len(assess) >= 15 and high_scores >= 15

    # experience-duration mismatch
    yoe = profile.get("years_of_experience") or 0
    sum_months = feat["sum_career_duration_months"]
    feat["honeypot_duration_mismatch"] = (
        yoe > 0 and sum_months > 0 and (sum_months / 12.0) > yoe * 1.5
    )

    # date inconsistencies
    date_issue = False
    for j in career:
        sd, ed, is_cur = j.get("start_date"), j.get("end_date"), j.get("is_current")
        if is_cur and ed:
            date_issue = True
        if sd and ed:
            try:
                if pd.to_datetime(sd) > pd.to_datetime(ed):
                    date_issue = True
            except Exception:
                pass
    feat["honeypot_date_issue"] = date_issue

    # --- behavioral signals (pass through) ---
    feat["open_to_work_flag"] = bool(signals.get("open_to_work_flag", False))
    feat["last_active_date"] = signals.get("last_active_date")
    months_inactive = months_since(signals.get("last_active_date"), reference_date)
    feat["days_since_active"] = months_inactive * 30.44 if months_inactive is not None else None
    feat["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0) or 0
    feat["interview_completion_rate"] = signals.get("interview_completion_rate", 0) or 0
    feat["profile_completeness_score"] = signals.get("profile_completeness_score", 0) or 0
    feat["github_activity_score"] = signals.get("github_activity_score", -1)
    feat["notice_period_days"] = signals.get("notice_period_days", None)
    feat["willing_to_relocate"] = bool(signals.get("willing_to_relocate", False))

    # --- text bundle ---
    feat["combined_text"] = build_candidate_text(c)
    feat["all_text_lower"] = all_text

    return feat


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def embed_tfidf(texts, jd_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english")
    all_texts = texts + [jd_text]
    matrix = vectorizer.fit_transform(all_texts)
    cand_matrix = matrix[:-1].toarray().astype(np.float32)
    jd_vector = matrix[-1].toarray().astype(np.float32)[0]
    return cand_matrix, jd_vector


def embed_sentence_transformers(texts, jd_text, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    cand_matrix = model.encode(texts, batch_size=64, show_progress_bar=True,
                                convert_to_numpy=True).astype(np.float32)
    jd_vector = model.encode([jd_text], convert_to_numpy=True).astype(np.float32)[0]
    return cand_matrix, jd_vector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz] or sample .json")
    parser.add_argument("--jd-text-file", required=True, help="Path to a plain text file with the JD text")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--embedder", choices=["tfidf", "sentence-transformers"], default="tfidf")
    parser.add_argument("--reference-date", default=None,
                         help="Date to compute recency features against (default: today)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_date = args.reference_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    with open(args.jd_text_file, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Extracting structured features ...")
    rows = [extract_features(c, reference_date) for c in candidates]
    df = pd.DataFrame(rows)

    # list/dict columns can't go to parquet directly without conversion;
    # store them as JSON strings, rank.py will json.loads them back.
    list_cols = ["skill_names", "career_titles", "career_companies_lower",
                 "career_descriptions", "education_tiers"]
    for col in list_cols:
        df[col] = df[col].apply(json.dumps)

    features_path = out_dir / "features.parquet"
    df.drop(columns=["combined_text", "all_text_lower"]).to_parquet(features_path, index=False)
    print(f"Saved structured features -> {features_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    print(f"Computing embeddings with backend={args.embedder} ...")
    texts = df["combined_text"].fillna("").tolist()
    if args.embedder == "tfidf":
        cand_emb, jd_emb = embed_tfidf(texts, jd_text)
    else:
        cand_emb, jd_emb = embed_sentence_transformers(texts, jd_text)

    np.save(out_dir / "embeddings.npy", cand_emb)
    np.save(out_dir / "jd_embedding.npy", jd_emb)
    # save the candidate_id order so rank.py can align rows safely
    df[["candidate_id"]].to_csv(out_dir / "embedding_row_order.csv", index=False)

    print(f"Saved embeddings -> {out_dir / 'embeddings.npy'} shape={cand_emb.shape}")
    print(f"Saved JD embedding -> {out_dir / 'jd_embedding.npy'} shape={jd_emb.shape}")
    print("Precompute complete.")


if __name__ == "__main__":
    main()
