# Redrob India Runs Hackathon — Candidate Ranking System

An AI-powered candidate ranker built for Redrob AI's "India Runs" hackathon
(Intelligent Candidate Discovery & Ranking Challenge). Given a job description
and a pool of 100,000 candidate profiles, this system produces a ranked
top-100 shortlist with per-candidate reasoning — designed to understand who
actually fits a role rather than who matches the most keywords.

## Approach in brief

A two-phase pipeline:

- **Phase A — offline precompute** (precompute.py, no time limit): extracts
  structured features from each candidate, runs consistency/honeypot checks,
  and computes sentence-transformer embeddings for semantic JD matching.
- **Phase B — timed ranking** (rank.py, must finish in 5 min / 16GB / CPU /
  no network): pure vectorized scoring over the precomputed artifacts,
  applying a JD-fit rubric (config/jd_profile.yaml) to produce the final
  ranked CSV with generated reasoning text.

The scoring formula is deliberately relevance-first and gated, not purely
additive: a candidate's experience, location, and behavioral signals can
modify their score, but cannot compensate for having no genuine relevance to
the role. Skill matches found only in a candidate's self-reported skills list
(easy to stuff with buzzwords) count for far less than matches grounded in
their actual career-history text — this is the core defense against the
keyword-stuffing trap the JD explicitly warns about.

## Repo structure

precompute.py                  Phase A: features + honeypot checks + embeddings
rank.py                        Phase B: scoring + CSV output
reasoning.py                   Fact-grounded, tone-matched reasoning generator
config/jd_profile.yaml         The JD rubric (single source of truth)
config/jd_text.txt             JD text used for embedding-based scoring
data/sample_candidates.json    50-candidate schema reference
artifacts/                     Precomputed features + embeddings (Git LFS)
outputs/submission.csv         Final top-100 ranked submission
validate_submission.py         Official format validator
submission_metadata.yaml       Filled submission metadata
requirements.txt               Python dependencies
RUNNING_100K.md                Step-by-step local run instructions

## Getting the data

The 100K-candidate pool (candidates.jsonl, ~465MB uncompressed) is not
committed to this repo. Place it at data/candidates.jsonl before running
the pipeline. See RUNNING_100K.md for full setup instructions.

## Quick start

pip install -r requirements.txt

python precompute.py --candidates data/candidates.jsonl --jd-text-file config/jd_text.txt --out-dir artifacts --embedder sentence-transformers

python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv --top-n 100

python validate_submission.py outputs/submission.csv

## Development notes

The rubric and scoring logic went through a structured iteration process: a
50-candidate development cycle that surfaced and fixed real scoring bugs,
followed by an 8-case adversarial stress test probing edge cases (true
honeypots, keyword-stuffed profiles, candidates who stopped coding years ago,
inactive-but-skilled candidates). The final 100K-scale output was spot-checked
against real candidate records to confirm every reasoning claim is factually
grounded with no hallucination.
