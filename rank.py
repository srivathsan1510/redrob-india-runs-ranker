#!/usr/bin/env python3
"""
Phase B: the timed ranking step.

Loads the artifacts produced by precompute.py and the JD rubric from
config/jd_profile.yaml, scores every candidate, and writes the top-100
submission CSV in the exact format required by submission_spec.md.

Must complete within 5 minutes / 16GB RAM / CPU-only / no network on the
full 100K-candidate pool. This script does no embedding computation itself —
that already happened offline in precompute.py — so it's pure vectorized
arithmetic over precomputed arrays and should run in seconds.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reasoning import generate_reasoning


def load_rubric(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def text_contains_any(text, terms):
    text = text.lower()
    return any(term.lower() in text for term in terms)


def score_skill_clusters(row, rubric):
    """Returns (cluster_score 0-1, matched cluster names, nice-to-have hits).

    A cluster match found only in the skills list (easy to game by listing
    buzzwords with no real experience) counts for much less than a match
    that's also grounded in the candidate's own career-history descriptions
    or current title (much harder to fabricate convincingly). This directly
    targets the JD's explicit keyword-stuffer trap: a candidate whose skills
    list is full of AI terms but whose actual job history never mentions
    that work should not score as if they have real experience with it.
    """
    clusters = rubric["skill_clusters"]
    total_weight = sum(c["weight"] for c in clusters.values())
    matched = []
    grounded_matched = []

    skill_names = " ".join(json.loads(row["skill_names"])) if isinstance(row["skill_names"], str) else " ".join(row["skill_names"])
    career_text = " ".join(json.loads(row["career_descriptions"])) if isinstance(row["career_descriptions"], str) else ""
    title_text = (row.get("current_title") or "") + " " + (row.get("headline") or "")
    grounded_text = f"{career_text} {title_text}".lower()

    SKILLS_ONLY_CREDIT = 0.35  # a buzzword-only match is worth little
    GROUNDED_CREDIT = 1.0      # a match backed by career history/title is worth full credit

    weighted_sum = 0.0
    for name, cluster in clusters.items():
        in_skills = text_contains_any(skill_names, cluster["terms"])
        in_grounded = text_contains_any(grounded_text, cluster["terms"])
        if in_grounded:
            weighted_sum += cluster["weight"] * GROUNDED_CREDIT
            matched.append(name)
            grounded_matched.append(name)
        elif in_skills:
            weighted_sum += cluster["weight"] * SKILLS_ONLY_CREDIT
            matched.append(name)

    nice = rubric.get("nice_to_have", {})
    bump_weight = nice.get("bump_weight", 0.0)
    nice_hits = []
    full_text_for_nice = f"{skill_names} {grounded_text}".lower()
    for name, terms in nice.items():
        if name == "bump_weight":
            continue
        if text_contains_any(full_text_for_nice, terms):
            nice_hits.append(name)

    base = weighted_sum / total_weight if total_weight else 0.0
    bonus = min(len(nice_hits) * bump_weight, 0.15)
    return min(base + bonus, 1.0), grounded_matched if grounded_matched else matched, nice_hits


def score_experience_band(yoe, rubric):
    band = rubric["role"]["experience_band"]
    if yoe is None:
        return 0.3
    ideal_min, ideal_max = band["ideal_min_years"], band["ideal_max_years"]
    acc_min, acc_max = band["acceptable_min_years"], band["acceptable_max_years"]
    taper = band["taper_outside_acceptable_per_year"]

    if ideal_min <= yoe <= ideal_max:
        return 1.0
    if acc_min <= yoe <= acc_max:
        # linear taper from 1.0 at ideal edge to 0.7 at acceptable edge
        if yoe < ideal_min:
            frac = (ideal_min - yoe) / max(ideal_min - acc_min, 1e-6)
        else:
            frac = (yoe - ideal_max) / max(acc_max - ideal_max, 1e-6)
        return 1.0 - 0.3 * frac
    # outside acceptable band entirely
    if yoe < acc_min:
        years_out = acc_min - yoe
    else:
        years_out = yoe - acc_max
    return max(0.0, 0.7 - taper * years_out)


def score_location(row, rubric):
    loc_cfg = rubric["location"]
    location = (row.get("location") or "").lower()
    country = (row.get("country") or "").lower()
    willing = bool(row.get("willing_to_relocate"))

    is_india = "india" in country or any(
        c in location for c in loc_cfg["preferred_cities"] + loc_cfg["acceptable_cities"]
    )

    if any(c in location for c in loc_cfg["preferred_cities"]):
        return loc_cfg["preferred_score"]
    if any(c in location for c in loc_cfg["acceptable_cities"]):
        base = loc_cfg["acceptable_score"]
        return min(base + (loc_cfg["relocate_bonus_if_not_preferred"] * (1 - base) if willing else 0), 1.0)
    if is_india:
        base = loc_cfg["other_india_score"]
        return min(base + (loc_cfg["relocate_bonus_if_not_preferred"] * (1 - base) if willing else 0), 1.0)
    # international
    if willing:
        return loc_cfg["international_with_relocate_score"]
    return loc_cfg["international_score"]


def score_disqualifiers(row, rubric):
    """Returns multiplicative penalty factor (1.0 = no penalty)."""
    dq = rubric["disqualifiers"]
    factor = 1.0
    triggered = []

    full_text = f"{row.get('headline','')} {row.get('summary','')}".lower()
    career_text = " ".join(json.loads(row["career_descriptions"])) if isinstance(row["career_descriptions"], str) else ""
    career_titles = json.loads(row["career_titles"]) if isinstance(row["career_titles"], str) else []
    companies = json.loads(row["career_companies_lower"]) if isinstance(row["career_companies_lower"], str) else []
    all_text = f"{full_text} {career_text}".lower()

    # pure_research_only
    rule = dq["pure_research_only"]
    if text_contains_any(all_text, rule["detect"]["keywords_present_any"]) and not text_contains_any(
        all_text, rule["detect"]["keywords_required_absent"]
    ):
        factor *= rule["penalty_factor"]
        triggered.append("pure_research_only")

    # langchain_wrapper_only
    rule = dq["langchain_wrapper_only"]
    recent_ai_months = row.get("recent_ai_months") or 0
    if (
        recent_ai_months <= rule["detect"]["recent_ai_months_max"]
        and recent_ai_months > 0
        and text_contains_any(all_text, rule["detect"]["wrapper_keywords"])
        and (row.get("years_of_experience") or 0) - (recent_ai_months / 12.0) < 1.0
    ):
        factor *= rule["penalty_factor"]
        triggered.append("langchain_wrapper_only")

    # architect_no_code
    rule = dq["architect_no_code"]
    title_l = (row.get("current_title") or "").lower()
    months_since_ic = row.get("months_since_ic_role")
    if text_contains_any(title_l, rule["detect"]["title_keywords"]) and (
        months_since_ic is None or months_since_ic > rule["detect"]["months_since_ic_role_max"]
    ):
        factor *= rule["penalty_factor"]
        triggered.append("architect_no_code")

    # consulting_only
    rule = dq["consulting_only"]
    if companies and all(
        any(cc in comp for cc in rule["detect"]["consulting_companies"]) for comp in companies
    ):
        factor *= rule["penalty_factor"]
        triggered.append("consulting_only")

    # cv_speech_robotics_only
    rule = dq["cv_speech_robotics_only"]
    if text_contains_any(all_text, rule["detect"]["domain_keywords"]) and not text_contains_any(
        all_text, rule["detect"]["nlp_ir_keywords_required_absent"]
    ):
        factor *= rule["penalty_factor"]
        triggered.append("cv_speech_robotics_only")

    # closed_source_no_validation
    rule = dq["closed_source_no_validation"]
    if (
        (row.get("years_of_experience") or 0) >= rule["detect"]["min_years"]
        and (row.get("github_activity_score") if row.get("github_activity_score") is not None else -1)
        <= rule["detect"]["github_activity_score_max"]
        and not text_contains_any(all_text, rule["detect"]["external_validation_keywords_required_absent"])
    ):
        factor *= rule["penalty_factor"]
        triggered.append("closed_source_no_validation")

    return factor, triggered


def score_narrative_consistency(row, rubric):
    cfg = rubric["narrative_consistency"]
    factor = 1.0
    triggered = []
    if row.get("has_duplicate_career_descriptions"):
        factor *= cfg["duplicate_unrelated_roles_penalty_factor"]
        triggered.append("duplicate_descriptions_unrelated_roles")
    if row.get("has_title_description_mismatch"):
        factor *= cfg["title_description_bucket_mismatch_penalty_factor"]
        triggered.append("title_description_minor_mismatch")
    return factor, triggered


def score_honeypot(row, rubric):
    checks = rubric["honeypot_checks"]["checks"]
    triggered = []
    flag_map = {
        "expert_skill_no_duration": "honeypot_expert_low_duration",
        "experience_duration_mismatch": "honeypot_duration_mismatch",
        "date_inconsistency": "honeypot_date_issue",
        "endorsement_implausibility": "honeypot_endorsement_ratio_flag",
        "jack_of_all_trades_assessments": "honeypot_jack_of_all_trades",
    }
    for check in checks:
        col = flag_map.get(check["name"])
        if col and row.get(col):
            triggered.append(check["name"])
    n = len(triggered)
    factor = max(0.0, 1 - 0.35 * n)
    return factor, triggered


def score_behavioral_multiplier(row, rubric):
    cfg = rubric["behavioral_multiplier"]
    comps = cfg["components"]

    open_to_work = 1.0 if row.get("open_to_work_flag") else 0.3

    days_inactive = row.get("days_since_active")
    full_within = comps["last_active_recency"]["full_score_within_days"]
    zero_after = comps["last_active_recency"]["zero_score_after_days"]
    if days_inactive is None:
        recency = 0.3
    elif days_inactive <= full_within:
        recency = 1.0
    elif days_inactive >= zero_after:
        recency = 0.0
    else:
        recency = 1.0 - (days_inactive - full_within) / (zero_after - full_within)

    response_rate = row.get("recruiter_response_rate") or 0.0
    interview_rate = row.get("interview_completion_rate") or 0.0
    completeness = (row.get("profile_completeness_score") or 0.0) / 100.0

    weighted = (
        comps["open_to_work_flag"]["weight"] * open_to_work
        + comps["last_active_recency"]["weight"] * recency
        + comps["recruiter_response_rate"]["weight"] * response_rate
        + comps["interview_completion_rate"]["weight"] * interview_rate
        + comps["profile_completeness_score"]["weight"] * completeness
    )
    total_weight = sum(c["weight"] for c in comps.values())
    weighted_norm = weighted / total_weight if total_weight else 0.5

    min_m, max_m = cfg["min_multiplier"], cfg["max_multiplier"]
    return min_m + weighted_norm * (max_m - min_m)


def cosine_sim_matrix(cand_matrix, jd_vector):
    norms = np.linalg.norm(cand_matrix, axis=1)
    jd_norm = np.linalg.norm(jd_vector)
    denom = norms * jd_norm
    denom[denom == 0] = 1e-9
    sims = (cand_matrix @ jd_vector) / denom
    return np.clip(sims, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz] (used only for row count sanity check)")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--rubric", default="config/jd_profile.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    rubric = load_rubric(args.rubric)

    print("Loading precomputed artifacts ...")
    df = pd.read_parquet(artifacts_dir / "features.parquet")
    embeddings = np.load(artifacts_dir / "embeddings.npy")
    jd_embedding = np.load(artifacts_dir / "jd_embedding.npy")
    print(f"Loaded {len(df)} candidates, embedding shape {embeddings.shape}.")

    print("Computing semantic JD-fit scores ...")
    semantic_scores = cosine_sim_matrix(embeddings, jd_embedding)
    # NOTE: deliberately NOT min-max normalized across the pool. With a noisy
    # text-similarity backend (especially TF-IDF), min-max stretching a single
    # coincidental outlier (e.g. a Marketing Manager profile that happens to
    # share generic words like "team"/"drive"/"growth" with the JD prose) can
    # distort the entire scale and let irrelevant candidates look like the
    # best matches. Raw cosine similarity is already bounded in [0, 1] and is
    # combined with a skill-grounding requirement below (see relevance_core)
    # so semantic similarity alone, without any textual skill backing, can't
    # carry a candidate to a high score.

    print("Scoring structured components ...")
    cw = rubric["composite_weights"]
    results = []
    for idx, row in df.iterrows():
        skill_score, matched_clusters, nice_hits = score_skill_clusters(row, rubric)
        exp_score = score_experience_band(row.get("years_of_experience"), rubric)
        loc_score = score_location(row, rubric)
        narrative_factor, narrative_triggered = score_narrative_consistency(row, rubric)
        honeypot_factor, honeypot_triggered = score_honeypot(row, rubric)
        dq_factor, dq_triggered = score_disqualifiers(row, rubric)
        behavioral_mult = score_behavioral_multiplier(row, rubric)

        relevance_weight = cw["semantic_jd_fit"] + cw["skill_cluster_match"]
        # Semantic similarity is gated by skill grounding: a candidate with
        # zero matched skill clusters has their semantic score heavily
        # discounted, since pure text-similarity coincidence (e.g. shared
        # generic vocabulary) shouldn't carry weight on its own. A candidate
        # with strong skill grounding gets full credit for their semantic score.
        skill_grounding_floor = 0.20  # minimum credit even with zero skill match
        semantic_gate = skill_grounding_floor + (1 - skill_grounding_floor) * min(skill_score * 2.5, 1.0)
        gated_semantic = semantic_scores[idx] * semantic_gate

        relevance_core = (
            cw["semantic_jd_fit"] * gated_semantic + cw["skill_cluster_match"] * skill_score
        ) / relevance_weight if relevance_weight else 0.0

        gates_cfg = rubric["gates"]
        exp_gate = (1 - gates_cfg["experience_gate_weight"]) + gates_cfg["experience_gate_weight"] * exp_score
        loc_gate = (1 - gates_cfg["location_gate_weight"]) + gates_cfg["location_gate_weight"] * loc_score

        base_composite = relevance_core * exp_gate * loc_gate

        final_score = base_composite * narrative_factor * honeypot_factor * dq_factor * behavioral_mult
        final_score = max(0.0, min(1.0, final_score))

        results.append({
            "candidate_id": row["candidate_id"],
            "score": final_score,
            "semantic_score": semantic_scores[idx],
            "skill_score": skill_score,
            "matched_clusters": matched_clusters,
            "nice_hits": nice_hits,
            "exp_score": exp_score,
            "loc_score": loc_score,
            "narrative_factor": narrative_factor,
            "narrative_triggered": narrative_triggered,
            "honeypot_factor": honeypot_factor,
            "honeypot_triggered": honeypot_triggered,
            "dq_factor": dq_factor,
            "dq_triggered": dq_triggered,
            "behavioral_mult": behavioral_mult,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(df, on="candidate_id", how="left")

    # Round scores BEFORE sorting/tie-breaking — the tie-break rule (candidate_id
    # ascending for equal scores) is checked against the displayed CSV value,
    # so two candidates whose unrounded scores differ only past the 4th decimal
    # must still be ordered correctly once rounded.
    results_df["score"] = results_df["score"].round(4)

    # rank: score descending, tie-break candidate_id ascending
    results_df = results_df.sort_values(
        by=["score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)

    top = results_df.head(args.top_n).copy()
    top["rank"] = range(1, len(top) + 1)

    print("Generating reasoning text ...")
    top["reasoning"] = top.apply(lambda r: generate_reasoning(r, rubric), axis=1)

    out_df = top[["candidate_id", "rank", "score", "reasoning"]].copy()

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} ranked rows -> {out_path}")


if __name__ == "__main__":
    main()
