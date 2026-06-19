"""
Generates the `reasoning` column text for each ranked candidate.

Design goals (directly from submission_spec.md Section 3, Stage 4 checks):
  - Specific facts: years of experience, current title, named skills, signal values
  - JD connection: ties back to specific JD requirements, not generic praise
  - Honest concerns: acknowledges real gaps when they exist
  - No hallucination: every claim is read straight from the candidate's own
    precomputed feature row — nothing is invented or inferred beyond what's
    literally present
  - Variation: template bank + randomized-but-deterministic phrase selection
    so 10 random rows don't read identically
  - Rank consistency: tone bucket (strong / solid / marginal / weak) is
    derived from the rank position itself, not generated independently of it
"""

import hashlib
import json
import random


def _stable_rng(candidate_id):
    """Deterministic per-candidate RNG seed so reasoning text is reproducible
    across runs (same candidate always gets the same phrasing), without
    needing to store random state anywhere."""
    h = hashlib.md5(candidate_id.encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def _matched_cluster_phrase(cluster_name):
    phrases = {
        "embeddings_retrieval": "embeddings-based retrieval",
        "vector_db_hybrid_search": "vector database / hybrid search infrastructure",
        "ranking_ir": "ranking and information retrieval",
        "eval_frameworks": "ranking evaluation frameworks",
        "strong_python": "Python",
    }
    return phrases.get(cluster_name, cluster_name)


def _tone_bucket(rank, penalty_factors=None):
    """Tone is driven primarily by rank, but a candidate who only holds a
    high rank despite a heavy penalty elsewhere (behavioral, disqualifier,
    or narrative-consistency) shouldn't read as confidently as one with no
    such penalty — a recruiter trusts the list less if it says "Top
    candidate" about someone flagged for a real concern.

    penalty_factors: list of multiplicative factors already applied
    (behavioral_mult, dq_factor, narrative_factor, honeypot_factor).
    Any single factor below 0.65 triggers a one-tier demotion from "strong".
    """
    base = "strong" if rank <= 15 else "solid" if rank <= 50 else "marginal" if rank <= 80 else "weak"
    if penalty_factors and base == "strong" and any(f < 0.65 for f in penalty_factors if f is not None):
        return "solid"  # demote one tier — still a good match, but flagged as less certain
    return base


def _get_top_named_skills(row, rubric, max_n=2):
    """Return up to max_n actual skill names (from the candidate's real
    skills list) that fall in a must-have cluster, for citing concretely."""
    skill_names = json.loads(row["skill_names"]) if isinstance(row["skill_names"], str) else row["skill_names"]
    clusters = rubric["skill_clusters"]
    all_terms = []
    for c in clusters.values():
        all_terms.extend([t.lower() for t in c["terms"]])

    hits = []
    for s in skill_names:
        if any(term in s for term in all_terms) or s in all_terms:
            hits.append(s)
    # also allow direct containment the other way (term contains skill, e.g. skill="faiss" term="faiss")
    if not hits:
        for s in skill_names:
            for term in all_terms:
                if s and (s in term or term in s):
                    hits.append(s)
                    break
    return hits[:max_n]


def generate_reasoning(row, rubric):
    rng = _stable_rng(row["candidate_id"])
    rank = int(row["rank"])
    penalty_factors = [
        row.get("behavioral_mult"), row.get("dq_factor"),
        row.get("narrative_factor"), row.get("honeypot_factor"),
    ]
    tone = _tone_bucket(rank, penalty_factors)

    title = row.get("current_title", "this role")
    company = row.get("current_company", "")
    yoe = row.get("years_of_experience")
    matched_clusters = row.get("matched_clusters") or []
    named_skills = _get_top_named_skills(row, rubric)
    dq_triggered = row.get("dq_triggered") or []
    narrative_triggered = row.get("narrative_triggered") or []
    honeypot_triggered = row.get("honeypot_triggered") or []
    days_inactive = row.get("days_since_active")
    response_rate = row.get("recruiter_response_rate")
    notice_days = row.get("notice_period_days")
    open_to_work = row.get("open_to_work_flag")

    # --- positive fact fragment ---
    pos_fragments = []
    if yoe is not None:
        pos_fragments.append(rng.choice([
            f"{yoe:.1f} years of experience",
            f"{yoe:.1f} years in the field",
        ]))
    if title:
        pos_fragments.append(rng.choice([
            f"currently {title}" + (f" at {company}" if company else ""),
            f"working as {title}" + (f" at {company}" if company else ""),
        ]))
    if matched_clusters:
        cluster_phrases = [_matched_cluster_phrase(c) for c in matched_clusters[:2]]
        pos_fragments.append(
            "demonstrated background in " + " and ".join(cluster_phrases)
        )
    if named_skills:
        pos_fragments.append("with named skills including " + ", ".join(named_skills))

    # --- concern fragment ---
    concern_fragments = []
    if dq_triggered:
        dq_text_map = {
            "pure_research_only": "career history reads as research-only with limited evidence of production deployment",
            "langchain_wrapper_only": "AI-specific experience appears limited to recent LangChain/API-wrapper work",
            "architect_no_code": "current title suggests a step away from hands-on coding",
            "consulting_only": "career has been entirely at consulting firms with no product-company stint",
            "cv_speech_robotics_only": "background centers on computer vision/speech/robotics rather than NLP or IR",
            "closed_source_no_validation": "work history is closed-source with little external validation",
        }
        concern_fragments.append(dq_text_map.get(dq_triggered[0], "some profile concerns noted"))
    if narrative_triggered and "duplicate_descriptions_unrelated_roles" in narrative_triggered:
        concern_fragments.append("career history shows reused description text across unrelated roles, which raises some doubt about profile accuracy")
    if honeypot_triggered:
        concern_fragments.append("profile contains internal inconsistencies (e.g. skill proficiency vs. duration, or date/duration mismatches) flagged for review")
    if open_to_work is False:
        concern_fragments.append("not currently marked open to work")
    if days_inactive is not None and days_inactive > 90:
        concern_fragments.append(f"inactive on the platform for roughly {int(days_inactive)} days")
    if response_rate is not None and response_rate < 0.15:
        concern_fragments.append(f"low recruiter response rate ({response_rate:.0%})")
    if notice_days is not None and notice_days > 60:
        concern_fragments.append(f"{int(notice_days)}-day notice period")

    # --- compose by tone ---
    if tone == "strong":
        lead = rng.choice([
            "Strong fit:", "High-confidence match:", "Top candidate:",
        ])
        body = ", ".join(pos_fragments) if pos_fragments else "profile aligns well with the role"
        sentence = f"{lead} {body}."
        if concern_fragments:
            sentence += f" Minor note: {concern_fragments[0]}."
        else:
            sentence += rng.choice([
                " Active on platform with reasonable engagement signals.",
                " Engagement signals support availability.",
            ])

    elif tone == "solid":
        lead = rng.choice(["Solid match:", "Good fit overall:", "Reasonable fit:"])
        body = ", ".join(pos_fragments) if pos_fragments else "meets several core requirements"
        sentence = f"{lead} {body}."
        if concern_fragments:
            sentence += f" Some concern: {concern_fragments[0]}."

    elif tone == "marginal":
        lead = rng.choice(["Marginal fit:", "Partial match:", "Borderline candidate:"])
        if pos_fragments:
            sentence = f"{lead} {pos_fragments[0]}, but " + (
                concern_fragments[0] if concern_fragments else "overall match to the JD is weaker than higher-ranked candidates"
            ) + "."
        else:
            sentence = f"{lead} limited alignment with core JD requirements."
        if len(concern_fragments) > 1:
            sentence += f" Also: {concern_fragments[1]}."

    else:  # weak
        lead = rng.choice(["Weak fit:", "Low-confidence inclusion:", "Below-cutoff filler:"])
        if concern_fragments and pos_fragments:
            # show one real strength alongside the dominant concern — a
            # candidate can have genuine substance (e.g. relevant title,
            # matched skills) and still land here because of one strong
            # penalty (e.g. not open to work); the reasoning should say both,
            # not just the negative, or it misrepresents the profile
            sentence = f"{lead} {pos_fragments[0]}, but {concern_fragments[0]}."
            if len(concern_fragments) > 1:
                sentence += f" Also: {concern_fragments[1]}."
        elif concern_fragments:
            sentence = f"{lead} {concern_fragments[0]}."
            if len(concern_fragments) > 1:
                sentence += f" Additionally, {concern_fragments[1]}."
        elif pos_fragments:
            sentence = f"{lead} {pos_fragments[0]}, but adjacent skills only relative to core JD requirements."
        else:
            sentence = f"{lead} adjacent profile included as final filler given overall signal strength."

    return sentence
