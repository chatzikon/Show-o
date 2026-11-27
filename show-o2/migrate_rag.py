#!/usr/bin/env python3
"""One-command migration of Show-o2 investigative RAG v0.4.3 to v0.5 (13 entries).

Run from show-o2:
    python migrate_rag_to_v05.py --root investigative_rag

Writes a timestamped backup, preserves old->new ID mapping, updates prompt,
and invokes rebuild_index_isolated.py using the current Python interpreter.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

# Explicit, reviewed mapping: new ID => (title, old IDs)
GROUPS = [
    ("Portrait photographs and identity misuse", ["VR-001"]),
    ("Professional portraits and contextual impersonation", ["VR-002"]),
    ("Private photographs and nonconsensual distribution", ["VR-003"]),
    ("Private images and extortion risk", ["VR-004"]),
    ("Children's privacy and identifiable locations", ["VR-005"]),
    ("Social relationships, group photographs, and bystander privacy", ["VR-006", "VR-011", "VR-024"]),
    ("Residential location and entrance exposure", ["VR-007", "VR-018"]),
    ("Residential security information exposure", ["VR-008"]),
    ("Travel photographs and contextual information exposure", ["VR-009"]),
    ("Geographic landmarks and background location clues", ["VR-010", "VR-013"]),
    ("School, club, and event affiliation exposure", ["VR-014", "VR-017"]),
    ("Professional and organizational affiliation exposure", ["VR-015"]),
    ("Private interior and household information exposure", ["VR-016"]),
]
REMOVED = {"VR-012", "VR-019", "VR-020", "VR-021", "VR-022", "VR-023"}
NEW_VERSION = "0.5.0"

# Human-curated, discriminative FAISS text for each new entry. Unioning all of the
# old retrieval strings tends to broaden matches and create redundant findings.
RETRIEVAL_TEXTS = [
    "Portrait photographs and identity misuse. Visual cues: clear single-person portrait, selfie or headshot showing a reusable likeness. Potential unauthorized reuse as a fake account or profile; image alone does not establish misuse.",
    "Professional portraits and contextual impersonation. Visual cues: portrait with professional uniform, badge, work attire, affiliation or recognizable role. Potential misuse of role-associated imagery to imply authority; no inference of actual impersonation.",
    "Private photographs and nonconsensual distribution. Visual cues: person photographed in an evidently private or intimate context. Unauthorized sharing could expose intimate or sensitive imagery; permission and distribution cannot be inferred visually.",
    "Private images and extortion risk. Visual cues: private or intimate imagery that could be misused in an extortion scenario. Extortion requires independent evidence of a coercive threat; a private photograph alone is insufficient.",
    "Children's privacy and identifiable locations. Visual cues: child near recognizable home, school, venue, club or background location identifiers. Unauthorized sharing may disclose contextual or location information; avoid inferring targeting or distress.",
    "Social relationships, group photographs, and bystander privacy. Visual cues: group photographs, household gatherings, visible bystanders, event participants or apparent social context. Sharing may expose associations and bystander images; consent status cannot be known from photo alone.",
    "Residential location and entrance exposure. Visual cues: distinctive home exterior, frontage, gate, porch, door, entryway or contextual residential identifiers. Sharing may reveal location or property details; no evidence of intrusion or targeting from ordinary exteriors.",
    "Residential security information exposure. Visual cues: doorbell camera, CCTV, alarm signage, entry control, gate, security device or recognizable access point. Sharing may reveal limited security arrangements; do not infer vulnerability or planned burglary.",
    "Travel photographs and contextual information exposure. Visual cues: travel scene, vacation setting, tourist backdrop, luggage or recognizable destination. Publication might disclose travel context; photograph alone cannot establish current absence from home.",
    "Geographic landmarks and background location clues. Visual cues: landmark, street or venue sign, distinctive skyline, readable geographic cues or identifiable background. Sharing may help contextualize location; no automatic inference of exact address.",
    "School, club, and event affiliation exposure. Visual cues: school uniform, club shirt, team logo, venue banner, affiliation badge, event or activity signage. Sharing may disclose group or institutional affiliations and attendance context.",
    "Professional and organizational affiliation exposure. Visual cues: visible workplace, uniform, company mark, professional badge, branded equipment or organizational environment. Sharing may reveal affiliations or workplace context; not evidence of wrongdoing.",
    "Private interior and household information exposure. Visual cues: room layouts, home interiors, family belongings, interior furnishing, household clues or identifiable private spaces. Unauthorized circulation may reveal private household context without suggesting a crime.",
]

# These instructions replace the removed negative-control entries, and sharpen
# the intended photo-misuse rather than scene-crime task.
PROMPT_RULES = (
    '        + "10. Ordinary photographs are not evidence of wrongdoing. "\n'
    '        + "A normal portrait does not establish impersonation, identity theft, "\n'
    '        + "extortion, or unauthorized use. Ordinary landscapes do not establish "\n'
    '        + "criminal activity or a crime location. The presence of domestic animals "\n'
    '        + "does not establish animal abuse, illegal trade, or criminal intent.\\n"\n'
    '        + "11. Focus on potential consequences of unauthorized photograph "\n'
    '        + "sharing or reuse, not general scene-condition assessment. "\n'
    '        + "Differentiate information visibly exposed in the photograph from "\n'
    '        + "hypothetical misuse that requires independent evidence.\\n"\n'
    '        + "12. Do not generate a finding solely to say nothing suspicious is "\n'
    '        + "visible. Explain only specific, image-supported privacy, location, "\n'
    '        + "identity or security information exposure from relevant retrieved VRs. "\n'
    '        + "If no substantive retrieved VR applies, return an empty findings list.\\n"\n'
)


def unique_list(values):
    seen, out = set(), []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def combine_text(rows, key):
    parts = unique_list([r.get(key, "").strip() for r in rows if isinstance(r.get(key), str) and r[key].strip()])
    return " ".join(parts)


def combine_list(rows, key):
    return unique_list([v for row in rows for v in row.get(key, [])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="investigative_rag", help="RAG root, relative to working directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    showo = root.parent
    corpus = root / "data" / "knowledge.jsonl"
    structured = root / "structured_output.py"
    builder = showo / "rebuild_index_isolated.py"
    migration_log = root / "data" / "vr_migration_v05.json"
    for file in (corpus, structured, builder):
        if not file.is_file():
            parser.error(f"Missing required file: {file}")

    rows = [json.loads(s) for s in corpus.read_text(encoding="utf-8").splitlines() if s.strip()]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        parser.error("Duplicate IDs in current knowledge database")
    expected = REMOVED | {old for _, ids in GROUPS for old in ids}
    if set(by_id) != expected or len(rows) != 24:
        parser.error("Expected the exact original 24-entry v0.4.3 database; refusing to alter a different corpus")

    original_prompt = structured.read_text(encoding="utf-8")
    if "def structured_prompt(" not in original_prompt or "def _unfence(" not in original_prompt:
        parser.error("Unrecognized structured_output.py; refusing automatic prompt edit")
    if '"9. Be sensitive to relevant investigative information, "' not in original_prompt:
        parser.error("Expected rule 9 not found; refusing automatic prompt edit")
    if '"10. Ordinary photographs are not evidence of wrongdoing. "' in original_prompt:
        parser.error("Rule 10 already present; migration appears already partially applied")

    # Insert new rules at end of structured_prompt return value, before next function.
    fence = '    )\n\n\ndef _unfence('
    if original_prompt.count(fence) != 1:
        parser.error("Unexpected structure around structured_prompt; refusing automatic prompt edit")
    updated_prompt = original_prompt.replace(fence, PROMPT_RULES + fence, 1)

    migrated = []
    id_map = {}
    for number, (title, old_ids) in enumerate(GROUPS, 1):
        new_id = f"VR-{number:03d}"
        old_rows = [by_id[old] for old in old_ids]
        new = copy.deepcopy(old_rows[0])
        new["id"] = new_id
        new["title"] = title
        new["database_version"] = NEW_VERSION
        new["merged_from_vr_ids"] = list(old_ids)
        new["retrieval_text"] = RETRIEVAL_TEXTS[number - 1]
        for key in ("observation_cues", "alternative_explanations", "source_ids", "attack_mappings",
                    "visual_evidence_required", "risk_preconditions", "not_supported_inferences"):
            new[key] = combine_list(old_rows, key)
        for key in ("information_exposed", "source_supported_fact", "potential_misuse_hypothesis",
                    "verification_needed", "retrieval_safety_note"):
            new[key] = combine_text(old_rows, key)
        if len(old_rows) > 1:
            new["risk_category"] = "photograph_information_exposure"
            new["source_verified_for_general_claim"] = all(r.get("source_verified_for_general_claim", False) for r in old_rows)
            new["source_support_level"] = "merged_source_guidance_with_analyst_interpretation"
            new["review_note"] = ("Merged and renumbered from " + ", ".join(old_ids)
                                  + "; inherited hypotheses and limitations require expert review.")
        else:
            new["review_note"] = (new.get("review_note", "") + f" Renumbered from {old_ids[0]} in v0.5.").strip()
        migrated.append(new)
        for old_id in old_ids:
            id_map[old_id] = new_id

    assert len(migrated) == 13
    assert [r["id"] for r in migrated] == [f"VR-{i:03d}" for i in range(1, 14)]
    assert len(set(id_map)) == 18
    assert set(id_map) | REMOVED == set(by_id)
    print("13-entry mapping:")
    for row in migrated:
        print(f"  {row['id']}  <= {', '.join(row['merged_from_vr_ids'])}: {row['title']}")
    print("Removed:", ", ".join(sorted(REMOVED)))
    if args.dry_run:
        print("DRY RUN ONLY: no files changed")
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "backups" / f"pre_v05_{timestamp}"
    if backup.exists():
        parser.error(f"Backup directory already exists: {backup}")
    backup.mkdir(parents=True)
    for file in (corpus, structured, root / "index" / "manifest.json", root / "index" / "knowledge.faiss"):
        if file.is_file():
            rel = file.relative_to(root)
            dest = backup / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    print("Backed up previous corpus, prompt, and index to:", backup)

    migration_log.write_text(json.dumps({
        "database_version": NEW_VERSION,
        "previous_entry_count": 24,
        "new_entry_count": 13,
        "old_to_new_ids": id_map,
        "removed_ids": sorted(REMOVED),
        "note": "Historical IDs map to conceptual successors, not always one-to-one. Inspect original backup for full wording.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in migrated) + "\n", encoding="utf-8")
    structured.write_text(updated_prompt, encoding="utf-8")
    print("Wrote 13-entry knowledge.jsonl and updated structured_output.py")

    command = [sys.executable, "-I", str(builder), "--root", str(root)]
    print("Rebuilding FAISS:", " ".join(command))
    subprocess.run(command, cwd=showo, check=True)
    manifest_path = root / "index" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [r["id"] for r in migrated]
    if manifest.get("ids") != expected_ids:
        raise RuntimeError(f"FAISS manifest IDs differ: {manifest.get('ids')}")
    print("SUCCESS: FAISS index rebuilt with 13 consecutively numbered entries.")
    print("Migration mapping:", migration_log)
    print("Restart inference to reload the index.")


if __name__ == "__main__":
    main()