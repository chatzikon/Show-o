"""Structured, VR-grounded RAG output. No ML dependencies or imports.

The first-pass image description is authoritative for this report; the second
model pass produces investigative findings only. Both JSON and readable text
are derived from the same validated structure.
"""
import json
import re

OUTPUT_FORMAT_VERSION = "vr_structured_v1"


class StructuredOutputError(ValueError):
    pass


def structured_prompt(existing_prompt, retrieved_records):
    """Append an output contract to the already-built RAG investigative prompt."""
    allowed = [f"{r['id']} — {r['title']}" for r in retrieved_records]
    allowed_text = "\n".join(allowed) if allowed else "(none: return an empty findings array)"
    return (
        existing_prompt.rstrip()
        + "\n\n=== FINAL RESPONSE FORMAT (OVERRIDES EARLIER FORMAT INSTRUCTIONS) ===\n"
        + "Respond with ONLY valid JSON, no markdown/code fences and no commentary. "
        + "The neutral image description is generated separately; do not repeat it.\n"
        + "Allowed knowledge entries for this image:\n"
        + allowed_text + "\n"
        + "Return exactly this JSON structure:\n"
        + '{"investigative_findings": [{"vr_id": "VR-000", "vr_title": "Exact title", '
        + '"observation": "Visible, concrete scene evidence", '
        + '"significance": "Relevant investigative implication or meaningful reassuring context", '
        + '"risk": "Low", "confidence": "Medium"}]}\n'
        + "The values above are illustrative schema placeholders, NOT findings or permitted IDs.\n"
        + "RULES:\n"
        + "1. Select every retrieved VR entry that is visually relevant to the photograph. "
        + "A VR entry does NOT require evidence of criminal activity to be applicable. "
        + "Include entries related to privacy exposure, security information, contextual "
        + "clues, or other investigative significance when supported by visible details. "
        + "A retrieved entry is only a candidate; use exact allowed VR IDs and titles.\n"
        + "2. Generate one finding per applicable VR entry. "
        + "An image may have MULTIPLE applicable VR entries. "
        + "Prefer 1–4 meaningful findings when supported by the image, "
        + "but do not force a minimum. "
        + "Each finding must contribute distinct information. "
        + "Avoid repeating the same observation under different VR entries. "
        + "Return [] only when no retrieved VR entry has meaningful visual relevance.\n"
        + "3. Avoid boilerplate negatives such as 'no evidence of cybercrime, burglary, "
        + "trafficking or fraud' when those topics are not relevant to the visible scene.\n"
        + "4. Meaningful reassuring evidence IS welcome within a relevant VR finding "
        + "(e.g. a child walking calmly, no apparent distress); describe only what "
        + "is visible, not proof that the person is safe or no crime occurred.\n"
        + "5. Observation must describe concrete, visible facts. "
        + "Significance must explain WHY those facts may matter to an investigator. "
        + "Consider plausible privacy exposure, location disclosure, security context, "
        + "or other investigative implications without requiring proof of a crime. "
        + "Meaningful reassuring observations are also valuable. "
        + "Do not invent criminal intent, concealment, threats, or illegal activity "
        + "without specific supporting indicators.\n"
        + "6. Risk must be Low, Medium, High, or Not assessable. "
        + "Use Not assessable only if visual evidence is insufficient to assess a "
        + "meaningful risk; proof of an actual crime is not required for a potential-risk assessment.\n"
        + "7. Confidence must be Low, Medium, or High and reflects the complete finding.\n"
        + "8. Do not invent VR entries or introduce other keys.\n"
        + "9. Be sensitive to relevant investigative information, "
        + "not automatically suspicious of ordinary objects or behavior. "
        + "For example, ordinary boxes do not imply trafficking, "
        + "and a normal room does not imply concealment. "
        + "Avoid generic crime-related speculation.\n"
        + "10. Ordinary photographs are not evidence of wrongdoing. "
        + "A normal portrait does not establish impersonation, identity theft, "
        + "extortion, or unauthorized use. Ordinary landscapes do not establish "
        + "criminal activity or a crime location. The presence of domestic animals "
        + "does not establish animal abuse, illegal trade, or criminal intent.\n"
        + "11. Focus on potential consequences of unauthorized photograph "
        + "sharing or reuse, not general scene-condition assessment. "
        + "Differentiate information visibly exposed in the photograph from "
        + "hypothetical misuse that requires independent evidence.\n"
        + "12. Do not generate a finding solely to say nothing suspicious is "
        + "visible. Explain only specific, image-supported privacy, location, "
        + "identity or security information exposure from relevant retrieved VRs. "
        + "If no substantive retrieved VR applies, return an empty findings list.\n"
    )


def _unfence(s):
    s = s.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.I | re.S)
    return match.group(1).strip() if match else s


def _normalize_choice(value, allowed, field):
    if not isinstance(value, str):
        raise StructuredOutputError(f"{field} must be a string")
    normalized = " ".join(value.strip().lower().split())
    lookup = {v.lower(): v for v in allowed}
    if normalized not in lookup:
        raise StructuredOutputError(f"Invalid {field}: {value!r}")
    return lookup[normalized]


def _generic_absence(item):
    """Suppress generic lists of absent crime categories, not reassuring visible cues."""
    obs = item["observation"].lower().strip()
    sig = item["significance"].lower().strip()
    return bool(
        re.match(r"^(no visible (signs|indicators|evidence) of|no signs of)\b", obs)
        and re.match(r"^(no evidence of|no indicators of|no signs of)\b", sig)
    )


def parse_and_validate(raw_text, retrieved_records):
    """Return (validated findings, warnings). Raise for malformed top-level JSON.

    Only IDs actually retrieved for this image are allowed. The title is
    *replaced* from trusted retrieval metadata, never accepted from generation.
    Invalid individual findings are dropped and described in warnings.
    """
    try:
        obj = json.loads(_unfence(raw_text))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("investigative_findings"), list):
        raise StructuredOutputError("Expected a JSON object with an investigative_findings list")

    allowed = {str(r["id"]): r["title"] for r in retrieved_records}
    out, seen, warnings = [], set(), []
    invalid_count = 0
    for ix, item in enumerate(obj["investigative_findings"], 1):
        if not isinstance(item, dict):
            invalid_count += 1
            warnings.append(f"Finding {ix}: not an object; discarded")
            continue
        vr_id = item.get("vr_id")
        if vr_id not in allowed:
            invalid_count += 1
            warnings.append(f"Finding {ix}: VR ID {vr_id!r} was not retrieved; discarded")
            continue
        if vr_id in seen:
            warnings.append(f"Finding {ix}: duplicate {vr_id}; discarded")
            continue
        try:
            observation = item.get("observation")
            significance = item.get("significance")
            if not isinstance(observation, str) or not observation.strip():
                raise StructuredOutputError("missing observation")
            if not isinstance(significance, str) or not significance.strip():
                raise StructuredOutputError("missing significance")
            risk = _normalize_choice(item.get("risk"),
                                     ("Low", "Medium", "High", "Not assessable"), "risk")
            confidence = _normalize_choice(item.get("confidence"),
                                           ("Low", "Medium", "High"), "confidence")
        except StructuredOutputError as exc:
            invalid_count += 1
            warnings.append(f"Finding {ix} ({vr_id}): {exc}; discarded")
            continue
        clean = {"vr_id": vr_id, "vr_title": allowed[vr_id],
                 "observation": observation.strip(),
                 "significance": significance.strip(),
                 "risk": risk, "confidence": confidence}
        if _generic_absence(clean):
            warnings.append(f"Finding {ix} ({vr_id}): generic absence-only finding; discarded")
            continue
        if item.get("vr_title") != allowed[vr_id]:
            warnings.append(f"Finding {ix} ({vr_id}): corrected title from knowledge database")
        out.append(clean)
        seen.add(vr_id)
    if not out and invalid_count:
        raise StructuredOutputError("No valid findings remained after validation: " + "; ".join(warnings))
    return out, warnings


def render_report(image_description, findings, error=None, raw_output=None):
    """Produce human-readable TXT/console report from validated JSON findings."""
    lines = ["IMAGE DESCRIPTION", "=" * 60,
             (image_description or "(description unavailable)").strip(), "",
             "INVESTIGATIVE FINDINGS", "=" * 60]
    if error:
        lines += ["STRUCTURED OUTPUT ERROR: " + error,
                  "The raw model response is retained for debugging; no findings were validated.",
                  "", "RAW MODEL RESPONSE", "-" * 60, (raw_output or "").strip()]
    elif not findings:
        lines.append("No applicable VR knowledge entries identified in the image.")
    else:
        for f in findings:
            lines += ["", f"{f['vr_id']} — {f['vr_title']}", "-" * 60,
                      f"Observation: {f['observation']}",
                      f"Significance: {f['significance']}",
                      f"Risk: {f['risk']}", f"Confidence: {f['confidence']}"]
    return "\n".join(lines).rstrip() + "\n"