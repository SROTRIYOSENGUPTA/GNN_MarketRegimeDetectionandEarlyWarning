"""
Guard against prose drifting away from the evidence.

Motivation. RESULTS.md carries a claim-status table listing which claims are
Robust, Supported, Not supported, Retracted or Rejected. The table and the
prose around it are maintained by hand and separately, so they drift: a claim
gets retracted in the table while two prose sections still assert it, or a
statistic is corrected in one place and left stale in three others. Both
happened in this project. With eleven findings, several retractions and a
LaTeX draft that duplicates every number, it will happen again.

Four checks, run over RESULTS.md, README.md, CHANGELOG.md and paper/*.tex:

  A. SUPERSEDED VALUES   A statistic that has been corrected must not appear
                         anywhere except where it is explicitly narrated as
                         history.
  B. RETRACTED CLAIMS    A claim the table marks Retracted / Rejected / Not
                         supported must not be asserted in prose except where
                         the retraction itself is being discussed.
  C. REGISTRY SYNC       Every falsified row in the claim table must have a
                         guard here. This is what keeps the check alive as the
                         findings change -- a new retraction with no guard is
                         itself reported.
  D. ANCHORED NUMBERS    Numbers quoted in prose are re-derived from the JSON
                         they came from, so re-running a pipeline cannot
                         silently orphan the text.

Quarantine. Checks A and B work at paragraph granularity: a match is accepted
if its paragraph also contains a correction cue ("retracted", "did not
replicate", "was published as", ...). This is why the claim table itself does
not trip check B -- the word "Retracted" sits in the same block. It is also
why CHANGELOG entries that quote old values are fine.

The patterns are deliberately loose. This reports things for a human to look
at; it is not a proof of correctness. False positives are cheap, a stale
number in a submitted paper is not.

Usage:
    python scripts/check_claim_drift.py            # check, exit 1 on findings
    python scripts/check_claim_drift.py --list     # show the registry
Exit codes: 0 clean, 1 findings, 2 the registry is out of sync with the table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files whose prose must not drift. CHANGELOG is included because a correction
# entry there is the canonical place old values are allowed to appear -- the
# quarantine cue mechanism handles it.
SCAN = ["RESULTS.md", "README.md", "CHANGELOG.md",
        "paper/main.tex", "paper/results_section.tex"]

# A paragraph containing any of these is discussing a correction, so a stale
# value or retracted claim appearing inside it is intentional.
CUES = [
    r"retract", r"supersed", r"correct(?:ed|ion)", r"did not replicate",
    r"was published as", r"were published as", r"earlier (?:version|revision|draft)",
    r"previously", r"no longer", r"an earlier", r"incorrectly", r"stale",
    r"not supported", r"rejected", r"n\.s\.", r"failed to replicate",
    r"we retract", r"first reported", r"originally",
]
CUE_RE = re.compile("|".join(CUES), re.I)

# ---------------------------------------------------------------- check A
# Statistics that have been corrected. `bad` is intentionally context-anchored
# (e.g. requires a nearby "t") so a bare coincidental numeral does not fire.
SUPERSEDED = [
    dict(name="horizon t-stat, 5d",
         bad=r"t[^0-9\n]{0,12}1\.99",  canonical="t = +1.55",
         why="seed replicates were pooled as independent observations (3x n)"),
    dict(name="horizon t-stat, 20d",
         bad=r"t[^0-9\n]{0,12}0\.32",  canonical="t = -0.26",
         why="same seed-replicate inflation"),
    dict(name="horizon t-stat, 60d",
         bad=r"t[^0-9\n]{0,12}1\.22",  canonical="t = -1.00",
         why="same seed-replicate inflation"),
    dict(name="Sprint 1 5d-quintile bootstrap p",
         bad=r"bootstrap\s*(?:\$?p\$?)\s*[=<]\s*0?\.001\b",
         canonical="bootstrap p = 0.017",
         why="pooled bootstrap saw each calendar return once per seed"),
]

# ---------------------------------------------------------------- check B
# Keyed by the claim text in the RESULTS.md status table, so check C can verify
# coverage. `asserts` are patterns that mean the claim is being *made*.
RETRACTED = {
    "Monotone granularity gradient": dict(
        asserts=[r"monotonic(?:ally)?\s+(?:as|with|from)",
                 r"rises monotonic", r"monotone gradient",
                 r"improves monotonic"],
        note="the robust result is a STEP saturating at ~25 groups"),
    "Signal works better as industry rotation": dict(
        asserts=[r"industry[- ]rotation book (?:is|works) better",
                 r"rotation (?:book )?outperform"],
        note="group-level rotation is significantly worse (-5.35 bp, t=-2.78)"),
    "Graph adds information at tradeable horizons": dict(
        asserts=[r"graph adds information at (?:longer|tradeable)",
                 r"incremental $R\^?2\$? is positive"],
        note="incremental R2 is negative at 20d and 60d"),
    "Better portfolio construction recovers the economic value": dict(
        asserts=[r"construction recovers the (?:economic )?value",
                 r"better (?:portfolio )?construction (?:does )?recover"],
        note="pre-registered cell is null (d Sharpe -0.032, t -0.15)"),
    "Graph signals improve tradeable long-short spread": dict(
        asserts=[r"improves? the (?:tradeable |realised )?long-short spread",
                 r"widens the long-short spread"],
        note="worse in 6/7 windows"),
    "Supply-chain edges beat a same-granularity public classification": dict(
        asserts=[r"supply-chain (?:edges|graph) (?:beat|outperform|exceed)",
                 r"proprietary (?:data|relationship data) (?:beats|outperforms)",
                 r"Bloomberg (?:graph )?(?:beats|outperforms)"],
        note="+0.0015, n.s. -- free GICS is statistically indistinguishable"),
    "Signal is distinct from momentum / reversal / industry effects": dict(
        asserts=[r"survives controls for momentum",
                 r"distinct from (?:momentum|standard factors)",
                 r"independent of momentum"],
        note="retains 23% of magnitude, pooled t=+0.64"),
    "5-day graph advantage is exploitable": dict(
        asserts=[r"5-day (?:advantage|edge) is (?:exploitable|tradeable)",
                 r"exploit the (?:five|5)-day"],
        note="every 5d CER is negative at 30-38 turns/yr"),
    "Look-ahead from the static snapshot explains the results": dict(
        asserts=[r"look-ahead explains", r"driven by look-ahead",
                 r"contamination (?:explains|drives) the"],
        note="advantage is smallest in the most-contaminated window"),
}

FALSIFIED_STATUSES = {"retracted", "rejected", "not supported"}

# ---------------------------------------------------------------- check D
# Prose numbers re-derived from the JSON that produced them.
ANCHORS = [
    dict(source="results/sprint1/sprint1_seedfix.json",
         path=["pf_H20_continuous", "d_sharpe"], fmt="{:.3f}",
         must_appear=["-0.032", "−0.032"],
         label="Sprint 1 primary spec, delta Sharpe"),
    dict(source="results/sprint1/sprint1_seedfix.json",
         path=["pf_H20_continuous", "t"], fmt="{:.2f}",
         must_appear=["-0.15", "−0.15"],
         label="Sprint 1 primary spec, t-statistic"),
    dict(source="results/sprint1/sprint1_seedfix.json",
         path=["pf_H20_continuous", "boot_p"], fmt="{:.2f}",
         must_appear=["0.42", ".42"],
         label="Sprint 1 primary spec, bootstrap p"),
]


def paragraphs(text):
    """Yield (start_line, block). Paragraph granularity, because markdown and
    LaTeX both wrap sentences across lines -- a line window would split a
    claim from its own retraction note."""
    line = 1
    for block in re.split(r"\n\s*\n", text):
        yield line, block
        line += block.count("\n") + 2


def strip_comments(text, is_tex):
    return re.sub(r"(?<!\\)%.*", "", text) if is_tex else text


def line_of(block_start, block, idx):
    return block_start + block[:idx].count("\n")


def load_files():
    out = {}
    for rel in SCAN:
        p = REPO / rel
        if p.exists():
            out[rel] = strip_comments(p.read_text(), rel.endswith(".tex"))
    return out


def check_superseded(files):
    hits = []
    for rel, text in files.items():
        for start, block in paragraphs(text):
            quarantined = bool(CUE_RE.search(block))
            for entry in SUPERSEDED:
                for m in re.finditer(entry["bad"], block, re.I):
                    if quarantined:
                        continue
                    hits.append(dict(
                        check="A superseded value", file=rel,
                        line=line_of(start, block, m.start()),
                        found=m.group(0).replace("\n", " "),
                        detail=f'{entry["name"]}: canonical is {entry["canonical"]} '
                               f'({entry["why"]})'))
    return hits


def check_retracted(files):
    hits = []
    for rel, text in files.items():
        for start, block in paragraphs(text):
            if CUE_RE.search(block):
                continue
            for claim, spec in RETRACTED.items():
                # One report per (file, line, claim). Several patterns guard
                # each claim and they overlap by design, so a single sentence
                # can match more than one; reporting each match would triple
                # the noise for one problem.
                seen = set()
                for pat in spec["asserts"]:
                    for m in re.finditer(pat, block, re.I):
                        ln = line_of(start, block, m.start())
                        if (ln, claim) in seen:
                            continue
                        seen.add((ln, claim))
                        hits.append(dict(
                            check="B retracted claim asserted", file=rel,
                            line=ln,
                            found=m.group(0).replace("\n", " "),
                            detail=f'"{claim}" -- {spec["note"]}'))
    return hits


STATUS_HEADING = re.compile(r"^#+\s*\d*\.?\s*Summary of claim status", re.I | re.M)


def parse_status_table():
    """Rows of the RESULTS.md claim-status table as (claim, status).

    Scoped to the "Summary of claim status" section. Scanning the whole file
    would treat any three-column table as claims -- RESULTS.md has a dozen
    of them, and a per-window results table whose last column happened to
    read like a status would be silently mistaken for a claim row.
    """
    text = (REPO / "RESULTS.md").read_text()
    m = STATUS_HEADING.search(text)
    if not m:
        return []
    rest = text[m.end():]
    nxt = re.search(r"^#+\s", rest, re.M)
    section = rest[:nxt.start()] if nxt else rest

    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3 or set(cells[0]) <= set("-: "):
            continue
        status = re.sub(r"[*_]", "", cells[2]).strip()
        if status.lower() in {"status"}:
            continue
        rows.append((re.sub(r"[*_]", "", cells[0]).strip(), status))
    return rows


def check_registry_sync():
    problems = []
    rows = parse_status_table()
    if not rows:
        problems.append(dict(check="C registry sync", file="RESULTS.md", line=0,
                             found="no table parsed",
                             detail="could not find the claim-status table; "
                                    "the parser or the table format changed"))
        return problems
    falsified = [(c, s) for c, s in rows if s.lower() in FALSIFIED_STATUSES]
    for claim, status in falsified:
        if claim not in RETRACTED:
            problems.append(dict(
                check="C registry sync", file="scripts/check_claim_drift.py",
                line=0, found=f'unguarded: "{claim}" [{status}]',
                detail="a falsified claim with no entry in RETRACTED, so prose "
                       "asserting it would not be caught. Add assert patterns."))
    for claim in RETRACTED:
        if claim not in {c for c, _ in rows}:
            problems.append(dict(
                check="C registry sync", file="scripts/check_claim_drift.py",
                line=0, found=f'stale guard: "{claim}"',
                detail="guarded here but absent from the status table; the "
                       "claim was renamed or removed. Update the key."))
    return problems


def check_anchors(files):
    hits = []
    prose = "\n".join(files.values())
    for a in ANCHORS:
        src = REPO / a["source"]
        if not src.exists():
            hits.append(dict(check="D anchored number", file=a["source"], line=0,
                             found="source missing",
                             detail=f'{a["label"]}: cannot verify, file absent'))
            continue
        node = json.loads(src.read_text())
        for k in a["path"]:
            node = node[k]
        actual = a["fmt"].format(node)
        variants = {actual, actual.replace("-", "−")}
        if not (variants & set(a["must_appear"])):
            hits.append(dict(
                check="D anchored number", file=a["source"], line=0,
                found=f"JSON now says {actual}",
                detail=f'{a["label"]}: prose is written for '
                       f'{a["must_appear"][0]}. Re-run produced a different '
                       f'value -- update the prose and this anchor together.'))
            continue
        if not any(v in prose for v in variants):
            hits.append(dict(
                check="D anchored number", file="(prose)", line=0,
                found=f"{actual} not found in any scanned file",
                detail=f'{a["label"]}: the JSON value is no longer quoted '
                       f'anywhere; the text may have been dropped or reworded.'))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--list", action="store_true",
                    help="print the registry and the parsed status table")
    args = ap.parse_args()

    files = load_files()

    if args.list:
        print(f"Scanning {len(files)} files: {', '.join(files)}\n")
        print(f"Check A -- {len(SUPERSEDED)} superseded values guarded:")
        for e in SUPERSEDED:
            print(f"  {e['name']:32s} canonical {e['canonical']}")
        print(f"\nCheck B -- {len(RETRACTED)} falsified claims guarded:")
        for c in RETRACTED:
            print(f"  {c}")
        print(f"\nCheck D -- {len(ANCHORS)} anchored numbers:")
        for a in ANCHORS:
            print(f"  {a['label']:38s} <- {a['source']}:{'.'.join(a['path'])}")
        print("\nStatus table as parsed:")
        for claim, status in parse_status_table():
            mark = "!" if status.lower() in FALSIFIED_STATUSES else " "
            print(f" {mark} [{status:18s}] {claim}")
        return 0

    sync = check_registry_sync()
    hits = check_superseded(files) + check_retracted(files) + check_anchors(files)

    if not sync and not hits:
        rows = parse_status_table()
        print(f"claim-drift check: clean "
              f"({len(files)} files, {len(rows)} claims, "
              f"{len(SUPERSEDED)} guarded values, {len(ANCHORS)} anchors)")
        return 0

    for group, title in ((sync, "REGISTRY OUT OF SYNC"), (hits, "FINDINGS")):
        if not group:
            continue
        print(f"\n{title}")
        print("=" * len(title))
        for h in sorted(group, key=lambda x: (x["check"], x["file"], x["line"])):
            loc = f'{h["file"]}:{h["line"]}' if h["line"] else h["file"]
            print(f"\n  [{h['check']}] {loc}")
            print(f"    found : {h['found']}")
            print(f"    why   : {h['detail']}")

    print(f"\n{len(sync)} sync problem(s), {len(hits)} finding(s).")
    print("A finding is not automatically a bug -- if the text is genuinely "
          "narrating history, add a cue word to that paragraph.")
    return 2 if sync else 1


if __name__ == "__main__":
    sys.exit(main())
