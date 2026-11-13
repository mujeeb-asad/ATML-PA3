from __future__ import annotations

import re
from importlib.metadata import version

from math_verify import parse, verify

#answer equivalence, used both to form answer groups (never looking at the gold answer) and to grade them.
#
#  math       math-verify. two extracted answers belong to the same group when verify(representative, candidate)
#             holds, the representative being the group's first answer in rollout order. a group is graded by
#             verify(gold, representative), the same call with the gold answer in the representative's place
#  mathvista  the official mathvista rules (evaluation/calculate_score.py, normalize_extracted_answer) with the
#             released question metadata ->
#               multi_choice  "(B)" -> B, a letter inside the option range is kept, anything else is mapped to the most
#                             similar option text (levenshtein), and every answer is represented as its option letter
#               integer       str(int(float(x)))
#               float         str(round(float(x), precision))
#               list          str(x)
#             a free-form extraction the rules cannot convert (e.g. "75\%", "10-15 years") gets no normalized value
#             in the official scorer, so it can never be correct. it stays its own answer group, keyed by its raw text
#  both       a missing extraction (null) -> the single NO_ANSWER group, kept in voting and costs, always incorrect

NO_ANSWER = "NO_ANSWER"


def scorer_versions():
    return {"math_verify": version("math-verify"), "latex2sympy2_extended": version("latex2sympy2_extended"),
            "mathvista_rules": "calculate_score.py normalize_extracted_answer, testmini.json at commit 452b0f8"}


def last_boxed(text):
    #the export's math extractor -> content of the last literal \boxed{ with balanced braces, stripped. no box, an
    #empty box or an unclosed box -> None (no numeric fallback)
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j].strip() or None
    return None


#--------------------------------------------------------------------------------------------- math

def math_parse(answer):
    #wrapping in \boxed{} makes math-verify read the whole string as one latex answer
    return parse(f"\\boxed{{{answer}}}")


def math_groups(answers, gold):
    #answers -> extracted strings (None for missing) in rollout order. returns one group index per rollout and a list
    #of groups {key, correct, size}. exact string matches skip the (slower) symbolic check
    gold_parsed = math_parse(gold)
    groups, reps, index_of, out = [], [], {}, []
    for answer in answers:
        key = NO_ANSWER if answer is None else answer.strip()
        if key not in index_of:
            if key == NO_ANSWER:
                index_of[key] = len(groups)
                groups.append({"key": NO_ANSWER, "correct": False, "size": 0})
                reps.append(None)
            else:
                parsed = math_parse(key)
                match = next((g for g, rep in enumerate(reps) if rep is not None and verify(rep, parsed)), None)
                if match is None:
                    match = len(groups)
                    groups.append({"key": key, "correct": bool(verify(gold_parsed, parsed)), "size": 0})
                    reps.append(parsed)
                index_of[key] = match
        groups[index_of[key]]["size"] += 1
        out.append(index_of[key])
    return out, groups


#--------------------------------------------------------------------------------------------- mathvista

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, meta):
    #returns the normalized answer (an option letter, or the rule's string), or None when the rule gives no value
    text = str(extraction).strip()
    if meta["question_type"] == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(meta["choices"]))]
        found = re.findall(r"\(([a-zA-Z])\)", text)
        if found:
            text = found[0].upper()
        if text in letters:
            return text
        distances = [levenshtein(text, choice) for choice in meta["choices"]]
        return letters[distances.index(min(distances))] #first closest option, as in the official get_most_similar
    try:
        if meta["answer_type"] == "integer":
            return str(int(float(text)))
        if meta["answer_type"] == "float":
            return str(round(float(text), int(meta["precision"])))
        return text #list
    except ValueError:
        return None


def mathvista_groups(answers, gold, meta):
    #groups by the normalized answer. gold is the supplied answer (an option letter for multiple choice), normalized by
    #the same rule. an unconvertible extraction keeps its raw text as the key and is marked parsed = False
    gold_key = mathvista_normalize(gold, meta)
    groups, index_of, out = [], {}, []
    for answer in answers:
        if answer is None:
            key, parsed = NO_ANSWER, False
        else:
            norm = mathvista_normalize(answer, meta)
            key, parsed = (norm, True) if norm is not None else (answer.strip(), False)
        if key not in index_of:
            index_of[key] = len(groups)
            groups.append({"key": key, "correct": parsed and key == gold_key, "parsed": parsed, "size": 0})
        groups[index_of[key]]["size"] += 1
        out.append(index_of[key])
    return out, groups
