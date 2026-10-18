import uuid

from app.services.replies import Source, parse_draft, relevant, reply_prompt, sources_json


def sources(*sims):
    return [Source(uuid.uuid4(), f"subject {i}", f"fix {i}", s) for i, s in enumerate(sims)]


def test_relevant_filters_by_similarity():
    assert [s.similarity for s in relevant(sources(0.9, 0.4, 0.6), 0.5)] == [0.9, 0.6]


def test_prompt_contains_numbered_sources_and_ticket():
    prompt = reply_prompt("SSO loops", sources(0.9, 0.8))
    assert "[S1]" in prompt and "[S2]" in prompt and "fix 1" in prompt and "SSO loops" in prompt
    assert "NO_SUGGESTION" in prompt


def test_parse_draft_with_citations():
    d = parse_draft("Clear your cache [S1]. Then re-login [S2].", 3)
    assert d.status == "ready" and d.cited == [1, 2]


def test_parse_draft_no_suggestion():
    assert parse_draft("NO_SUGGESTION", 3).status == "no_suggestion"
    assert parse_draft("  no_suggestion.", 3).status == "no_suggestion"
    assert parse_draft("", 3).status == "no_suggestion"


def test_uncited_draft_is_discarded():
    d = parse_draft("Have you tried turning it off and on again?", 3)
    assert d.status == "no_suggestion" and "cited no" in d.reason


def test_invalid_citations_removed():
    d = parse_draft("Do this [S1] and that [S7].", 2)
    assert d.cited == [1]
    assert "[S7]" not in d.text and "[S1]" in d.text


def test_draft_citing_only_nonexistent_sources_is_discarded():
    assert parse_draft("Do that [S9].", 2).status == "no_suggestion"


def test_sources_json_marks_cited_and_links():
    s = sources(0.91234, 0.7)
    out = sources_json(s, [2])
    assert [o["cited"] for o in out] == [False, True]
    assert out[0]["similarity"] == 0.9123
    assert out[1]["url"] == f"/tickets/{s[1].ticket_id}"
