from app.services.classifier import FieldPrediction
from app.services.routing import classification_prompt, decide, parse_llm_labels

FIELDS = ["category", "team"]


def pred(value, conf, **probs):
    return FieldPrediction(value, conf, probs or {value: conf})


def primary(cat=("bug", 0.9), team=("Product", 0.9), prio=("P3", 0.5)):
    return {"category": pred(*cat), "team": pred(*team), "priority": pred(*prio)}


def test_confident_classifier_auto_routes_without_llm():
    d = decide(primary(), None, FIELDS, 0.7)
    assert d.routing == "auto" and not d.used_llm


def test_low_priority_confidence_does_not_block_routing():
    d = decide(primary(prio=("P2", 0.3)), None, FIELDS, 0.7)
    assert d.routing == "auto"


def test_uncertain_without_llm_goes_to_review():
    d = decide(primary(team=("Product", 0.5)), None, FIELDS, 0.7)
    assert d.routing == "review"
    assert "team" in d.reason


def test_llm_agreement_auto_routes():
    d = decide(primary(team=("Product", 0.5)), {"category": "bug", "team": "Product", "priority": "P3"}, FIELDS, 0.7)
    assert d.routing == "auto" and d.used_llm
    assert d.labels["team"].value == "Product"


def test_llm_disagreement_goes_to_review_with_llm_label():
    p = primary(team=("Product", 0.5))
    p["team"].probabilities = {"Product": 0.5, "Platform": 0.4}
    d = decide(p, {"team": "Platform"}, FIELDS, 0.7)
    assert d.routing == "review"
    assert d.labels["team"].value == "Platform"
    assert d.labels["team"].confidence == 0.4


def test_llm_missing_an_uncertain_field_goes_to_review():
    d = decide(primary(team=("Product", 0.5)), {"category": "bug"}, FIELDS, 0.7)
    assert d.routing == "review"


def test_confident_fields_ignore_llm():
    d = decide(primary(team=("Product", 0.5)), {"category": "billing", "team": "Product"}, FIELDS, 0.7)
    assert d.routing == "auto"
    assert d.labels["category"].value == "bug"


def test_no_classifier_never_auto_routes():
    d = decide({}, {"category": "bug", "team": "Product", "priority": "P2"}, FIELDS, 0.7)
    assert d.routing == "review"
    assert d.labels["team"].value == "Product"


def test_no_classifier_no_llm_review_with_no_labels():
    d = decide({}, None, FIELDS, 0.7)
    assert d.routing == "review" and d.labels == {}


def test_parse_llm_labels_validates_and_normalizes_case():
    assert parse_llm_labels({"category": "BUG", "priority": "p1", "team": "Marketing"}) == {
        "category": "bug", "priority": "P1"}
    assert parse_llm_labels(None) == {}
    assert parse_llm_labels({"category": 3}) == {}


def test_classification_prompt_lists_every_label():
    prompt = classification_prompt("my sso is broken")
    for label in ["login_auth", "data_export", "Payments", "Support", "P1", "P4", "my sso is broken"]:
        assert label in prompt
