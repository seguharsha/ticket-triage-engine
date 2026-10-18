"""Check the configured LLM works for both jobs, and how fast it is on this machine.

  python -m scripts.check_llm

Runs one classification and one reply draft through the same prompts and parsers as the worker.
"""

import uuid

from app.config import settings
from app.services.llm import LLMError, extract_json, get_llm_client
from app.services.replies import Source, parse_draft, reply_prompt
from app.services.routing import classification_prompt, parse_llm_labels

TICKET = ("SSO login loop\n\nSince this morning our whole team gets sent back to the Okta page every time "
          "we try to sign in to TaskFlow. Cleared cookies, tried Chrome and Edge. Nobody can work!")
SOURCES = [
    Source(uuid.uuid4(), "Okta sign-in keeps redirecting", "Hi! This happens when the SAML certificate in "
           "TaskFlow is outdated after an Okta rotation. Go to Admin > Security > SSO, upload the new certificate "
           "from Okta, and click Save. Users can sign in again immediately. - TaskFlow Support", 0.82),
    Source(uuid.uuid4(), "Invoice missing VAT number", "Add the VAT number under Billing > Company details; "
           "new invoices will include it. - TaskFlow Support", 0.21),
]


def main() -> None:
    llm = get_llm_client()
    if llm is None:
        raise SystemExit("LLM_PROVIDER is none")
    print(f"provider {settings.llm_provider}: {llm.model}")
    try:
        result = llm.complete(classification_prompt(TICKET), max_tokens=100, json_mode=True)
        labels = parse_llm_labels(extract_json(result.text))
        print(f"\nclassify: {labels or result.text!r}  ({result.latency_ms / 1000:.1f}s, "
              f"{result.input_tokens} in / {result.output_tokens} out tokens)")
        if labels != {"category": "login_auth", "team": "Platform", "priority": "P1"}:
            print("  (expected login_auth / Platform / P1; small models miss sometimes, which the router handles)")

        result = llm.complete(reply_prompt(TICKET, SOURCES), max_tokens=400)
        draft = parse_draft(result.text, len(SOURCES))
        print(f"\ndraft ({draft.status}, cites {draft.cited}, {result.latency_ms / 1000:.1f}s):\n{draft.text or result.text}")
    except LLMError as exc:
        raise SystemExit(f"LLM error: {exc}") from exc


if __name__ == "__main__":
    main()
