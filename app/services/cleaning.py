"""Turn raw ticket text (often an email) into the text we embed and classify.

Removes HTML, quoted reply chains and signatures, so two customers reporting the same
problem look alike even if one replied from Outlook and the other from a phone.
"""

import html
import re

from bs4 import BeautifulSoup

_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_BLOCK_TAGS = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol"]

# Everything from the first of these headers onwards is an older message being quoted.
_REPLY_HEADERS = [
    re.compile(r"^On\b[^\n]{0,300}?(?:\n[^\n]{0,300}?)?\bwrote:[ \t]*$", re.M | re.I),  # Gmail, Apple Mail
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.M | re.I),
    re.compile(r"^From:[^\n]*\n(?:[^\n]*\n){0,3}?(?:Sent|Date):", re.M | re.I),  # Outlook
    re.compile(r"^_{20,}[ \t]*$", re.M),
]
_SIG_DELIMITER = re.compile(r"^--[ \t]*$", re.M)
_DEVICE_SIG = re.compile(
    r"^(?:Sent from my \S[^\n]*|Sent from Mail for Windows[^\n]*|Get Outlook for \S[^\n]*)$", re.M | re.I
)
_SIGN_OFF = re.compile(
    r"^(?:thanks|thank you|thanks again|many thanks|thx|regards|best regards|kind regards|warm regards"
    r"|best|best wishes|cheers|sincerely|br)[\s,.!]*$",
    re.I,
)
_SUBJECT_PREFIX = re.compile(r"^\s*(?:(?:re|fw|fwd|aw)\s*:\s*)+", re.I)


def strip_html(text: str) -> str:
    if not _HTML_TAG.search(text):
        return html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.select("blockquote, div.gmail_quote, div.yahoo_quoted"):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(_BLOCK_TAGS):
        block.append("\n")
    return soup.get_text()


def strip_quoted_reply(text: str) -> str:
    cut = len(text)
    for pattern in _REPLY_HEADERS:
        match = pattern.search(text)
        # A header at position 0 means there is no new text above it; keep everything.
        if match and match.start() > 0:
            cut = min(cut, match.start())
    lines = [line for line in text[:cut].splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(lines)


def strip_signature(text: str) -> str:
    match = _SIG_DELIMITER.search(text)
    if match and match.start() > 0:
        text = text[: match.start()]
    text = _DEVICE_SIG.sub("", text)

    # "Thanks,\nJane Doe\nAcme Corp" near the end: drop the sign-off and the short lines after it.
    lines = text.rstrip().splitlines()
    for i in range(max(1, len(lines) - 6), len(lines)):
        if _SIGN_OFF.match(lines[i].strip()):
            tail = [line for line in lines[i + 1 :] if line.strip()]
            if len(tail) <= 5 and all(len(line) <= 60 for line in tail):
                return "\n".join(lines[:i])
    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_subject(subject: str) -> str:
    return normalize_whitespace(_SUBJECT_PREFIX.sub("", strip_html(subject)))


def clean_body(body: str) -> str:
    plain = normalize_whitespace(strip_html(body))
    cleaned = normalize_whitespace(strip_signature(strip_quoted_reply(plain)))
    # Never return nothing for a non-empty ticket; fall back to the de-HTML'd text.
    return cleaned or plain


def clean_ticket_text(subject: str, body: str) -> str:
    parts = [clean_subject(subject), clean_body(body)]
    return "\n\n".join(part for part in parts if part)
