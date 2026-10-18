from app.services.cleaning import (
    clean_body,
    clean_subject,
    clean_ticket_text,
    strip_html,
    strip_quoted_reply,
    strip_signature,
)


def test_plain_text_is_unchanged():
    assert clean_body("I can't log in since this morning.") == "I can't log in since this morning."


def test_strip_html_keeps_inline_text_together_and_breaks_blocks():
    text = strip_html("<p>My <b>login</b> fails</p><p>Error&nbsp;500</p><script>x()</script>")
    assert "My login fails" in text
    assert "x()" not in text
    assert clean_body("<p>My <b>login</b> fails</p><p>Error 500</p>") == "My login fails\nError 500"


def test_strip_html_removes_gmail_quote_block():
    body = '<div>New problem here</div><div class="gmail_quote">On Mon wrote: old stuff</div>'
    assert clean_body(body) == "New problem here"


def test_html_entities_unescaped_in_plain_text():
    assert clean_body("Price &gt; expected &amp; wrong") == "Price > expected & wrong"


def test_strip_gmail_reply_chain():
    body = (
        "Still broken after clearing cache.\n\n"
        "On Mon, Oct 12, 2026 at 10:00 AM Support <support@taskflow.io> wrote:\n"
        "> Please try clearing your cache.\n"
    )
    assert clean_body(body) == "Still broken after clearing cache."


def test_strip_gmail_reply_header_split_over_two_lines():
    body = "Still broken.\n\nOn Mon, Oct 12, 2026 at 10:00 AM Support Team\n<support@taskflow.io> wrote:\nold"
    assert clean_body(body) == "Still broken."


def test_strip_outlook_reply_chain():
    body = (
        "Invoice is still wrong.\n\n"
        "From: TaskFlow Billing <billing@taskflow.io>\n"
        "Sent: Monday, October 12, 2026 9:00 AM\n"
        "To: me\n"
        "Subject: Your invoice\n\nHere is your invoice."
    )
    assert clean_body(body) == "Invoice is still wrong."


def test_strip_original_message_marker():
    body = "Please refund.\n-----Original Message-----\nFrom: x"
    assert strip_quoted_reply(body).strip() == "Please refund."


def test_reply_header_at_start_is_kept():
    body = "On Monday the export wrote: nothing useful"
    assert strip_quoted_reply(body) == body


def test_strip_dash_dash_signature():
    body = "Export is missing columns.\n-- \nJane Doe\nACME Corp | +1 555 0100"
    assert clean_body(body) == "Export is missing columns."


def test_strip_device_signature():
    assert clean_body("App crashes on launch.\n\nSent from my iPhone") == "App crashes on launch."


def test_strip_sign_off_block():
    body = "The board won't load.\nIt spins forever.\n\nThanks,\nJane Doe\nHead of Ops, Acme"
    assert strip_signature(body).strip() == "The board won't load.\nIt spins forever."


def test_sign_off_with_long_tail_is_not_a_signature():
    body = (
        "Hi team\nThanks\n"
        "Since the update yesterday every time I open a project the timeline view crashes my browser tab"
    )
    assert "timeline view crashes" in clean_body(body)


def test_body_that_is_only_a_signature_falls_back_to_original():
    assert clean_body("Thanks,\nJane") != ""


def test_clean_subject_drops_reply_prefixes():
    assert clean_subject("RE: Fwd: re:  Login broken") == "Login broken"


def test_clean_ticket_text_combines_subject_and_body():
    text = clean_ticket_text("Re: SSO down", "<p>Can't sign in with Okta.</p>\n--\nBob")
    assert text == "SSO down\n\nCan't sign in with Okta."


def test_whitespace_normalized():
    assert clean_body("a  \t b\r\n\r\n\r\n\r\nc") == "a b\n\nc"
