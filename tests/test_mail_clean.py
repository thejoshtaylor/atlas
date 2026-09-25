"""Tests for `mcp/atlas_mcp/mail_clean.py`.

Every name and address below is invented (`dana@example.com`, "Example Corp"),
matching the public-repository rule -- no real household appears here.

`mail-parser-reply` is a real runtime dependency (Task 1, human-approved), so
most tests exercise the library path directly. A second, smaller set of tests
forces the standard-library fallback path via `monkeypatch` to prove the
backstop rule set holds up on its own, per the threat model's "two layers"
requirement (T-09-09).
"""

from __future__ import annotations

import base64

from atlas_mcp import mail_clean


def _b64url_nopad(data: bytes) -> str:
    """Gmail's own encoding: base64url with the trailing `=` padding stripped."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


# --- clean_body: quoted replies, forwarded blocks, signatures -------------


def test_gmail_style_reply_strips_quoted_history():
    text = (
        "Hi Dana,\n\nThanks for the update. Let's meet Friday.\n\n"
        "On Fri, Oct 2, 2026 at 9:00 AM Dana Example <dana@example.com> wrote:\n"
        "> Original message text here\n"
        "> more quoted text\n"
    )
    assert mail_clean.clean_body(text) == "Hi Dana,\n\nThanks for the update. Let's meet Friday."


def test_outlook_style_header_block_stripped():
    text = (
        "New reply text here.\n\n"
        "From: Dana Example <dana@example.com>\n"
        "Sent: Friday, October 2, 2026 9:00 AM\n"
        "To: Someone <someone@example.com>\n"
        "Subject: Re: Thing\n\n"
        "Original message body.\n"
    )
    assert mail_clean.clean_body(text) == "New reply text here."


def test_forwarded_message_banner_stripped():
    text = (
        "See below.\n\n"
        "---------- Forwarded message ---------\n"
        "From: Dana Example <dana@example.com>\n"
        "Date: Fri, Oct 2, 2026\n"
        "Subject: Thing\n\n"
        "Forwarded body text.\n"
    )
    assert mail_clean.clean_body(text) == "See below."


def test_dash_dash_signature_stripped():
    text = "Reply text here.\n\n-- \nDana Example\nExample Corp\n"
    assert mail_clean.clean_body(text) == "Reply text here."


def test_sent_from_my_iphone_line_stripped():
    text = "Reply text here.\n\nSent from my iPhone\n"
    assert mail_clean.clean_body(text) == "Reply text here."


def test_body_that_is_only_quoted_history_cleans_to_empty_string():
    text = (
        "On Fri, Oct 2, 2026 at 9:00 AM Dana Example <dana@example.com> wrote:\n"
        "> quoted line 1\n"
        "> quoted line 2\n"
    )
    assert mail_clean.clean_body(text) == ""


# --- clean_body: standard-library fallback (library disabled) -------------


def test_fallback_cleans_gmail_style_reply_without_library(monkeypatch):
    monkeypatch.setattr(mail_clean, "_HAS_LIBRARY", False)
    text = (
        "Hi Dana,\n\nThanks for the update.\n\n"
        "On Fri, Oct 2, 2026 at 9:00 AM Dana Example <dana@example.com> wrote:\n"
        "> Original message text here\n"
    )
    assert mail_clean.clean_body(text) == "Hi Dana,\n\nThanks for the update."


def test_fallback_strips_forwarded_banner_without_library(monkeypatch):
    monkeypatch.setattr(mail_clean, "_HAS_LIBRARY", False)
    text = (
        "See below.\n\n"
        "---------- Forwarded message ---------\n"
        "From: Dana Example <dana@example.com>\n"
        "Date: Fri, Oct 2, 2026\n\n"
        "Forwarded body text.\n"
    )
    assert mail_clean.clean_body(text) == "See below."


def test_fallback_body_that_is_only_quoted_history_cleans_to_empty_string(monkeypatch):
    monkeypatch.setattr(mail_clean, "_HAS_LIBRARY", False)
    text = (
        "On Fri, Oct 2, 2026 at 9:00 AM Dana Example <dana@example.com> wrote:\n"
        "> quoted line 1\n"
    )
    assert mail_clean.clean_body(text) == ""


# --- html_to_text / extract_text: HTML-only bodies -------------------------


def test_html_only_body_strips_quote_and_signature_when_cleaned():
    html = (
        "<div>Hi Dana,<br>Thanks for the update.<br>"
        '<div class="gmail_quote">On Fri, Oct 2, 2026 at 9:00 AM Dana Example '
        "&lt;dana@example.com&gt; wrote:<blockquote>Original quoted text</blockquote></div>"
        '<div class="gmail_signature">Sent from my iPhone</div></div>'
    )
    payload = {
        "mimeType": "text/html",
        "body": {"data": _b64url_nopad(html.encode("utf-8"))},
        "headers": [{"name": "Content-Type", "value": "text/html; charset=UTF-8"}],
    }
    cleaned = mail_clean.clean_body(mail_clean.extract_text(payload))
    assert "Hi Dana," in cleaned
    assert "Thanks for the update." in cleaned
    assert "Original quoted text" not in cleaned
    assert "dana@example.com" not in cleaned
    assert "Sent from my iPhone" not in cleaned


def test_html_to_text_converts_br_and_p_to_newlines_and_decodes_entities():
    html = "<p>Tom &amp; Dana</p><p>Line one<br>Line two</p>"
    text = mail_clean.html_to_text(html)
    assert "Tom & Dana" in text
    assert "Line one" in text
    assert "Line two" in text
    # br and the paragraph boundary both became real line breaks
    assert "Line one\nLine two" in text


# --- extract_text: Gmail API payload walking --------------------------------


def test_extract_text_prefers_plain_part_in_multipart_payload():
    plain = "Hi Dana,\n\nThanks for the update."
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {"data": _b64url_nopad(plain.encode("utf-8"))},
                "headers": [{"name": "Content-Type", "value": "text/plain; charset=UTF-8"}],
            },
            {
                "mimeType": "text/html",
                "body": {"data": _b64url_nopad(b"<p>Hi Dana,</p><p>Thanks for the update.</p>")},
                "headers": [{"name": "Content-Type", "value": "text/html; charset=UTF-8"}],
            },
        ],
    }
    assert mail_clean.extract_text(payload) == plain


def test_extract_text_falls_back_to_html_part_when_no_plain_part():
    payload = {
        "mimeType": "text/html",
        "body": {"data": _b64url_nopad(b"<p>Hi Dana,</p><p>Thanks for the update.</p>")},
        "headers": [{"name": "Content-Type", "value": "text/html; charset=UTF-8"}],
    }
    text = mail_clean.extract_text(payload)
    assert "Hi Dana," in text
    assert "Thanks for the update." in text


def test_extract_text_finds_plain_part_nested_under_mixed_multipart():
    plain = "Hi Dana, see attached."
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64url_nopad(plain.encode("utf-8"))},
                        "headers": [],
                    },
                ],
            },
            {
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body": {"attachmentId": "abc123"},
            },
        ],
    }
    assert mail_clean.extract_text(payload) == plain


def test_extract_text_decodes_body_data_missing_base64_padding():
    plain = "Hi"
    encoded = _b64url_nopad(plain.encode("utf-8"))
    assert not encoded.endswith("=")  # sanity: our own helper stripped the padding
    payload = {"mimeType": "text/plain", "body": {"data": encoded}, "headers": []}
    assert mail_clean.extract_text(payload) == plain


def test_extract_text_honours_declared_charset():
    text = "café"
    encoded = _b64url_nopad(text.encode("iso-8859-1"))
    payload = {
        "mimeType": "text/plain",
        "body": {"data": encoded},
        "headers": [{"name": "Content-Type", "value": "text/plain; charset=ISO-8859-1"}],
    }
    assert mail_clean.extract_text(payload) == text


# --- cap_text ----------------------------------------------------------------


def test_cap_text_returns_unchanged_when_under_limit():
    text = "Short message."
    capped, truncated = mail_clean.cap_text(text, 100)
    assert capped == text
    assert truncated is False


def test_cap_text_cuts_at_sentence_boundary_when_over_limit():
    text = "First sentence here. Second sentence continues on and on and on and on."
    capped, truncated = mail_clean.cap_text(text, 30)
    assert truncated is True
    assert capped == "First sentence here."
    assert len(capped) <= 30


def test_cap_text_falls_back_to_word_boundary_without_sentence_end():
    text = "one two three four five six seven eight nine ten"
    capped, truncated = mail_clean.cap_text(text, 12)
    assert truncated is True
    assert capped == "one two"
    assert len(capped) <= 12
    assert text.startswith(capped)


def test_model_input_cap_is_eight_thousand():
    assert mail_clean.MODEL_INPUT_CAP == 8000
