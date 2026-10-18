"""TaskFlow label space: categories, teams, priorities, and the subtopics tickets are generated from.

Each subtopic fixes the owning team, so team is learnable from the text rather than random noise.
"""

from dataclasses import dataclass

CATEGORIES = ["login_auth", "billing", "bug", "feature_request", "performance", "data_export"]
TEAMS = ["Platform", "Payments", "Product", "Support"]
PRIORITIES = ["P1", "P2", "P3", "P4"]

PRIORITY_MEANING = {
    "P1": "outage or data loss; blocks many users or the whole company; customer is urgent",
    "P2": "major problem blocking this customer's work, no workaround",
    "P3": "normal issue or question with a workaround",
    "P4": "low urgency: minor annoyance, how-to question or wish",
}


@dataclass(frozen=True)
class Subtopic:
    key: str
    category: str
    team: str
    description: str
    priorities: dict[str, float]  # sampling weights
    incident_prone: bool = False  # can plausibly hit many customers at once


SUBTOPICS = [
    # login_auth
    Subtopic("sso_failure", "login_auth", "Platform", "SSO / SAML sign-in (Okta, Google, Azure AD) fails or loops",
             {"P1": 0.3, "P2": 0.5, "P3": 0.2}, incident_prone=True),
    Subtopic("password_reset", "login_auth", "Support", "password reset email never arrives or link is invalid",
             {"P2": 0.3, "P3": 0.6, "P4": 0.1}),
    Subtopic("two_factor", "login_auth", "Platform", "two-factor codes rejected, or user lost their 2FA device",
             {"P2": 0.6, "P3": 0.4}),
    Subtopic("session_logout", "login_auth", "Platform", "user keeps getting logged out / session expires too fast",
             {"P2": 0.3, "P3": 0.7}),
    Subtopic("invite_accept", "login_auth", "Support", "invited teammate cannot accept invite or join workspace",
             {"P3": 0.7, "P4": 0.3}),
    # billing
    Subtopic("double_charge", "billing", "Payments", "charged twice or charged an unexpected amount",
             {"P1": 0.2, "P2": 0.6, "P3": 0.2}, incident_prone=True),
    Subtopic("card_declined", "billing", "Payments", "card declined when upgrading or renewing a plan",
             {"P2": 0.5, "P3": 0.5}),
    Subtopic("invoice_request", "billing", "Payments", "needs an invoice, receipt, or VAT/company details on invoice",
             {"P3": 0.4, "P4": 0.6}),
    Subtopic("refund_cancel", "billing", "Payments", "wants a refund or to cancel the subscription",
             {"P3": 0.7, "P4": 0.3}),
    Subtopic("seat_limits", "billing", "Support", "confused about seat count, plan limits or what a plan includes",
             {"P3": 0.5, "P4": 0.5}),
    # bug
    Subtopic("upload_fails", "bug", "Product", "file upload to a task hangs, fails or the file is corrupted",
             {"P2": 0.5, "P3": 0.5}, incident_prone=True),
    Subtopic("board_dragdrop", "bug", "Product", "Kanban board drag and drop of cards is broken",
             {"P2": 0.4, "P3": 0.6}),
    Subtopic("notifications_missing", "bug", "Platform", "email or in-app notifications are not being sent",
             {"P2": 0.5, "P3": 0.5}, incident_prone=True),
    Subtopic("timezone_dates", "bug", "Product", "task due dates or reminders show the wrong date or time zone",
             {"P3": 0.8, "P4": 0.2}),
    Subtopic("mobile_crash", "bug", "Product", "mobile app (iOS/Android) crashes or freezes",
             {"P2": 0.5, "P3": 0.5}, incident_prone=True),
    Subtopic("comments_vanish", "bug", "Product", "comments or task edits disappear after saving",
             {"P2": 0.6, "P3": 0.4}),
    # feature_request
    Subtopic("gantt_view", "feature_request", "Product", "asks for a Gantt / timeline view or dependencies",
             {"P3": 0.2, "P4": 0.8}),
    Subtopic("dark_mode", "feature_request", "Product", "asks for dark mode or UI customization",
             {"P4": 1.0}),
    Subtopic("integrations", "feature_request", "Product", "asks for an integration (Slack, GitHub, Jira, Teams)",
             {"P3": 0.3, "P4": 0.7}),
    Subtopic("recurring_tasks", "feature_request", "Product", "asks for recurring tasks or task templates",
             {"P4": 1.0}),
    Subtopic("custom_fields", "feature_request", "Product", "asks for custom fields, labels or richer filters",
             {"P3": 0.2, "P4": 0.8}),
    # performance
    Subtopic("dashboard_slow", "performance", "Platform", "dashboard or project pages load very slowly",
             {"P2": 0.4, "P3": 0.6}, incident_prone=True),
    Subtopic("search_slow", "performance", "Platform", "search is slow, times out or returns nothing",
             {"P2": 0.3, "P3": 0.7}),
    Subtopic("app_down", "performance", "Platform", "app is completely down / 500 or 502 errors everywhere",
             {"P1": 0.9, "P2": 0.1}, incident_prone=True),
    Subtopic("sync_delay", "performance", "Platform", "changes take minutes to sync between web, mobile, teammates",
             {"P2": 0.3, "P3": 0.7}),
    # data_export
    Subtopic("csv_export_broken", "data_export", "Product", "CSV/Excel export has missing columns or garbled characters",
             {"P2": 0.3, "P3": 0.7}),
    Subtopic("export_stuck", "data_export", "Platform", "export job never finishes or download link expired/broken",
             {"P2": 0.5, "P3": 0.5}, incident_prone=True),
    Subtopic("gdpr_export", "data_export", "Support", "needs a full account data export for GDPR / legal / offboarding",
             {"P2": 0.3, "P3": 0.7}),
    Subtopic("api_rate_limit", "data_export", "Platform", "hitting API rate limits or API errors when pulling data",
             {"P2": 0.4, "P3": 0.6}),
    Subtopic("import_fails", "data_export", "Support", "importing projects from Trello/Asana/CSV fails",
             {"P2": 0.3, "P3": 0.7}),
]

SUBTOPICS_BY_KEY = {s.key: s for s in SUBTOPICS}
