"""Conservative read-only selectors for FloCareer dashboard content."""

from __future__ import annotations


INTERVIEW_ROW_SELECTORS = (
    '[data-testid="interview-row"]',
    '[data-testid*="interview"][role="row"]',
    "table tbody tr",
    "mat-row",
    ".mat-row",
    ".interview-card",
    ".scheduled-interview",
    '[role="rowgroup"] [role="row"]',
)

LOGGED_OUT_TEXT = (
    "looks like you are logged out",
    "please log back again",
    "sign in",
    "log in",
)

LOADING_SELECTORS = (
    "ngx-skeleton-loader",
    ".skeleton-loader",
    '[class*="skeleton"]',
    '[aria-busy="true"]',
)
