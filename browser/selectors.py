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

JOIN_CARD_SELECTORS = (
    '[data-testid="interview-row"]',
    '[data-testid*="interview"][role="row"]',
    "table tbody tr",
    "mat-row",
    ".mat-row",
    ".interview-card",
    ".scheduled-interview",
    "mat-card",
    "article",
)

CANDIDATE_MENU_BUTTON_SELECTORS = (
    'button[aria-label*="more" i]',
    'button[aria-label*="menu" i]',
    'button[aria-label*="option" i]',
    'button:has-text("⋮")',
    'button:has-text("more_vert")',
)

ACTIVE_MENU_SELECTORS = (
    '[role="menu"]',
    ".mat-mdc-menu-panel",
    ".mat-menu-panel",
    ".cdk-overlay-pane",
)

JOINED_INTERVIEW_SELECTORS = (
    'button[aria-label*="hang up" i]',
    'button[aria-label*="end call" i]',
    '[role="button"][aria-label*="hang up" i]',
    '[role="button"][aria-label*="end call" i]',
    '[title*="hang up" i]',
    '[title*="end call" i]',
    '[data-testid*="hangup" i]',
    '[data-testid="CallEndIcon"]',
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
