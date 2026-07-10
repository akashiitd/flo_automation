from __future__ import annotations

from playwright.sync_api import sync_playwright

from browser.flocareer_page import FloCareerPage


def test_dashboard_scan_extracts_rows_without_clicking_actions() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <table>
              <thead>
                <tr><th>Candidate</th><th>Role</th><th>Company</th><th>Time</th></tr>
              </thead>
              <tbody>
                <tr data-testid="interview-row">
                  <td>Candidate Alpha</td><td>Platform Engineer</td>
                  <td>Example Corp</td><td>Today 11:00 AM</td>
                  <td><button onclick="window.actionClicked = true">Actions</button></td>
                </tr>
                <tr data-testid="interview-row">
                  <td>Candidate Beta</td><td>Data Scientist</td>
                  <td>Northwind Labs</td><td>Today 4:00 PM</td>
                  <td><button onclick="window.actionClicked = true">Actions</button></td>
                </tr>
              </tbody>
            </table>
            """
        )

        interviews = FloCareerPage(page).scan_scheduled_interviews()
        action_clicked = page.evaluate("Boolean(window.actionClicked)")
        browser.close()

    assert [interview.candidate_name for interview in interviews] == [
        "Candidate Alpha",
        "Candidate Beta",
    ]
    assert interviews[0].role == "Platform Engineer"
    assert interviews[0].company == "Example Corp"
    assert interviews[0].scheduled_time == "Today 11:00 AM"
    assert action_clicked is False


def test_dashboard_scan_extracts_visible_scheduled_interview_cards() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <section>
              <h2>SCHEDULED<br>INTERVIEWS (3)</h2>
              <label>Subscribe to WhatsApp<br>notification</label>
              <article>
                <div>Candidate Alpha</div>
                <div>ML Engineer</div><div>Example Corp</div>
                <div>Jul 11, 2026</div><div>at 11:00 AM (IST)</div>
                <div>(in 1 days)</div><button>⋮</button>
              </article>
              <article>
                <div>Candidate Beta</div>
                <div>Data Scientist</div><div>Northwind Labs</div>
                <div>Jul 11, 2026</div><div>at 4:00 PM (IST)</div>
                <div>(in 1 days)</div><button>⋮</button>
              </article>
              <article>
                <div>Candidate Gamma</div>
                <div>Backend Engineer</div>
                <div>Contoso Systems</div>
                <div>Jul 13, 2026</div><div>at 11:00 AM (IST)</div>
                <div>(in 3 days)</div><button>⋮</button>
              </article>
            </section>
            <section><h2>PENDING ACTIONS (3)</h2><div>Candidate Delta</div></section>
            """
        )

        interviews = FloCareerPage(page).scan_scheduled_interviews()
        browser.close()

    assert [interview.candidate_name for interview in interviews] == [
        "Candidate Alpha",
        "Candidate Beta",
        "Candidate Gamma",
    ]
    assert interviews[0].role == "ML Engineer"
    assert interviews[0].company == "Example Corp"
    assert interviews[0].scheduled_time == "Jul 11, 2026 at 11:00 AM (IST)"
    assert interviews[2].company == "Contoso Systems"


def test_initial_state_wait_detects_a_delayed_logged_out_dialog() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <script>
              setTimeout(() => {
                document.body.insertAdjacentHTML(
                  'beforeend',
                  '<div role="dialog">Looks like you are logged out. Please log back again.</div>'
                );
              }, 100);
            </script>
            """
        )

        state = FloCareerPage(page).wait_for_initial_state(
            timeout_seconds=1,
            settle_seconds=0.2,
        )
        browser.close()

    assert state == "login_required"


def test_dashboard_is_not_ready_while_skeleton_loaders_are_visible() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <ngx-skeleton-loader>
              <div class="skeleton-loader" style="width: 100px; height: 20px"></div>
            </ngx-skeleton-loader>
            """
        )

        ready = FloCareerPage(page).is_dashboard_ready()
        browser.close()

    assert ready is False


def test_dashboard_must_remain_ready_during_the_stability_window() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <script>
              setTimeout(() => {
                document.body.insertAdjacentHTML(
                  'beforeend',
                  '<div role="dialog">Looks like you are logged out. Please log back again.</div>'
                );
              }, 250);
            </script>
            """
        )

        stable = FloCareerPage(page).remains_dashboard_ready(duration_seconds=0.6)
        browser.close()

    assert stable is False
