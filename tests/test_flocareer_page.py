from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from browser.flocareer_page import FloCareerPage
from browser.join_workflow import JoinWorkflowError, PostLaunchState
from browser.question_workflow import ExtractedQuestion


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


def test_join_candidate_menu_is_scoped_to_the_selected_card() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <table><tbody>
              <tr data-testid="interview-row">
                <td>Candidate Alpha</td><td>Engineer</td><td>Example Corp</td>
                <td>Today 11:00 AM</td>
                <td><button aria-label="More options" aria-controls="alpha-menu" onclick="window.opened='alpha'; showMenu('alpha-menu')">⋮</button></td>
              </tr>
              <tr data-testid="interview-row">
                <td>Candidate Beta</td><td>Engineer</td><td>Northwind Labs</td>
                <td>Today 4:00 PM</td>
                <td><button aria-label="More options" aria-controls="2253030" onclick="window.opened='beta'; showMenu('2253030')">⋮</button></td>
              </tr>
            </tbody></table>
            <div role="menu"><button>Launch Video Interview</button></div>
            <div id="alpha-menu" role="menu" hidden><button>Launch Video Interview</button></div>
            <div id="2253030" role="menu" hidden><button>Launch Video Interview</button></div>
            <script>
              function showMenu(id) {
                document.getElementById(id).hidden = false;
              }
            </script>
            """
        )
        flocareer = FloCareerPage(page)

        candidates = flocareer.list_join_candidates()
        flocareer.open_candidate_menu(candidates[1])

        assert page.evaluate("window.opened") == "beta"
        assert flocareer.visible_launch_control_count() == 1
        browser.close()


def test_join_candidates_are_bound_only_from_scheduled_parser_results() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <article><div>Help Center</div><button>⋮</button></article>
            <section>
              <h2>SCHEDULED INTERVIEWS (1)</h2>
              <article>
                <div>Candidate Alpha</div><div>ML Engineer</div>
                <div>Example Corp</div><div>Jul 11, 2026</div>
                <div>at 11:00 AM (IST)</div><button aria-label="More options">⋮</button>
              </article>
            </section>
            <section><h2>PENDING ACTIONS</h2></section>
            """
        )

        candidates = FloCareerPage(page).list_join_candidates()

        assert [candidate.candidate_name for candidate in candidates] == [
            "Candidate Alpha"
        ]
        browser.close()


def test_material_ui_today_cards_are_parsed_and_bound_to_their_own_menu() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1><h2>SCHEDULED<br>INTERVIEWS (2)</h2>
            <div>
              <div>
                <div><div class="jss44">Candidate Alpha</div></div>
                <div>ML Engineer</div><div>Example Corp</div>
                <div>TODAY</div><div>at 12:00 PM (IST)</div><div>It's Time</div>
                <button aria-label="more" onclick="window.opened='alpha'">⋮</button>
              </div>
              <div>
                <div><div class="jss44">Candidate Beta</div></div>
                <div>Data Analyst</div><div>Northwind Labs</div>
                <div>TODAY</div><div>at 4:00 PM (IST)</div>
                <button aria-label="more" onclick="window.opened='beta'">⋮</button>
              </div>
            </div>
            <h2>PENDING ACTIONS</h2>
            """
        )
        flocareer = FloCareerPage(page)

        interviews = flocareer.scan_scheduled_interviews()
        candidates = flocareer.list_join_candidates()

        assert [item.candidate_name for item in interviews] == [
            "Candidate Alpha",
            "Candidate Beta",
        ]
        assert [item.candidate_name for item in candidates] == [
            "Candidate Alpha",
            "Candidate Beta",
        ]
        browser.close()


def test_approved_launch_reaches_pre_call_and_approved_join_enters_interview() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <table><tbody><tr data-testid="interview-row">
              <td>Candidate Alpha</td><td>Engineer</td><td>Example Corp</td>
              <td>Tomorrow 11:00 AM</td>
              <td><button aria-label="more" aria-controls="991100"
                onclick="document.getElementById('991100').hidden=false">⋮</button></td>
            </tr></tbody></table>
            <div id="991100" role="menu" hidden>
              <button onclick="showConsent()">Launch Video Interview</button>
            </div>
            <script>
              function showConsent() {
                document.body.innerHTML = `
                  <button>OK</button>
                  <div role="dialog">
                    <h2>Interviewer Consent Form</h2>
                    <p>By clicking OK you acknowledge the instructions.</p>
                    <button onclick="showPreCall()">OK</button>
                  </div>`;
              }
              function showPreCall() {
                document.body.innerHTML = `
                  <h1>Joining as Fictional Interviewer</h1>
                  <button onclick="showInterview()">Join</button>`;
              }
              function showInterview() {
                document.body.innerHTML = `
                  <h1>Interview room</h1>
                  <button aria-label="Hang up">End</button>`;
              }
            </script>
            """
        )
        flocareer = FloCareerPage(page)
        candidate = flocareer.list_join_candidates()[0]
        flocareer.open_candidate_menu(candidate)
        unrelated_page = context.new_page()
        unrelated_page.set_content("<h1>Community chat</h1><button>Join</button>")

        flocareer.click_launch_interview()
        flocareer.wait_for_consent_form()

        assert flocareer.visible_consent_ok_count() == 1
        flocareer.click_consent_ok()
        flocareer.wait_for_pre_call_page()

        assert flocareer.visible_join_control_count() == 1
        flocareer.click_join()
        flocareer.wait_for_joined_interview()
        assert page.get_by_role("heading", name="Interview room").is_visible()
        browser.close()


def test_join_button_disappearing_on_an_error_page_is_not_joined() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<h1>Unable to join interview</h1>")
        flocareer = FloCareerPage(page)

        with pytest.raises(JoinWorkflowError, match="room-ready marker"):
            flocareer.wait_for_joined_interview(timeout_seconds=0.05)

        browser.close()


def test_join_click_revalidates_pre_call_after_operator_pause() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <h1>Dashboard</h1>
            <table><tbody><tr data-testid="interview-row">
              <td>Candidate Alpha</td><td>Engineer</td><td>Example Corp</td>
              <td>Tomorrow 11:00 AM</td>
              <td><button aria-label="more" aria-controls="991101"
                onclick="document.getElementById('991101').hidden=false">⋮</button></td>
            </tr></tbody></table>
            <div id="991101" role="menu" hidden>
              <button onclick="document.body.innerHTML='<h1>Joining as AS123</h1><button>Join</button>'">
                Launch Video Interview
              </button>
            </div>
            """
        )
        flocareer = FloCareerPage(page)
        candidate = flocareer.list_join_candidates()[0]
        flocareer.open_candidate_menu(candidate)
        flocareer.click_launch_interview()

        state = flocareer.wait_for_consent_or_pre_call()
        assert state is PostLaunchState.PRE_CALL

        page.set_content("<h1>Community chat</h1><button>Join</button>")
        with pytest.raises(JoinWorkflowError, match="no longer verified"):
            flocareer.click_join()


def test_extract_questions_expands_text_and_only_detects_code_editor() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-1" class="clMainSingleFESug"
              style="width:700px; min-height:220px"><span>1</span>
              <button name="title" onclick="this.nextElementSibling.hidden=false">
                Explain model drift affecting a production LLM...
              </button>
              <div class="MuiCollapse-root" hidden>
                <div class="clFESingleSugDet">
                  <p>Explain model drift affecting a production LLM.</p>
                  <p>Include monitoring, diagnosis, and remediation steps.</p>
                </div>
                <div>====================</div>
                <div>Ideal Answer (5 Star)</div><p>Monitor, diagnose, and roll back.</p>
                <div>Guidelines for 4 star rating</div><p>Misses one detail.</p>
              </div>
              <button>Bookmark in Video</button><button>Mark as</button>
              <textarea placeholder="Feedback *"></textarea><div>YOUR RATING</div>
            </section>
            <section id="container-normal-2" class="clMainSingleFESug"
              style="width:700px; min-height:220px"><span>2</span>
              <button name="title">Implement an LRU cache.</button>
              <div role="tab">Question</div><div role="tab">Code Editor</div>
              <label><input type="checkbox">SHOW CODE EDITOR TO CANDIDATE</label>
              <button>Bookmark in Video</button><button>Mark as</button>
              <textarea placeholder="Feedback *"></textarea><div>YOUR RATING</div>
            </section>
            """
        )
        flocareer = FloCareerPage(page)

        questions = flocareer.extract_questions()

        assert questions == [
            ExtractedQuestion(
                id=1,
                question_text=(
                    "Explain model drift affecting a production LLM.\n"
                    "Include monitoring, diagnosis, and remediation steps."
                ),
                has_code_editor=False,
                ideal_answer="Monitor, diagnose, and roll back.",
                guidelines={"4_star": "Misses one detail."},
                feedback_field_locator_hint="question:1:feedback",
                rating_locator_hint="question:1:rating",
                mark_as_locator_hint="question:1:mark_as",
            ),
            ExtractedQuestion(
                id=2,
                question_text="Implement an LRU cache.",
                has_code_editor=True,
                ideal_answer="",
                guidelines={},
                feedback_field_locator_hint="question:2:feedback",
                rating_locator_hint="question:2:rating",
                mark_as_locator_hint="question:2:mark_as",
            ),
        ]
        assert page.locator('input[type="checkbox"]').is_checked() is False
        browser.close()
