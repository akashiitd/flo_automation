from __future__ import annotations

import hashlib

import pytest
from playwright.sync_api import sync_playwright

from browser.code_editor_workflow import CodeEditorVisibility, CodeEditorWorkflowError
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
              <button role="tab">Question</button><button role="tab">Code Editor</button>
              <div role="tabpanel"><div>Editor placeholder</div></div>
              <label><input type="checkbox"><div class="clFloSwithTxt">
                SHOW CODE EDITOR TO CANDIDATE
              </div></label>
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


def test_question_detection_requires_semantic_code_editor_tab() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug" data-question-id="1">
              <span>1</span><button name="title">Explain a Python iterator.</button>
              <div>Code Editor</div>
              <button>Bookmark in Video</button><button>Mark as</button>
              <textarea></textarea><div>YOUR RATING</div>
            </section>
            """
        )

        questions = FloCareerPage(page).extract_questions()

        assert len(questions) == 1
        assert questions[0].has_code_editor is False
        browser.close()


def test_code_editor_dom_inspection_is_read_only_and_reports_unique_association() -> (
    None
):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-12" class="clMainSingleFESug">
              <span class="question-number">12</span>
              <button name="title">Explain generators.</button>
              <button role="tab">Question</button>
            </section>
            <section id="container-normal-13" class="clMainSingleFESug">
              <span class="question-number">13</span>
              <button name="title">Implement an LRU cache.</button>
              <button role="tab">Question</button>
              <button role="tab" aria-selected="false"
                onclick="window.editorTabClicked = true">Code Editor</button>
              <div role="tabpanel">
                <label class="editor-visibility-control">
                  <input type="checkbox" role="switch"
                    data-testid="candidate-editor-switch"
                    class="MuiSwitch-input"
                    value="private-candidate-value"
                    onclick="window.editorSwitchClicked = true">
                  <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
                </label>
              </div>
            </section>
            """
        )

        observations = FloCareerPage(page).inspect_code_editor_dom()

        assert len(observations) == 1
        observation = observations[0]
        assert observation.question_id == 13
        assert observation.question_id_source == "question-number-element"
        assert observation.question_id_candidates == (13,)
        assert observation.code_editor_tab_count == 1
        assert observation.rendered_code_editor_tab_count == 1
        assert observation.visibility_labels == ("SHOW CODE EDITOR TO CANDIDATE",)
        assert observation.visibility_label_rendered == (True,)
        assert observation.association_status == "unique"
        assert len(observation.switch_candidates) == 1
        assert observation.switch_candidates[0].tag_name == "input"
        assert observation.switch_candidates[0].role == "switch"
        assert observation.switch_candidates[0].test_id == ("candidate-editor-switch")
        assert observation.switch_candidates[0].rendered is True
        assert "editor-visibility-control" in (
            observation.control_wrapper_outer_html.html
            if observation.control_wrapper_outer_html
            else ""
        )
        assert (
            "container-normal-13" in observation.association_container_outer_html.html
        )
        assert (
            "Implement an LRU cache"
            not in observation.association_container_outer_html.html
        )
        assert "[redacted]" in observation.association_container_outer_html.html
        assert (
            "private-candidate-value"
            not in observation.association_container_outer_html.html
        )
        assert (
            observation.association_container_outer_html.sha256
            == hashlib.sha256(
                observation.association_container_outer_html.html.encode("utf-8")
            ).hexdigest()
        )
        assert page.evaluate("Boolean(window.editorTabClicked)") is False
        assert page.evaluate("Boolean(window.editorSwitchClicked)") is False
        assert page.locator('[role="switch"]').is_checked() is False
        browser.close()


def test_code_editor_dom_inspection_reports_ambiguous_switch_candidates() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug" data-question-id="7">
              <button role="tab">Code Editor</button>
              <label>
                <input type="checkbox" role="switch">
                <input type="checkbox" role="switch">
                <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
              </label>
            </section>
            """
        )

        observation = FloCareerPage(page).inspect_code_editor_dom()[0]

        assert observation.question_id == 7
        assert observation.association_status == "ambiguous"
        assert len(observation.switch_candidates) == 2
        assert not page.locator('[role="switch"]').nth(0).is_checked()
        assert not page.locator('[role="switch"]').nth(1).is_checked()
        browser.close()


def test_code_editor_dom_inspection_captures_hidden_mounted_controls() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug" data-question-id="9">
              <button role="tab">Code Editor</button>
              <div role="tabpanel" hidden>
                <label>
                  <input type="checkbox" role="switch">
                  <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
                </label>
              </div>
            </section>
            """
        )

        observation = FloCareerPage(page).inspect_code_editor_dom()[0]

        assert observation.association_status == "unique"
        assert observation.visibility_label_rendered == (False,)
        assert observation.switch_candidates[0].rendered is False
        browser.close()


def test_code_editor_dom_inspection_deduplicates_nested_question_roots() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-4" class="clMainSingleFESug"
              data-question-id="4">
              <div data-testid="question-card">
                <button role="tab">Code Editor</button>
                <label><input type="checkbox" role="switch">
                  <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
                </label>
              </div>
            </section>
            """
        )

        observations = FloCareerPage(page).inspect_code_editor_dom()

        assert len(observations) == 1
        assert observations[0].question_id == 4
        browser.close()


def test_code_editor_dom_inspection_keeps_separate_mixed_root_shapes() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-4" class="clMainSingleFESug"
              data-question-id="4">
              <button role="tab">Code Editor</button>
            </section>
            <article class="question-card" data-question-id="8">
              <button role="tab">Code Editor</button>
            </article>
            """
        )

        observations = FloCareerPage(page).inspect_code_editor_dom()

        assert [observation.question_id for observation in observations] == [4, 8]
        browser.close()


def test_code_editor_dom_inspection_does_not_guess_from_unscoped_numbers() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-dynamic" class="clMainSingleFESug">
              <span>13</span><span>5</span>
              <button role="tab">Code Editor</button>
              <label><input type="checkbox" role="switch">
                <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
              </label>
            </section>
            """
        )

        observation = FloCareerPage(page).inspect_code_editor_dom()[0]

        assert observation.question_id is None
        assert observation.question_id_source == "unresolved"
        assert observation.question_id_candidates == (5, 13)
        assert observation.association_status == "ambiguous"
        browser.close()


def test_code_editor_dom_structural_snapshot_is_size_bounded() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug" data-question-id="3">
              <button role="tab">Code Editor</button>
              <label><input type="checkbox" role="switch">
                <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
              </label>
            </section>
            """
        )
        page.locator(".clMainSingleFESug").evaluate(
            "(element) => element.className += ' ' + 'x'.repeat(60000)"
        )

        observation = FloCareerPage(page).inspect_code_editor_dom()[0]

        snapshot = observation.association_container_outer_html
        assert snapshot.truncated is True
        assert len(snapshot.html) == 50_000
        assert len(snapshot.sha256) == 64
        page.locator(".clMainSingleFESug").evaluate(
            "(element) => element.className += 'different-after-prefix'"
        )
        changed = FloCareerPage(page).inspect_code_editor_dom()[0]
        assert changed.association_container_outer_html.html == snapshot.html
        assert changed.association_container_outer_html.sha256 != snapshot.sha256
        browser.close()


def test_code_editor_actions_are_scoped_to_exact_question() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section id="container-normal-12" class="clMainSingleFESug">
              <span class="clSeqGreen">12</span>
              <button role="tab">Question</button>
              <button role="tab" aria-selected="false"
                onclick="this.setAttribute('aria-selected', 'true')">Code Editor</button>
              <div class="editor-switch">
                <input type="checkbox" role="switch" name="codeSwitch-4012"
                  onchange="this.nextElementSibling.textContent = this.checked
                    ? 'HIDE CODE EDITOR TO CANDIDATE'
                    : 'SHOW CODE EDITOR TO CANDIDATE'">
                <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
              </div>
            </section>
            <section id="container-normal-13" class="clMainSingleFESug">
              <span class="clSeqGreen">13</span>
              <button role="tab">Question</button>
              <button role="tab" aria-selected="false"
                onclick="this.setAttribute('aria-selected', 'true')">Code Editor</button>
              <div class="editor-switch">
                <input type="checkbox" role="switch" name="codeSwitch-4013"
                  onchange="this.nextElementSibling.textContent = this.checked
                    ? 'HIDE CODE EDITOR TO CANDIDATE'
                    : 'SHOW CODE EDITOR TO CANDIDATE'">
                <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
              </div>
            </section>
            """
        )
        flocareer = FloCareerPage(page)

        flocareer.open_code_editor_tab(13)
        before = flocareer.read_code_editor_visibility(13)
        assert before is CodeEditorVisibility.HIDDEN
        assert not page.locator("#container-normal-13 [role='switch']").is_checked()
        flocareer.click_show_code_editor(13)
        flocareer.wait_for_code_editor_visibility(
            13,
            CodeEditorVisibility.VISIBLE,
        )

        assert (
            page.locator("#container-normal-13 [role='tab']")
            .get_by_text("Code Editor", exact=True)
            .get_attribute("aria-selected")
            == "true"
        )
        assert page.locator("#container-normal-13 [role='switch']").is_checked()
        assert not page.locator("#container-normal-12 [role='switch']").is_checked()
        assert (
            page.locator("#container-normal-12 [role='tab']")
            .get_by_text("Code Editor", exact=True)
            .get_attribute("aria-selected")
            == "false"
        )
        browser.close()


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["SHOW CODE EDITOR TO CANDIDATE", "HIDE CODE EDITOR TO CANDIDATE"],
        ["UNKNOWN CODE EDITOR STATE"],
    ],
)
def test_code_editor_visibility_fails_closed_when_state_is_ambiguous(
    labels: list[str],
) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        label_html = "".join(
            f'<div class="clFloSwithTxt">{label}</div>' for label in labels
        )
        page.set_content(
            f"""
            <section class="clMainSingleFESug"><span class="question-number">13</span>
              <button role="tab">Code Editor</button>
              {label_html}
            </section>
            """
        )

        with pytest.raises(CodeEditorWorkflowError, match="visibility state"):
            FloCareerPage(page).read_code_editor_visibility(13)

        browser.close()


def test_code_editor_show_fails_closed_without_real_switch_contract() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug"><span class="question-number">13</span>
              <button role="tab">Code Editor</button>
              <button onclick="window.guessedClick = true">Unlabelled visual toggle</button>
              <div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>
            </section>
            """
        )

        with pytest.raises(CodeEditorWorkflowError, match="code-editor switch"):
            FloCareerPage(page).click_show_code_editor(13)

        assert page.evaluate("Boolean(window.guessedClick)") is False
        browser.close()


def test_question_action_identity_ignores_unrelated_numeric_content() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <section class="clMainSingleFESug">
              <span class="question-number">12</span>
              <p>Return exactly 13 records.</p><span>13</span>
              <button role="tab">Code Editor</button>
            </section>
            <section class="clMainSingleFESug">
              <span class="question-number">13</span>
              <button role="tab" aria-selected="false"
                onclick="this.setAttribute('aria-selected', 'true')">Code Editor</button>
            </section>
            """
        )

        FloCareerPage(page).open_code_editor_tab(13)

        assert (
            page.locator(".clMainSingleFESug")
            .filter(has=page.locator(".question-number", has_text="13"))
            .get_by_role("tab", name="Code Editor", exact=True)
            .count()
            == 1
        )
        browser.close()
