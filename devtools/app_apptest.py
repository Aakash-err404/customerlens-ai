"""End-to-end test through Streamlit's own AppTest harness.

AppTest runs app.py in a real script-run context, so the sidebar, the cache
decorators, st.rerun() and widget callbacks are all exercised the way a browser
would exercise them. Not part of the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check(condition: bool, message: str) -> None:
    print(("  [ok]   " if condition else "  [FAIL] ") + message)
    if not condition:
        raise AssertionError(message)


def state(at: AppTest, key: str) -> object:
    """AppTest's session_state has no .get(); membership test instead."""
    return at.session_state[key] if key in at.session_state else None


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()

    check(not at.exception, f"cold start renders without exception ({at.exception})")
    check(len(at.sidebar.radio) == 2, "sidebar has navigation and dataset-source radios")
    pages = at.sidebar.radio[0].options
    check(list(pages) == ["Results", "Final Summary"], f"pages discovered: {list(pages)}")
    check(
        state(at, "analysis") is None,
        "no dataset loaded before the user provides one",
    )

    # --- load a bundled example from the sidebar -------------------------
    at.sidebar.radio[1].set_value("Bundled example").run()
    check(not at.exception, f"example picker renders ({at.exception})")

    examples = at.selectbox[0].options
    print(f"  examples offered: {examples}")
    check(len(examples) >= 1, "at least one bundled example is offered")

    at.selectbox[0].set_value(examples[0]).run()
    loaders = [b for b in at.button if b.label == "Load dataset"]
    check(len(loaders) == 1, "load button present")
    loaders[0].click().run()

    check(not at.exception, f"dataset load + rerun succeeds ({at.exception})")
    check(state(at, "analysis") is not None, "analysis stored in session state")
    check(state(at, "modelling") is not None, "models fitted automatically on load")
    print(f"  success banners: {[s.value for s in at.success]}")
    print(f"  metrics on Results: {[m.value for m in at.metric]}")

    # --- walk every page in the real runtime ---------------------------
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        check(not at.exception, f"page '{page}' renders ({at.exception})")
        if page == "Final Summary":
            body = "\n".join(m.value for m in at.markdown)
            check(
                "Final summary" in body and "What was analysed" in body,
                "the final summary is rendered on its page",
            )
            print(f"  summary headings rendered: {body.count('###')}")

    # --- error path: nothing loaded ------------------------------------
    resets = [b for b in at.button if b.label == "Reset session"]
    check(len(resets) == 1, "reset button present")
    resets[0].click().run()
    check(not at.exception, f"reset renders ({at.exception})")
    check(state(at, "analysis") is None, "reset clears the loaded dataset")

    # --- empty upload path, on a fresh session --------------------------
    # A new AppTest is used here on purpose: resetting the session calls
    # st.cache_data.clear(), after which the old element tree can no longer
    # serialise its widget state.
    fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    fresh.run()
    fresh.sidebar.radio[1].set_value("Upload a file").run()
    check(not fresh.exception, f"empty upload state renders ({fresh.exception})")
    check(
        len([c for c in fresh.sidebar.caption if "Choose a CSV file" in c.value]) == 1,
        "prompt shown when no file has been chosen",
    )

    print("=" * 70)
    print("AppTest: both pages render in a real Streamlit runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
