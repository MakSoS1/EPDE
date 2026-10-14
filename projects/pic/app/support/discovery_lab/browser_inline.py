"""Self-contained, first-party PySINDy browser frontend for the Streamlit app.

Streamlit's st.iframe(src=<HTML>) accepts trusted HTML from the checked-in
repository. This avoids a third-party HTML-hosting/CDN dependency for the UI,
especially important for visitors whose local networks restrict GitHub CDNs.
The official Pyodide WASM and wheel downloads remain external. All numerical
computation stays in a browser Web Worker; Streamlit gets no dataset or result.
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "browser_demo"
CSS_LINK = '<link rel="stylesheet" href="./style.css">'
SCRIPT_LINK = '<script src="./app.js" defer></script>'
FETCH_LOADER = '''    const response = await fetch(new URL("./science.py", self.location.href));
    if (!response.ok) throw new Error("Cannot load science.py");
    const source = await response.text();'''


def _js_string(contents: str) -> str:
    # JSON-as-JS-string, resistant to an embedded </script> terminating the
    # enclosing HTML script; no user-supplied content is interpolated.
    return json.dumps(contents, ensure_ascii=True).replace("</", "<\\/")


def build_browser_html(*, embedded: bool = True) -> str:
    """Return static, trusted browser-only PySINDy HTML, without external iframe."""
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    css = (SOURCE / "style.css").read_text(encoding="utf-8")
    js = (SOURCE / "app.js").read_text(encoding="utf-8")
    worker = (SOURCE / "science_worker.js").read_text(encoding="utf-8")
    science = (SOURCE / "science.py").read_text(encoding="utf-8")
    if (html.count(CSS_LINK) != 1 or html.count(SCRIPT_LINK) != 1 or
            worker.count(FETCH_LOADER) != 1):
        raise RuntimeError("Browser scientific assets changed; update tested inline renderer")
    worker = worker.replace(FETCH_LOADER, "    const source = " + _js_string(science) + ";")
    # The new Worker is created from a Blob of this exact scientific script.
    # Pyodide and PySINDy are not reimplemented and fail explicitly if missing.
    inline_js = ("window.EPDE_WORKER_SOURCE = " + _js_string(worker) + ";\\n" + js)
    html = html.replace(CSS_LINK, "<style>\\n" + css + "\\n</style>")
    html = html.replace(SCRIPT_LINK, "<script>\\n" + inline_js + "\\n</script>")
    if embedded:
        html = html.replace("<body>", '<body class="embedded">')
        # The browser script also sets this flag for external ?embed=1 mode.
        # This mode is intentionally PySINDy only; not educational EvoSparse.
        html = html.replace("window.EPDE_WORKER_SOURCE =", 
            "document.addEventListener('DOMContentLoaded', () => { "
            "document.getElementById('method-evo').checked = false; });\\n"
            "window.EPDE_WORKER_SOURCE =")
    return html


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_browser_html(), encoding="utf-8")
