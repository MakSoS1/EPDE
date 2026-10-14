"""Real Chromium verification of first-party, install-free PySINDy srcdoc.

The HTML is generated from the SAME checkout that Streamlit uses. No source is
fetched from raw.githack/Pages to initialize the browser component.
"""
import { chromium } from "playwright";

const base = process.env.BROWSER_INLINE_BASE || "http://127.0.0.1:8767";
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1150,height:900}});
const consoleErrors = [];
page.on("pageerror", e => consoleErrors.push(e.message));
try {
  await page.goto(base, {waitUntil:"domcontentloaded",timeout:30000});
  await page.locator("#run").waitFor({timeout:15000});
  if (!(await page.locator("body.embedded").count())) throw Error("Missing inline Streamlit mode");
  if (await page.locator("#method-evo").isChecked()) throw Error("Educational demo enabled by default");
  const source = await page.evaluate(() => typeof window.EPDE_WORKER_SOURCE === "string");
  if (!source) throw Error("Worker not bundled into HTML");
  if ((await page.locator("script[src*='raw.githack.com']").count()) > 0) {
    throw Error("External HTML source used inside Streamlit");
  }
  await page.locator("#dataset").selectOption("decay");
  await page.locator("#run").click();
  console.log("Loading Pyodide and official PySINDy from INLINE Streamlit HTML…");
  await page.waitForFunction(() => {
    const text = document.querySelector("#status")?.textContent || "";
    return text.includes("Finished.") || text.includes("Could not finish:");
  }, undefined, {timeout:270000});
  const status = await page.locator("#status").textContent();
  if (!status.includes("Finished.")) throw Error(status);
  await page.locator(".nav-btn[data-page='results']").click();
  const r2 = Number(await page.locator("#result-detail .metric strong").first().textContent());
  if (!(r2 > 0.95)) throw Error("Scientific regression failed R²>0.95: "+r2);
  await page.reload({waitUntil:"domcontentloaded"});
  await page.waitForFunction(() => document.querySelector("#count")?.textContent === "1");
  if (consoleErrors.length) throw Error(consoleErrors.join("\n"));
  console.log("INLINE_PYSINDY_PASS R2=" + r2 + " IndexedDB persists");
} finally {
  await page.screenshot({path:"browser-inline-smoke.png",fullPage:true});
  await browser.close();
}
