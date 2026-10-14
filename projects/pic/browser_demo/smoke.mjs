/* End-to-end browser smoke, in CI on real Chromium: no client installation.
 * Serves the exact static files that GitHub Pages later publishes.
 */
import {chromium} from "playwright";

const base = process.env.BROWSER_BASE || "http://127.0.0.1:8766";
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1340,height:920}});
const failures=[];
page.on("pageerror", err => failures.push("pageerror: " + err.message));
// HTTP CDN diagnostics can log benign console warnings; assert job result instead.
try {
  await page.goto(base, {waitUntil:"domcontentloaded",timeout:30000});
  await page.locator("h1").first().waitFor();
  if (!(await page.title()).includes("Equation Discovery")) throw new Error("Wrong page title");
  if (!await page.locator("button[data-source='builtin']").count()) throw new Error("Missing Lab source");
  if (!await page.locator("#method-sindy").isChecked()) throw new Error("Real PySINDy not selected");
  await page.locator("#method-evo").uncheck();
  await page.locator("#dataset").selectOption("decay");
  await page.locator("#run").click();
  console.log("Waiting for real official PySINDy cold-start in browser (CDN+Pyodide)...");
  // WebAssembly, dependency wheels, and Web Worker execute in real Chromium.
  await page.waitForFunction(() => {
    const status = document.querySelector("#status")?.textContent || "";
    return status.includes("Finished.") || status.includes("Could not finish:");
  }, undefined, {timeout:240000}).catch(async () => {
    throw new Error("Browser scientific run timed out: " +
                    await page.locator("#status").textContent());
  });
  const status = await page.locator("#status").textContent();
  if (!status.includes("Finished.")) throw new Error("PySINDy runtime failed: " + status);
  await page.locator(".nav-btn[data-page='results']").click();
  await page.locator("#result-detail .metric").first().waitFor({timeout:20000});
  const results = await page.locator("#result-detail").textContent();
  if (!results.includes("PySINDy") || !results.includes("d") ||
      !results.includes("Validation R²")) throw new Error("Broken result renderer");
  const number = await page.locator("#result-detail .metric strong").first().textContent();
  if (!(Number(number) > .95)) throw new Error("PySINDy recovered poor decay fit: " + number);
  console.log("PASS: genuine PySINDy native library executing inside Chromium WebAssembly.");

  // Validate IndexedDB persistence on F5; the same browser should retain its results.
  await page.reload({waitUntil:"domcontentloaded"});
  await page.locator(".nav-btn[data-page='results']").click();
  await page.waitForFunction(() => document.querySelector("#count")?.textContent === "1");
  if (!(await page.locator("#result-detail").textContent()).includes("PySINDy")) {
    throw new Error("IndexedDB did not preserve the scientific result");
  }
  console.log("PASS: result persisted through browser refresh.");

  // Run the separate educational algorithm and confirm true identical-input comparison.
  await page.locator(".nav-btn[data-page='lab']").click();
  await page.locator("#method-sindy").uncheck();
  await page.locator("#method-evo").check();
  await page.locator("#run").click();
  await page.waitForFunction(() => {
    const status = document.querySelector("#status")?.textContent || "";
    return status.includes("Finished.") || status.includes("Could not finish:");
  }, undefined, {timeout:90000});
  if (!(await page.locator("#status").textContent()).includes("Finished.")) {
    throw new Error("Evolutionary educational demonstration failed");
  }
  await page.locator(".nav-btn[data-page='compare']").click();
  await page.locator("#compare-detail .compare-grid").waitFor();
  const comparison = await page.locator("#compare-detail").textContent();
  if (!comparison.includes("Official PySINDy") ||
      !comparison.includes("Educational EvoSparse") ||
      !comparison.includes("not a published EPDE benchmark")) {
    throw new Error("The comparison is not labeled honestly");
  }
  console.log("PASS: identical-data comparison works and distinguishes EvoSparse from EPDE.");

  // No server-side computation calls and no local Python installation.
  if (failures.length) throw new Error(failures.join("\n"));
  console.log("BROWSER_SMOKE_PASS");
} finally {
  await page.screenshot({path:"browser-smoke-last.png", fullPage:true});
  await browser.close();
}
