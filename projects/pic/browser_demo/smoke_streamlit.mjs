/* Test the ACTUAL Streamlit page with browser-only PySINDy embedded in st.iframe.
 * This is not a static-HTML smoke: the real Streamlit websocket/navigation
 * creates the iframe and the visitor's browser must compute the equation.
 */
import { chromium } from "playwright";

const BASE = process.env.STREAMLIT_URL || "http://127.0.0.1:8501";
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
const failedRequests = [];
const httpErrors = [];
const consoleErrors = [];
page.on("pageerror", err => errors.push(err.message));
page.on("requestfailed", req => failedRequests.push([req.url(),req.failure()?.errorText]));
page.on("response", response => {
  if(response.status() >= 400) httpErrors.push([response.status(),response.url()]);
});
page.on("console", message => {
  if(message.type() === "error") consoleErrors.push(message.text());
});
try {
  await page.goto(BASE, {waitUntil:"domcontentloaded", timeout:45000});
  const sindyNav = page.getByRole("link", {name:/PySINDy.*your PC/i}).first();
  try {
    await sindyNav.waitFor({timeout:60000});
  } catch (err) {
    console.error("LIVE_SITE_DIAG url="+page.url()+" title="+await page.title());
    console.error("LIVE_SITE_BODY "+(await page.locator("body").innerText()).slice(0,4500));
    const controls=await page.getByRole("button").allTextContents();
    console.error("LIVE_SITE_BUTTONS "+JSON.stringify(controls.slice(0,30)));
    console.error("LIVE_SITE_REQUEST_FAILURES "+JSON.stringify(failedRequests.slice(-40)));
    console.error("LIVE_SITE_HTTP_ERRORS "+JSON.stringify(httpErrors.slice(-40)));
    console.error("LIVE_SITE_JS_EXCEPTIONS "+JSON.stringify(errors.slice(-30)));
    console.error("LIVE_SITE_CONSOLE_ERRORS "+JSON.stringify(consoleErrors.slice(-35)));
    console.error("LIVE_SITE_DOM "+JSON.stringify(await page.evaluate(() => ({
      ready:document.readyState,
      root:document.querySelector("#root")?.innerHTML?.slice(0,600),
      scripts:[...document.scripts].map(x=>x.src||"inline").slice(0,20),
      frames:document.querySelectorAll("iframe").length,
    }))));
    throw err;
  }
  await sindyNav.click();
  const iframe = page.frameLocator("iframe[title*='PySINDy 2.1.0']");
  await iframe.locator("#run").waitFor({timeout:45000});
  if (await iframe.locator("#method-evo").isChecked()) {
    throw new Error("The educational EvoSparse search is selected in the user-facing PySINDy tab");
  }
  await iframe.locator("#dataset").selectOption("decay");
  await iframe.locator("#run").click();
  await iframe.locator("#status").waitFor({timeout:20000});
  await page.waitForFunction(() => {
    const frame = [...document.querySelectorAll("iframe")]
      .find(f => (f.title || "").includes("PySINDy 2.1.0"));
    const message = frame?.contentDocument?.querySelector("#status")?.textContent || "";
    return message.includes("Finished.") || message.includes("Could not finish:");
  }, undefined, {timeout:270000});
  const message = await iframe.locator("#status").textContent();
  if (!message?.includes("Finished.")) throw new Error("Browser PySINDy failed inside Streamlit: "+message);
  await iframe.locator(".nav-btn[data-page='results']").click();
  const r2 = Number(await iframe.locator("#result-detail .metric strong").first().textContent());
  if (!(r2 > 0.95)) throw new Error("PySINDy returned poor validation R²: "+r2);
  console.log("STREAMLIT_INTEGRATION_SINDY_PASS R2="+r2);

  // The Streamlit server must not own or need this result. Browser IndexedDB
  // should survive a page refresh in the same origin.
  await page.reload({waitUntil:"domcontentloaded"});
  const newIframe = page.frameLocator("iframe[title*='PySINDy 2.1.0']");
  await newIframe.locator("#count").waitFor({timeout:45000});
  await page.waitForFunction(() => {
    const f = [...document.querySelectorAll("iframe")]
      .find(x => (x.title || "").includes("PySINDy 2.1.0"));
    return f?.contentDocument?.querySelector("#count")?.textContent === "1";
  },undefined,{timeout:30000});
  console.log("STREAMLIT_INTEGRATION_INDEXEDDB_PASS");

  const labNav = page.getByRole("link", {name:/Lab.*EPDE/i}).first();
  await labNav.click();
  await page.getByText("4 · Compute").waitFor({timeout:40000});
  const pageText = await page.locator("body").textContent();
  if (!pageText?.includes("Execution") || !pageText.includes("Scientific settings")
      || !(pageText.includes("Demo server") || pageText.includes("This app · local CPU"))) {
    throw new Error("Original EPDE lab or execution selector missing: "+
                    String(pageText).slice(0,1900));
  }
  console.log("STREAMLIT_INTEGRATION_EPDE_LAB_PASS");

  // Full catalogue is tested through actual Streamlit navigation, not just
  // Python mocks. The page must show EVERY registered ODE/PDE dataset without
  // reading full Navier–Stokes tensors on the shared server.
  await page.getByRole("link", {name:/All datasets/i}).first().click();
  await page.getByText("Registered datasets").first().waitFor({timeout:45000});
  await page.getByText(/Showing 59 of 59 registered scientific datasets/).waitFor({timeout:30000});
  console.log("STREAMLIT_COMPLETE_CATALOG_PASS 59/59");

  if (errors.length) throw new Error("Browser page exceptions: "+errors.join(" | "));
  console.log("STREAMLIT_SITE_E2E_PASS");
} finally {
  await page.screenshot({path:"streamlit-e2e.png",fullPage:true});
  await browser.close();
}
