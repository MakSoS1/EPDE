/* Test the ACTUAL Streamlit page with browser-only PySINDy embedded in st.iframe.
 * This is not a static-HTML smoke: the real Streamlit websocket/navigation
 * creates the iframe and the visitor's browser must compute the equation.
 */
import { chromium } from "playwright";

const BASE = process.env.STREAMLIT_URL || "http://127.0.0.1:8501";
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on("pageerror", err => errors.push(err.message));
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
  if (!pageText?.includes("Demo server") || !pageText.includes("Scientific settings")) {
    throw new Error("Original EPDE lab or server-execution selector missing");
  }
  console.log("STREAMLIT_INTEGRATION_EPDE_LAB_PASS");
  if (errors.length) throw new Error("Browser page exceptions: "+errors.join(" | "));
  console.log("STREAMLIT_SITE_E2E_PASS");
} finally {
  await page.screenshot({path:"streamlit-e2e.png",fullPage:true});
  await browser.close();
}
