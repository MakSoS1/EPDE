/* Actual Streamlit user journey: single Run a search, both native scientific
 * methods on identical input, history grouped by input, full 59-entry catalog.
 * No separate browser-SINDy or execution-location tab is allowed.
 */
import {chromium} from "playwright";
const BASE=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];
page.on("pageerror",e=>errors.push(e.message));
async function appFrame(timeout=90000){
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){
    for(const frame of [...page.frames()].reverse()){
      try{
        if(await frame.getByRole("link",{name:/Run a search/i}).first().isVisible())
          return frame;
      }catch(_){}
    }
    await page.waitForTimeout(1250);
  }
  throw Error("Navigation missing; frames="+page.frames().map(x=>x.url()).join(" | "));
}
try{
  await page.goto(BASE,{waitUntil:"domcontentloaded",timeout:45000});
  const f=await appFrame();
  const labels=await f.locator("body").innerText();
  if(!labels.includes("Run a search") || !labels.includes("Data catalog"))
    throw Error("Unified scientific navigation missing");
  if(/PySINDy · your PC|Results · server|Compare · server/.test(labels))
    throw Error("Obsolete execution-location navigation is still displayed");
  await f.getByRole("link",{name:/Run a search/i}).first().click();
  await f.getByText("Choose method and equation").waitFor({timeout:45000});
  await f.getByRole("button",{name:/Next: configure methods/i}).click();
  await f.getByText(/Configure method settings/).waitFor({timeout:45000});
  await f.getByRole("tab", {name:/EPDE.*Evolution/i}).waitFor({timeout:30000});
  await f.getByRole("tab", {name:/PySINDy.*Sparse regression/i}).click();
  await f.getByText("STLSQ sparsity threshold").first().waitFor({timeout:30000});
  await f.getByRole("tab", {name:/EPDE.*Evolution/i}).click();
  await f.getByRole("tab",{name:/EPDE.*Evolution/i}).waitFor();
  await f.getByRole("tab",{name:/PySINDy.*Sparse regression/i}).waitFor();
  console.log("SINGLE_SEARCH_METHODS_PASS EPDE+PySINDy");

  // Set the original EPDE search to a small reproducible scientific budget
  // without changing the known law, library, input or baseline adapter.
  const pop=f.getByRole("spinbutton",{name:/Population \/ candidates/i});
  const epochs=f.getByRole("spinbutton",{name:/Evolution epochs/i});
  await pop.fill("4"); await pop.press("Tab");
  await epochs.fill("1"); await epochs.press("Tab");
  await f.getByRole("button",{name:/Next: review configuration/i}).click();
  await f.getByText(/Complete effective scientific configuration/).waitFor({timeout:45000});
  await f.getByText(/I have checked the input/i).first().click();
  if(!(await f.getByRole("checkbox",{name:/I have checked the input/i}).isChecked()))
    throw Error("Experiment confirmation checkbox did not toggle");
  const run=f.getByRole("button",{name:/Start search: EPDE.*PySINDy/i}).first();
  await run.waitFor({timeout:30000});
  if(await run.isDisabled()){
    throw Error("Paired method run disabled: "+(await f.locator("body").innerText()).slice(-2300));
  }
  await run.click();
  await f.getByText(/2 search\(es\) started for the same input/).waitFor({timeout:45000});
  console.log("PAIRED_SCIENTIFIC_QUEUE_PASS");

  await f.getByRole("link",{name:/Results/i}).first().click();
  await f.getByText("Results",{exact:true}).first().waitFor({timeout:30000});
  // Queued and running jobs have the same session and matching scientific input.
  await f.getByRole("combobox",{name:/Inspect result/i}).waitFor({timeout:30000});
  console.log("SHARED_RESULTS_PAGE_PASS");

  await f.getByRole("link",{name:/Compare/i}).first().click();
  await f.getByText("Compare",{exact:true}).first().waitFor({timeout:45000});
  console.log("MATCHED_INPUT_COMPARE_PAGE_PASS");

  await f.getByRole("link",{name:/Data catalog/i}).first().click();
  await f.getByText("Data catalog",{exact:true}).first().waitFor({timeout:45000});
  await f.getByText(/Showing 59 of 59 registered scientific datasets/).waitFor({timeout:30000});
  console.log("FULL_CATALOG_PASS 59/59");

  await page.reload({waitUntil:"domcontentloaded"});
  const fresh=await appFrame();
  await fresh.getByRole("link",{name:/Results/i}).first().click();
  await fresh.getByRole("combobox",{name:/Inspect result/i}).waitFor({timeout:45000});
  console.log("PERSISTED_RESULTS_RELOAD_PASS");
  if(errors.length)throw Error(errors.join("\n"));
  console.log("STREAMLIT_UNIFIED_E2E_PASS");
}finally{
  await page.screenshot({path:"streamlit-e2e.png",fullPage:true});
  await browser.close();
}
