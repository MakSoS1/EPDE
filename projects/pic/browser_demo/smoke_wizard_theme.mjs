/* Regression: real three-page research wizard, disabled until confirmation,
 * backwards navigation, editable saved fields, and light/dark appearance.
 */
import {chromium} from "playwright";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:980}});
const base=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
try{
  await page.goto(base,{waitUntil:"domcontentloaded",timeout:55000});
  let f=page;
  await f.getByRole("link",{name:/Run a search/i}).first().waitFor({timeout:65000});
  await f.getByText("Light",{exact:true}).first().waitFor({timeout:30000});
  await f.getByText("Dark",{exact:true}).first().click();
  await page.waitForTimeout(900);
  const dark=await f.evaluate(()=>getComputedStyle(document.querySelector(".stApp")).backgroundColor);
  if(!/rgb\(16, 26, 41\)/.test(dark))throw Error("Dark theme not applied to app: "+dark);
  console.log("DARK_THEME_SWITCH_PASS "+dark);
  await f.getByText("Light",{exact:true}).first().click();
  await page.waitForFunction(
    ()=>getComputedStyle(document.querySelector(".stApp")).backgroundColor==="rgb(250, 252, 255)",
    null,{timeout:15000});
  const light=await f.evaluate(()=>getComputedStyle(document.querySelector(".stApp")).backgroundColor);
  console.log("LIGHT_THEME_SWITCH_PASS "+light);

  await f.getByRole("link",{name:/Run a search/i}).first().click();
  await f.getByRole("heading",{name:"Run a search"}).waitFor({timeout:45000});
  await f.getByText("Governing equation",{exact:true}).waitFor({timeout:45000});
  await f.getByText("Data preview",{exact:true}).waitFor({timeout:45000});
  if(await f.getByText("Configure method settings",{exact:true}).count())
    throw Error("Step two is already visible in step one");
  await page.screenshot({path:"wizard-step-1.png",fullPage:true});
  await f.getByRole("button",{name:/Next: configure methods/i}).click();
  await f.getByText("Configure method settings",{exact:true}).waitFor({timeout:45000});
  const pop=f.getByRole("spinbutton",{name:/Population \/ candidates/i}).first();
  await pop.fill("8");await pop.press("Tab");
  await page.screenshot({path:"wizard-step-2.png",fullPage:true});
  await f.getByRole("button",{name:/Next: review configuration/i}).click();
  await f.getByText("Complete effective scientific configuration",{exact:true})
    .waitFor({timeout:45000});
  if(await f.getByText("Configure method settings",{exact:true}).count())
    throw Error("Settings panel still visible on review screen");
  let run=f.getByRole("button",{name:/Start search: EPDE.*PySINDy/i}).first();
  await run.waitFor({timeout:30000});
  if(!(await run.isDisabled()))throw Error("Run became active without confirmation");
  await page.screenshot({path:"wizard-step-3.png",fullPage:true});
  console.log("WIZARD_REVIEW_CONFIRMATION_REQUIRED_PASS");
  await f.getByRole("button",{name:/Back to settings/i}).click();
  await f.getByText("Configure method settings",{exact:true}).waitFor({timeout:35000});
  const restored=f.getByRole("spinbutton",{name:/Population \/ candidates/i}).first();
  if(await restored.inputValue()!=="8")
    throw Error("EPDE numeric setting lost on back navigation: "+await restored.inputValue());
  await f.getByRole("button",{name:/Back to data/i}).click();
  await f.getByText("Governing equation",{exact:true}).waitFor({timeout:35000});
  await f.getByText("Data preview",{exact:true}).waitFor({timeout:35000});
  console.log("WIZARD_THREE_SCREENS_AND_BACK_NAVIGATION_PASS");
}catch(e){
  console.error("WIZARD_THEME_FAILURE "+e.message);
  console.error("LAST_VIEW "+(await page.locator("body").innerText()).slice(-4400));
  throw e;
}finally{await browser.close();}
