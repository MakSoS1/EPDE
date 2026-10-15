/* An actual native EPDE+PySINDy result ZIP -> Streamlit Results -> Compare.
 * The ZIP is produced by portable_experiment.py with the original solvers,
 * not a synthetic status-ok fixture.
 */
import {chromium} from "playwright";
import path from "node:path";
import fs from "node:fs";
const zip=path.resolve("../../../experiment-results.zip");
if(!fs.existsSync(zip))throw Error("Real paired native fixture missing: "+zip);
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:980}});
const base=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
async function frame(){
  const until=Date.now()+90000;
  while(Date.now()<until){
    for(const f of page.frames())try{
      if(await f.getByRole("link",{name:/Results/i}).first().isVisible())return f;
    }catch(_){}
    await page.waitForTimeout(950);
  }
  throw Error("Streamlit not navigable");
}
try{
  await page.goto(base,{waitUntil:"domcontentloaded",timeout:45000});
  let f=await frame();
  await f.getByRole("link",{name:/Results/i}).first().click();
  await f.getByText(/Import local results/i).first().waitFor({timeout:45000});
  await f.locator('input[type="file"]').setInputFiles(zip);
  await f.getByRole("button",{name:/Import these results/i}).click();
  await f.getByText(/Imported 2 scientifically identified result/i)
    .waitFor({timeout:45000});
  console.log("PORTABLE_REAL_NATIVE_ZIP_UI_IMPORT_PASS");
  await f.getByRole("link",{name:/Compare/i}).first().click();
  await f.getByText("Input group").first().waitFor({timeout:40000});
  await f.getByText(/Input fingerprint:/i).first().waitFor({timeout:30000});
  await f.getByRole("tab",{name:/Equations/i}).first().waitFor();
  const body=await f.locator("body").innerText();
  if(!/EPDE/.test(body)||!/PySINDy/.test(body)||!/Experiments\s*2/.test(body))
    throw Error("Imported native result pair not visible together: "+body.slice(-2700));
  await f.getByRole("tab",{name:/Metrics/i}).click();
  await f.getByText(/internal scores are not interchangeable/i)
    .waitFor({timeout:35000});
  await page.screenshot({path:"ui-imported-native-compare.png",fullPage:true});
  console.log("PORTABLE_NATIVE_INPUT_GROUP_COMPARE_PASS");
}catch(e){
  console.error("IMPORT_COMPARE_E2E_FAILURE: "+e.message);
  for(const f of page.frames())try{
    console.error("CURRENT_PAGE_EXCERPT "+(await f.locator("body").innerText()).slice(-4200))
  }catch(_){}
  throw e;
}finally{await browser.close();}
