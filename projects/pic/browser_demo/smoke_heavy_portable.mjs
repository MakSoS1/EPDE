/* User journey: heavy PDE config in the public browser -> downloadable native job.
 * Heavy arrays must never be loaded on the Streamlit server.
 */
import {chromium} from "playwright";

const base=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});

async function appFrame(timeout=90000){
  const until=Date.now()+timeout;
  while(Date.now()<until){
    for(const frame of [...page.frames()].reverse()){
      try{
        if(await frame.getByRole("link",{name:/Run a search/i}).first().isVisible())
          return frame;
      }catch(_){}
    }
    await page.waitForTimeout(750);
  }
  throw new Error("Streamlit navigation not found");
}
try{
  await page.goto(base,{waitUntil:"domcontentloaded",timeout:45000});
  let f=await appFrame();
  await f.getByRole("link",{name:/Data catalog/i}).first().click();
  const searchBox=f.getByRole("textbox",{name:/Search dataset or equation/i});
  await searchBox.fill("Navier");
  await searchBox.press("Enter"); // Streamlit commits text inputs on Enter/blur
  await f.getByText(/Navier.Stokes/i).first().waitFor({timeout:30000});
  console.log("CATALOG_PRESELECT", (await f.locator("body").innerText()).slice(-950));
  await f.getByRole("button",{name:/Use this record in Lab/i}).click();
  f=await appFrame();
  await f.getByRole("heading",{name:"Run a search"}).waitFor({timeout:30000});
  console.log("HEAVY_SELECTED_PAGE", (await f.locator("body").innerText()).slice(0,5100));
  await f.getByText(/requires native scientific computing/i).waitFor({timeout:45000});
  await f.getByText("Configure method settings").waitFor({timeout:30000});
  // The initial choice includes two methods; original PySINDy adapter must
  // say unsupported for Navier-Stokes instead of fabricating an equation.
  await f.getByText(/PySINDy.s current scientific adapter cannot represent/i)
    .waitFor({timeout:30000});
  await f.getByText("PySINDy (sparse regression)",{exact:true}).first().click();
  const pop=f.getByRole("spinbutton",{name:/Population \/ candidates/i}).first();
  await pop.fill("8");
  await pop.press("Tab");
  const downloadButton=f.getByRole("button",{name:/Download experiment.zip/i});
  await downloadButton.waitFor({timeout:30000});
  if(await downloadButton.isDisabled()) throw Error("Native configuration download disabled");
  const [download]=await Promise.all([
    page.waitForEvent("download",{timeout:45000}),
    downloadButton.click(),
  ]);
  if(download.suggestedFilename()!=="experiment.zip")
    throw Error("Wrong experiment file name "+download.suggestedFilename());
  if(await f.getByRole("button",{name:/Run EPDE/i}).count())
    throw Error("Heavy PDE unexpectedly exposed an in-browser run action");
  console.log("HEAVY_PDE_BROWSER_CONFIG_DOWNLOAD_PASS");

  await f.getByRole("link",{name:/Results/i}).first().click();
  f=await appFrame();
  const importPanel=f.getByText(/Import completed experiments/i).first();
  await importPanel.waitFor({timeout:30000});
  await importPanel.click();
  await f.getByText(/Import experiment-results.zip/i).first().waitFor();
  console.log("RESULTS_IMPORT_UI_PASS");
}catch(error){
  console.error("HEAVY_BROWSER_FAILURE:",error.message);
  for(const frame of page.frames()){
    try{console.error("PAGE_ON_FAILURE",frame.url(),(await frame.locator("body").innerText()).slice(0,8500))}catch(_){}
  }
  throw error;
}finally{
  await page.screenshot({path:"portable-heavy-e2e.png",fullPage:true});
  await browser.close();
}
