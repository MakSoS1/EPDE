/* Read-only public deployment probe. Avoid running scientific jobs on the
 * shared public Community Cloud deployment merely to test the UI.
 */
import {chromium} from "playwright";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:950}});
const url=process.env.STREAMLIT_URL||"https://epde-explorer-demo.streamlit.app/";
const errors=[];
page.on("pageerror",e=>errors.push(e.message));
async function ready(){
  const until=Date.now()+180000;
  while(Date.now()<until){
    for(const frame of [...page.frames()].reverse()){
      try {
        if(await frame.getByRole("link",{name:/Run a search/i}).first().isVisible())
          return frame;
      }catch(_){}
    }
    await page.waitForTimeout(1700);
  }
  throw Error("Published site did not show the six-page Streamlit navigation");
}
try {
  await page.goto(url,{waitUntil:"domcontentloaded",timeout:60000});
  const f=await ready();
  await f.getByRole("heading",{name:"EPDE Explorer"}).waitFor({timeout:25000});
  await f.getByRole("link",{name:/Run a search/i}).first().click();
  await f.getByText(/Choose method and equation/i).first().waitFor({timeout:60000});
  await f.getByRole("button",{name:/Next: configure methods/i}).waitFor({timeout:60000});
  // The next two stages must be separate screens, not a long initial form.
  if(await f.locator('[data-testid="stException"]').count())
    throw Error("Published search page displays a Python exception");
  await page.screenshot({path:"public-minimal-run-search.png",fullPage:true});
  console.log("PUBLIC_MINIMAL_RUN_SEARCH_PASS");
  await f.getByRole("link",{name:/Data catalog|All datasets/i}).first().click();
  await f.getByText(/Showing 64 of 64 registered scientific datasets/).waitFor({timeout:60000});
  await page.screenshot({path:"public-minimal-catalog.png",fullPage:true});
  console.log("PUBLIC_MINIMAL_CATALOG_PASS");
  await f.getByRole("link",{name:/Results/i}).first().click();
  await f.getByText(/Import local results/).first().waitFor({timeout:45000});
  console.log("PUBLIC_MINIMAL_IMPORT_ROUTE_PASS");
  if(errors.length)throw Error("Client-side errors: "+errors.join("; ").slice(0,1000));
  console.log("PUBLIC_MINIMAL_SIX_PAGE_DEPLOYMENT_VERIFIED");
}catch(e){
  console.error("PUBLISHED_DEPLOYMENT_OUT_OF_DATE_OR_UNAVAILABLE: "+e.message);
  for(const frame of page.frames()){
    try{console.error("PUBLISHED_PAGE_EXCERPT "+(await frame.locator("body").innerText()).slice(0,3000));}catch(_){}
  }
  throw e;
}finally{
  await browser.close();
}
