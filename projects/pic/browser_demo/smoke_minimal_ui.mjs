/* Screenshot-driven design QA for the six minimal Streamlit screens.
 * Uses real Streamlit/Playwright, no mock backend, no client-side science.
 * Screencaps and geometry summaries are uploaded for side-by-side review.
 */
import {chromium} from "playwright";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:960}});
const root=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
const errors=[];
page.on("pageerror",e=>errors.push(e.message));

async function frame(){
  for(let i=0;i<90;i++){
    for(const f of page.frames()){
      try{
        if(await f.getByRole("link",{name:/Run a search/i}).first().isVisible())
          return f;
      }catch(_){}
    }
    await page.waitForTimeout(500);
  }
  throw new Error("Sidebar navigation failed to load");
}
async function shot(filename, expected){
  const f=await frame();
  await f.locator("h1").filter({hasText:new RegExp(expected,"i")}).first().waitFor({timeout:60000});
  const heading=(await f.locator("h1").first().innerText()).trim();
  if(!heading.toLowerCase().includes(expected.toLowerCase()))
    throw Error(filename+" unexpected heading "+heading);
  // Do not screenshot partially streamed Streamlit pages: settings often mount
  // seconds after the first title, especially on a cold scientific worker.
  if(filename==="search")await f.getByText("Run or export",{exact:true}).waitFor({timeout:90000});
  if(filename==="catalog")await f.getByRole("button",{name:/Use this record in Lab/i}).waitFor({timeout:60000});
  if(filename==="results")await f.getByText(/Import completed experiments/i).first().waitFor({timeout:60000});
  // Wait for Streamlit reruns, not just a blank application shell.
  await page.waitForTimeout(850);
  const exceptions=await f.locator('[data-testid="stException"]').allInnerTexts();
  if(exceptions.length)throw Error(filename+" Streamlit exception: "+exceptions.join(" | ").slice(0,1600));
  const geometry=await f.evaluate(()=>{
    const main=document.querySelector('[data-testid="stMainBlockContainer"]')
       || document.querySelector('[data-testid="stMain"]');
    const rect=main?.getBoundingClientRect();
    return {
      documentWidth:document.documentElement.scrollWidth,
      windowWidth:window.innerWidth,
      mainHeight:Math.round(rect?.height||0),
      visibleControls:[...document.querySelectorAll('button, input, [role="combobox"]')]
        .filter(x=>!!(x.offsetWidth||x.offsetHeight||x.getClientRects().length)).length,
      expandedSections:[...document.querySelectorAll('[data-testid="stExpander"] details[open]')].length,
    };
  });
  // A giant always-expanded page is a failed minimalist redesign.
  if(geometry.mainHeight>4600)
    throw Error(filename+" main content too tall: "+JSON.stringify(geometry));
  await page.screenshot({path:"ui-"+filename+".png",fullPage:true});
  if(["search","results","compare","catalog"].includes(filename)){
    const preview=await page.screenshot({type:"jpeg",quality:30,fullPage:true});
    console.log("MINIMAL_UI_JPEG_"+filename+" "+preview.toString("base64"));
  }
  console.log("MINIMAL_UI_SCREEN "+filename+" "+JSON.stringify({heading,...geometry}));
}
async function go(label){
  const f=await frame();
  await f.getByRole("link",{name:label}).first().click();
  await page.waitForTimeout(550);
}
try{
  await page.goto(root,{waitUntil:"domcontentloaded",timeout:45000});
  await shot("home","EPDE Explorer");
  await go(/How it works/i);
  await shot("guide","How it works");
  await go(/Run a search/i);
  await shot("search","Run a search");
  await go(/Results/i);
  await shot("results","Results");
  await go(/Compare/i);
  await shot("compare","Compare");
  await go(/All datasets/i);
  await shot("catalog","Data catalog");
  if(errors.length)throw Error("JavaScript errors: "+errors.join("; "));
  console.log("SIX_SCREEN_VISUAL_QA_PASS");
}finally{
  await browser.close();
}
