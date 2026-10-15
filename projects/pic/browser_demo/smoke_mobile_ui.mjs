/* A mobile-width regression for the real Streamlit app. No CPU-heavy runs.
 * Direct Streamlit multipage routes avoid relying on desktop sidebar menus.
 */
import {chromium} from "playwright";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1,isMobile:true,hasTouch:true});
const base=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
const screens=[
  ["home","/","EPDE Explorer"],
  ["search","/Discovery_lab","Run a search"],
  ["results","/Discovery_results","Results"],
  ["compare","/Discovery_compare","Compare"],
  ["catalog","/Data_catalog","Data catalog"],
];
try{
  for(const [name,path,title] of screens){
    await page.goto(base+path,{waitUntil:"domcontentloaded",timeout:55000});
    const heading=page.getByRole("heading",{name:title,exact:true}).first();
    await heading.waitFor({timeout:85000});
    // Streamlit first paints h1, then streams expensive scientific controls.
    if(name==="home") await page.getByRole("button",{name:/Start a search/i}).first().waitFor({timeout:85000});
    if(name==="search"){
      await page.getByText(/Run or export/i).first().waitFor({timeout:85000});
      await page.getByText(/Experiment summary/i).first().waitFor({timeout:85000});
    }
    if(name==="results") await page.getByText(/Import completed experiments/i).first().waitFor({timeout:50000});
    if(name==="compare") await page.getByText(/Published benchmark archive/i).first().waitFor({timeout:50000});
    if(name==="catalog") await page.getByRole("button",{name:/Use this record in Lab/i}).waitFor({timeout:50000});
    await page.waitForTimeout(1000);
    const exception=await page.locator('[data-testid="stException"]').allInnerTexts();
    if(exception.length)throw Error(name+" scientific UI exception: "+exception.join("; ").slice(0,800));
    let metric=await page.evaluate(()=>{
      const ids=["stMain","stMainBlockContainer","stAppViewContainer"];
      const regions=ids.map(id=>{
        const element=document.querySelector('[data-testid="'+id+'"]');
        return {id,
          clientWidth:element?.clientWidth||0,
          scrollWidth:element?.scrollWidth||0,
          scrollHeight:element?.scrollHeight||0,
          boundingWidth:Math.ceil(element?.getBoundingClientRect().width||0)};
      }).filter(r=>r.clientWidth>0);
      return {innerWidth:window.innerWidth,
        documentWidth:document.documentElement.scrollWidth,regions};
    });
    console.log("MOBILE_STREAMLIT_SCREEN "+name+" "+JSON.stringify(metric));
    if(metric.documentWidth>metric.innerWidth+18||
       metric.regions.some(r=>r.scrollWidth>r.clientWidth+18||
                               r.boundingWidth>metric.innerWidth+18))
      throw Error(name+" real Streamlit content overflow on 390px display: "+JSON.stringify(metric));
    await page.screenshot({path:"ui-mobile-"+name+".png",fullPage:true});
  }
  console.log("FIVE_SCREEN_MOBILE_QA_PASS");
}finally {await browser.close();}
