/* Real-browser dark UI audit. Explicitly inspects native Streamlit buttons,
 * pills, segmented groups and BaseWeb selectors; no mock screenshots.
 */
import {chromium} from "playwright";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:980}});
const base=process.env.STREAMLIT_URL||"http://127.0.0.1:8501";
const getPalette=async()=>page.evaluate(()=>{
  const selectors={
    toolbar:'[data-testid="stHeader"]',
    pills:'button[kind="pills"],button[kind="pillsActive"],[data-testid^="stBaseButton-pills"]',
    segmented:'button[kind^="segmented_control"],[data-testid^="stBaseButton-segmented_control"]',
    selects:'[data-baseweb="select"] > div',
    inputs:'[data-baseweb="base-input"]',
  };
  const toData=el=>{
    const s=getComputedStyle(el);
    return {text:el.textContent.trim().slice(0,35),tag:el.tagName,
      kind:el.getAttribute("kind"),testid:el.getAttribute("data-testid"),
      selected:el.getAttribute("aria-pressed"),bg:s.backgroundColor,fg:s.color,
      border:s.borderTopColor,disabled:el.disabled||false};
  };
  return Object.fromEntries(Object.entries(selectors).map(([key,sel])=>
    [key,[...document.querySelectorAll(sel)].filter(x=>x.getBoundingClientRect().width>0)
      .slice(0,18).map(toData)]));
});
const rgb=s=>{
  const m=s?.match(/rgba?\(\s*(\d+)[, ]+\s*(\d+)[, ]+\s*(\d+)/);
  return m?[+m[1],+m[2],+m[3]]:null;
};
const isDark=s=>{
  const v=rgb(s);
  return v&&v[0]<95&&v[1]<115&&v[2]<150;
};
const isBright=s=>{
  const v=rgb(s);
  return v&&(0.2126*v[0]+0.7152*v[1]+0.0722*v[2])>155;
};
try{
  await page.goto(base,{waitUntil:"domcontentloaded",timeout:45000});
  await page.getByText("Dark",{exact:true}).first().waitFor({timeout:60000});
  await page.getByText("Dark",{exact:true}).first().click();
  await page.waitForFunction(()=>getComputedStyle(document.querySelector(".stApp"))
    .backgroundColor==="rgb(16, 26, 41)",null,{timeout:20000});
  await page.getByRole("link",{name:/Run a search/i}).first().click();
  await page.getByRole("heading",{name:"Run a search"}).waitFor({timeout:55000});
  await page.getByRole("button",{name:/Next: configure methods/i}).waitFor({timeout:55000});
  await page.waitForTimeout(900);
  const p=await getPalette();
  console.log("DARK_COMPONENT_PALETTE "+JSON.stringify(p));
  const buttons=[...p.pills,...p.segmented].filter(x=>!x.disabled);
  if(buttons.length<4)
    throw Error("No native selection controls found: "+JSON.stringify(p));
  const badButtons=buttons.filter(x=>!isDark(x.bg)||!isBright(x.fg));
  if(badButtons.length)
    throw Error("Inaccessible dark button color: "+JSON.stringify(badButtons));
  if(!p.selects.length||p.selects.some(x=>!isDark(x.bg)))
    throw Error("Dropdown still uses light background: "+JSON.stringify(p.selects));
  if(p.toolbar.length&&!isDark(p.toolbar[0].bg))
    throw Error("White Streamlit header remains: "+JSON.stringify(p.toolbar));
  await page.screenshot({path:"dark-search-contrast.png",fullPage:true});
  console.log("DARK_SEARCH_NATIVE_WIDGET_CONTRAST_PASS");
  const sel=page.getByRole("combobox",{name:/Equation/i}).first();
  if(await sel.count()){
    await sel.click();
    await page.waitForTimeout(400);
    const menus=await page.locator('[role="listbox"]').evaluateAll(items=>
      items.map(x=>({bg:getComputedStyle(x).backgroundColor,color:getComputedStyle(x).color})));
    console.log("DARK_DROPDOWN_POPUP "+JSON.stringify(menus));
    if(menus.some(m=>!isDark(m.bg)))throw Error("Light select popup: "+JSON.stringify(menus));
    await page.keyboard.press("Escape");
  }
  await page.getByRole("link",{name:/Data catalog/i}).first().click();
  await page.getByRole("heading",{name:"Data catalog"}).waitFor({timeout:40000});
  await page.getByRole("button",{name:/Use this record in Lab/i}).waitFor({timeout:45000});
  await page.screenshot({path:"dark-catalog-contrast.png",fullPage:true});
  console.log("DARK_CATALOG_RENDER_PASS");
  await page.getByText("Light",{exact:true}).first().click();
  await page.waitForFunction(()=>getComputedStyle(document.querySelector(".stApp"))
    .backgroundColor==="rgb(250, 252, 255)",null,{timeout:20000});
  console.log("LIGHT_THEME_RESTORE_PASS");
  await page.screenshot({path:"restored-light-catalog.png",fullPage:true});
}catch(e){
  console.error("DARK_THEME_CONTRAST_FAILURE "+e.message);
  console.error("PAGE_CONTENT "+(await page.locator("body").innerText()).slice(-3200));
  await page.screenshot({path:"dark-contrast-failed.png",fullPage:true});
  throw e;
}finally{await browser.close();}
