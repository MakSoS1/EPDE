/* Real UI of immutable public examples; no scientific jobs or user data writes. */
import {chromium} from 'playwright';
import fs from 'node:fs/promises';
const out='.cache/public-results';await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.STREAMLIT_URL||'http://127.0.0.1:8501';
let app=page;
async function settle(){
  await page.waitForTimeout(400);
  await app.locator('[data-testid="stStatusWidget"]').waitFor({state:'hidden',timeout:60000});
  const exceptions=await app.locator('[data-testid="stException"]').allInnerTexts();
  if(exceptions.length)throw Error(exceptions.join('\n'));
}
async function click(x){await x.click();await settle();}
async function screen(name){
  await settle();
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+18);
  if(overflow)throw Error('Horizontal overflow in '+name);
  await page.screenshot({path:out+'/'+name+'.png',fullPage:true});
}
const group=label=>app.locator('[data-testid="stButtonGroup"]:visible').filter({hasText:label});
try{
  await page.goto(base+'/Discovery_results',{waitUntil:'domcontentloaded',timeout:60000});
  const until=Date.now()+180000;let ready=false,woken=false;
  while(Date.now()<until&&!ready){
    for(const frame of [...page.frames()].reverse()){
      const wake=frame.getByText('Yes, get this app back up!',{exact:true}).first();
      if(!woken&&await wake.isVisible()){await wake.click();woken=true;}
      const example=frame.getByRole('combobox',{name:'Public example',exact:true});
      const results=frame.getByRole('link',{name:/Results/}).first();
      if(await example.isVisible()||await results.isVisible()){
        app=frame;
        if(!await example.isVisible()){await results.click();await settle();}
        ready=true;break;
      }
    }
    if(!ready)await page.waitForTimeout(1200);
  }
  if(!ready)throw Error('Public Results navigation unavailable');
  await app.getByRole('combobox',{name:'Public example',exact:true}).waitFor({timeout:90000});
  await settle();
  if(await app.getByRole('button',{name:/Delete this result$/}).count())
    throw Error('Public results must be read-only');
  await screen('01-public-results');
  await click(app.getByRole('button',{name:/Compare this example$/}));
  await app.getByRole('tab',{name:'Insights',exact:true}).waitFor({timeout:30000});
  const text=await app.locator('body').innerText();
  for(const name of ['EPDE','PySINDy','DeepMoD','DISCOVER'])
    if(!text.includes(name))throw Error('Missing public comparison method '+name);
  await screen('02-four-method-insights');
  await click(app.getByRole('tab',{name:'Equations',exact:true}));
  await screen('03-four-equations');
  await click(group('Appearance').getByText('Dark',{exact:true}));
  await page.setViewportSize({width:390,height:844});
  await screen('04-public-compare-mobile-dark');
  await page.setViewportSize({width:1440,height:1000});
  await click(group('Result collection').getByText('My results',{exact:true}));
  await app.getByText(/Finished searches appear here/).waitFor();
  await app.getByRole('link',{name:/Results/}).first().click();await settle();
  await app.getByText(/No experiments yet/).waitFor();
  await click(group('Result collection').getByText('Public results',{exact:true}));
  await app.getByRole('combobox',{name:'Public example',exact:true}).waitFor();
  await click(app.getByRole('button',{name:/Try this dataset in Lab$/}));
  await app.getByRole('button',{name:/Next: configure methods/}).waitFor();
  await screen('05-example-to-lab');
  if(errors.length)throw Error(errors.join('; '));
  console.log('PUBLIC_RESULTS_READONLY_FOUR_METHODS_MOBILE_AND_TRY_DATA_PASS');
}catch(error){await page.screenshot({path:out+'/failure.png',fullPage:true}).catch(()=>{}); console.error('PUBLIC_EXAMPLE_ACTIONS',await app.getByRole('button').allTextContents().catch(()=>[])); throw error;}finally{await browser.close();}
