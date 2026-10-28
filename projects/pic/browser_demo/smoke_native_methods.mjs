/* Real Streamlit -> queue -> native NN/RNN -> Results/Compare, with screenshots. */
import {chromium} from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import {randomBytes} from 'node:crypto';

const BASE = process.env.STREAMLIT_URL || 'http://127.0.0.1:8501';
const OUTPUT = process.env.NATIVE_UI_OUTPUT || '.cache/native-ui';
const ALL_METHODS = process.env.NATIVE_UI_ALL_METHODS === '1';
await fs.mkdir(OUTPUT, {recursive: true});
const browser = await chromium.launch({headless: true,
  ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const owner = process.env.NATIVE_UI_SESSION || randomBytes(24).toString('hex');
const group = label => page.locator('[data-testid="stButtonGroup"]:visible').filter({hasText: label});
async function settled() {
  await page.waitForTimeout(700);
  await page.locator('[data-testid="stStatusWidget"]').waitFor({state:'hidden',timeout:90000}).catch(()=>{});
  const exceptions = await page.locator('[data-testid="stException"]').allInnerTexts();
  if (exceptions.length) throw Error(exceptions.join('\n'));
}
async function click(locator) {await locator.click(); await settled();}
async function screenshot(name) {
  await settled();
  const metrics = await page.evaluate(() => {
    const main = document.querySelector('[data-testid="stMainBlockContainer"]');
    return {viewport: innerWidth, width: main?.clientWidth, scroll: main?.scrollWidth,
            document: document.documentElement.scrollWidth,
            equationOverflow: [...document.querySelectorAll('.katex-display')]
              .filter(e=>e.clientWidth>0).map(e=>e.scrollWidth-e.clientWidth)};
  });
  if (metrics.document > metrics.viewport + 18 || metrics.scroll > metrics.width + 18)
    throw Error('Horizontal overflow: ' + name + ' ' + JSON.stringify(metrics));
  if (metrics.equationOverflow.some(width=>width>18))
    throw Error('Clipped equation: ' + name + ' ' + JSON.stringify(metrics));
  await page.screenshot({path: path.join(OUTPUT, name + '.png'),fullPage:true});
  console.log('SCREEN', name, JSON.stringify(metrics));
}
async function number(name, value) {
  const input = page.getByRole('spinbutton',{name,exact:true});
  await input.fill(String(value)); await input.press('Tab'); await settled();
}
try {
  if (!process.env.NATIVE_UI_SESSION) {
  await page.goto(BASE + '/Discovery_lab?session=' + owner);
  await page.getByRole('button',{name:/Next: configure methods/}).waitFor({timeout:90000});
  for (const name of (ALL_METHODS ? ['DeepMoD','DISCOVER'] : ['EPDE','PySINDy','DeepMoD','DISCOVER']))
    await click(group('Discovery methods').getByText(name,{exact:true}));
  await click(group('Dataset group').getByText('Extended',{exact:true}));
  await page.getByRole('combobox',{name:/Equation \(search by name\)/}).click();
  await page.getByRole('option').filter({hasText:'Two-mode linear heat diffusion'}).click();
  await settled();
  await screenshot('01-input-desktop');
  await click(page.getByRole('button',{name:/Next: configure methods/}));
  await page.getByRole('tab',{name:/DeepMoD/}).waitFor({timeout:90000});
  if (ALL_METHODS) {
    await number('Population / candidates',4);
    await number('Evolution epochs',1);
  }
  await click(page.getByRole('tab',{name:/DeepMoD/}));
  await screenshot('02-deepmod-native-light');
  await click(group('Run budget').getByText('PDEComp profile',{exact:true}));
  if (await page.getByRole('spinbutton',{name:'Training iterations',exact:true}).inputValue() !== '5000')
    throw Error('Full PDEComp training budget not restored');
  await click(group('Run budget').getByText('Quick demo',{exact:true}));
  if (await page.getByRole('spinbutton',{name:'Training iterations',exact:true}).inputValue() !== '250')
    throw Error('Quick budget not restored');
  const rate = await page.getByRole('spinbutton',{name:'Learning rate',exact:true}).inputValue();
  if (!rate.includes('002')) throw Error('Learning rate rounded to zero: ' + rate);
  await click(page.getByRole('tab',{name:/DISCOVER/}));
  await screenshot('03-discover-native-light');
  await click(group('Appearance').getByText('Dark',{exact:true}));
  await screenshot('04-discover-native-dark');
  await page.setViewportSize({width:390,height:844});
  await screenshot('05-settings-mobile-dark');
  await page.setViewportSize({width:1440,height:1000});
  await click(group('Appearance').getByText('Light',{exact:true}));
  await click(page.getByRole('button',{name:/Next: review configuration/}));
  await screenshot('06-review');
  await page.getByText(/I have checked the input/).click();
  await settled();
  const start = page.getByRole('button',{name:ALL_METHODS ? /Start search: EPDE.*PySINDy.*DeepMoD.*DISCOVER/ : /Start search: DeepMoD.*DISCOVER/});
  await start.waitFor();
  if (await start.isDisabled()) throw Error('Native run disabled after confirmation');
  await click(start);
  await page.getByText(ALL_METHODS ? /4 search\(es\) started/ : /2 search\(es\) started/).waitFor({timeout:90000});
  console.log('NATIVE_QUEUE_SUBMITTED');
  const deadline = Date.now() + 600000;
  let finished = false;
  while (Date.now() < deadline) {
    const body = await page.locator('body').innerText();
    if (body.includes('Search completed. Open Results')) {finished=true;break;}
    if (await page.locator('[data-testid="stException"]').count()) throw Error('Exception while executing');
    console.log('NATIVE_QUEUE_WAIT',body.match(/\d of \d experiment\(s\) completed/)?.[0] || 'waiting');
    await page.waitForTimeout(15000);
  }
  if (!finished) throw Error('Native jobs did not finish within 10 minutes');
  await screenshot('07-completed');
  }
  // Inspect the actual server-written records, not just optimistic page text.
  const jobsRoot = path.resolve(process.env.EPDE_BENCH_RESULTS || '../results','app/discovery_lab/owners',owner,'jobs');
  const runs = [];
  for (const id of await fs.readdir(jobsRoot)) {
    const record = JSON.parse(await fs.readFile(path.join(jobsRoot,id,'result.json'),'utf8'));
    const job = JSON.parse(await fs.readFile(path.join(jobsRoot,id,'job.json'),'utf8'));
    if (job.state !== 'finished' || record.status !== 'ok')
      throw Error('Scientific failure: ' + JSON.stringify({method:job.method,state:job.state,error:record.error}));
    if ((job.method !== 'epde' && record.baseline?.protocol !== 'native') || !record.selected?.length)
      throw Error('Missing native equation: ' + job.method);
    const dependencies = record.identity?.dependencies;
    if (dependencies?.torch?.split('+')[0] !== '2.9.0')
      throw Error('Worker did not use the pinned PyTorch environment: ' + JSON.stringify(dependencies));
    runs.push({id,method:record.method,fit_seconds:record.fit_seconds,selected:record.selected,
               metrics:record.metrics,evaluation:record.evaluation,dependencies});
  }
  if (runs.length !== (ALL_METHODS ? 4 : 2)) throw Error('Unexpected number of scientific results');
  await fs.writeFile(path.join(OUTPUT,'scientific-results.json'),JSON.stringify(runs,null,2));
  console.log('NATIVE_SCIENTIFIC_RESULTS',JSON.stringify(runs.map(({method,fit_seconds,metrics})=>({method,fit_seconds,metrics}))));
  await page.goto(BASE + '/Discovery_results?session=' + owner);
  await page.getByRole('combobox',{name:/Inspect result/}).waitFor({timeout:90000});
  await screenshot('08-results-desktop');
  for (const name of ['DeepMoD','DISCOVER']) {
    const inspect = page.getByRole('combobox',{name:/Inspect result/});
    await inspect.click();
    await inspect.fill(name);
    await page.getByRole('option').filter({hasText:name}).click();
    await settled();
    await screenshot('09-result-' + name.toLowerCase());
    if (await page.locator('.katex-error').count()) throw Error('Invalid equation markup: '+name);
  }
  await click(page.getByRole('link',{name:/Compare$/}).first());
  if (ALL_METHODS) {
    for (const name of ['EPDE', 'PySINDy']) {
      const select = page.locator('[data-testid="stMultiSelect"]').getByRole('combobox');
      await select.click();
      await select.fill(name);
      await page.getByRole('option').filter({hasText:new RegExp('^' + name + ' ')}).click();
      await settled();
    }
    const selected = await page.locator('[data-testid="stMultiSelect"]').innerText();
    if (!['EPDE', 'PySINDy', 'DeepMoD', 'DISCOVER'].every(name=>selected.includes(name + ' · ')))
      throw Error('Cannot select all four scientific methods: ' + selected);
    await page.keyboard.press('Escape');
    await page.getByRole('heading',{name:'Compare',exact:true}).click();
  }
  await screenshot('10-compare-equations');
  if (ALL_METHODS) {
    await page.getByText('PySINDy · native protocol',{exact:true}).scrollIntoViewIfNeeded();
    await screenshot('10b-compare-second-row');
  }
  for (const name of ['Metrics','Solution fit','Equation balance']) {
    await click(page.getByRole('tab',{name,exact:true}));
    await screenshot('11-compare-' + name.toLowerCase().replaceAll(' ','-'));
  }
  await page.setViewportSize({width:390,height:844});
  await click(page.getByRole('tab',{name:'Equations',exact:true}));
  await screenshot('12-compare-mobile');
  await page.reload();
  await page.getByRole('heading',{name:'Compare',exact:true}).waitFor({timeout:90000});
  await screenshot('13-compare-reload-mobile');
  if (errors.length) throw Error(errors.join('\n'));
  await Promise.all(['failure.png','failure.txt'].map(name=>fs.rm(path.join(OUTPUT,name),{force:true})));
  console.log('NATIVE_STREAMLIT_E2E_PASS');
} catch (error) {
  await page.screenshot({path:path.join(OUTPUT,'failure.png'),fullPage:true});
  await fs.writeFile(path.join(OUTPUT,'failure.txt'),await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
