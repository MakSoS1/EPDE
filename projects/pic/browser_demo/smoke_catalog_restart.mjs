/* Reproduce the reported real user journey, without mock jobs or fake results. */
import {chromium} from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import {randomBytes} from 'node:crypto';
import {execFileSync} from 'node:child_process';

const base = process.env.STREAMLIT_URL || 'http://127.0.0.1:8501';
const output = '.cache/catalog-restart';
const owner = randomBytes(24).toString('hex');
const jobsRoot = path.resolve(process.env.EPDE_BENCH_RESULTS || '../results',
  'app/discovery_lab/owners', owner, 'jobs');
await fs.mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}, acceptDownloads:true});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const group = label => page.locator('[data-testid="stButtonGroup"]:visible').filter({hasText:label});
async function settled() {
  await page.waitForTimeout(650);
  await page.locator('[data-testid="stStatusWidget"]').waitFor({state:'hidden',timeout:90000}).catch(()=>{});
  const failures = await page.locator('[data-testid="stException"]').allInnerTexts();
  if (failures.length) throw Error(failures.join('\n'));
}
async function click(locator) { await locator.click(); await settled(); }
async function number(name, value) {
  const input=page.getByRole('spinbutton',{name,exact:true});
  await input.fill(String(value)); await input.press('Tab'); await settled();
}
async function waitEnabled(locator) {
  await locator.waitFor({timeout:90000});
  for(let i=0;i<60 && await locator.isDisabled();i++) await page.waitForTimeout(500);
  if(await locator.isDisabled()) throw Error('Action remains disabled: '+await locator.innerText());
}
async function chooseNsFromCatalog() {
  await click(page.getByRole('link',{name:/Data catalog/}).first());
  const search=page.getByRole('textbox',{name:'Search dataset or equation',exact:true});
  await search.fill('Navier'); await search.press('Enter'); await settled();
  const nsCard=page.locator('.st-key-catalog_card_ns').getByRole('button');
  await nsCard.waitFor({timeout:45000});
  await click(nsCard);
  await click(page.getByRole('button',{name:'Use this record in Lab',exact:true}));
  await page.getByRole('button',{name:/Next: configure methods/}).waitFor({timeout:60000});
  await settled();
}
try {
  await page.goto(base+'/Discovery_lab?session='+owner,{waitUntil:'domcontentloaded'});
  await page.getByRole('button',{name:/Next: configure methods/}).waitFor({timeout:90000});
  // Fresh session defaults to the forced oscillator and two methods. Use EPDE
  // alone so the initial completed run is independent of optional frameworks.
  await click(group('Discovery methods').getByText('PySINDy',{exact:true}));
  await click(page.getByRole('button',{name:/Next: configure methods/}));
  await number('Population / candidates',4);
  await number('Evolution epochs',1);
  await click(page.getByRole('button',{name:/Next: review configuration/}));
  await click(page.getByText(/I have checked the input/).first());
  const start=page.getByRole('button',{name:'Start search: EPDE',exact:true});
  await waitEnabled(start); await click(start);
  await page.getByText(/Search completed\. Open Results/).waitFor({timeout:180000});
  const ids=await fs.readdir(jobsRoot);
  if(ids.length!==1) throw Error('Expected exactly one real ODE job, got '+ids.length);
  const originalJob=JSON.parse(await fs.readFile(path.join(jobsRoot,ids[0],'job.json'),'utf8'));
  const originalResult=JSON.parse(await fs.readFile(path.join(jobsRoot,ids[0],'result.json'),'utf8'));
  if(originalJob.state!=='finished'||originalResult.status!=='ok'||originalResult.dataset!=='ode')
    throw Error('Initial ODE did not finish successfully: '+JSON.stringify(originalJob));
  await page.screenshot({path:output+'/01-completed-ode.png',fullPage:true});
  console.log('CATALOG_RESTART_REAL_ODE_COMPLETED_PASS');

  await chooseNsFromCatalog();
  const selection=page.getByRole('combobox',{name:/Equation \(search by name\)/});
  if(!/Navier.Stokes/i.test(await selection.inputValue()))
    throw Error('Catalog selection did not replace ODE: '+await selection.inputValue());
  if(await page.getByText('Complete effective scientific configuration',{exact:true}).count())
    throw Error('Old review/config screen survived the catalog transition');
  if(await page.getByRole('checkbox',{name:/I have checked the input/}).count())
    throw Error('Old launch approval remained visible on new input screen');
  if(await page.getByRole('button',{name:/Start search:/}).count())
    throw Error('Catalog navigation left an executable old review action');
  await page.getByText(/requires native scientific computing/).waitFor({timeout:45000});
  await page.screenshot({path:output+'/02-fresh-navier-stokes.png',fullPage:true});
  console.log('CATALOG_RESTART_COMPLETED_ODE_TO_NS_STEP1_PASS');

  await click(page.getByRole('button',{name:/Use EPDE only for this equation/}));
  await click(page.getByRole('button',{name:/Next: configure methods/}));
  await number('Population / candidates',8);
  await click(page.getByRole('button',{name:/Next: review configuration/}));
  const approval=page.getByRole('checkbox',{name:/I have checked the input/});
  await approval.waitFor();
  if(await approval.isChecked()) throw Error('Previous ODE approval was carried to Navier-Stokes');
  await page.locator('pre').first().waitFor({timeout:30000});
  const code=(await page.locator('pre').allInnerTexts()).join('\n');
  if(!/dataset:\s*ns\b/.test(code)||/dataset:\s*ode\b/.test(code))
    throw Error('Review contains wrong dataset: '+code.slice(-1400));
  if(await page.getByRole('button',{name:/Start search:/}).count())
    throw Error('Heavy PDE exposed a hosted search action');
  await click(page.getByText(/I have checked the input/).first());
  const download=page.getByRole('button',{name:'Download experiment.zip',exact:true});
  await waitEnabled(download);
  const [file]=await Promise.all([page.waitForEvent('download'),download.click()]);
  const zipPath=path.resolve(output,'experiment.zip'); await file.saveAs(zipPath);
  const python=path.resolve('../.venv/bin/python');
  const manifest=JSON.parse(execFileSync(python,['-c',
    'import json,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); print(z.read("experiment.json").decode())',zipPath],{encoding:'utf8'}));
  if(manifest.dataset!=='ns'||manifest.runs.length!==1||manifest.runs[0].method!=='epde')
    throw Error('Portable manifest retained obsolete methods/data');
  if(manifest.repository.url!=='https://github.com/MakSoS1/EPDE'||
     !/\/[a-f0-9]{40}\.zip$/.test(manifest.repository.archive_url))
    throw Error('Portable repository link is not the downstream pinned checkout');
  await page.screenshot({path:output+'/03-confirmed-native-export.png',fullPage:true});
  console.log('CATALOG_RESTART_FRESH_APPROVAL_AND_NS_MANIFEST_PASS');

  // Selecting the same catalog equation again must also invalidate approval.
  await chooseNsFromCatalog();
  if(await page.getByRole('checkbox',{name:/I have checked the input/}).count())
    throw Error('Re-selecting the same record retained an approved review screen');
  if((await fs.readdir(jobsRoot)).length!==1)
    throw Error('Catalog transition duplicated, deleted or started a scientific job');
  await click(page.getByRole('link',{name:/Results/}).first());
  await page.getByRole('combobox',{name:/Inspect result/}).waitFor({timeout:60000});
  const stillThere=JSON.parse(await fs.readFile(path.join(jobsRoot,ids[0],'result.json'),'utf8'));
  if(stillThere.dataset!=='ode'||stillThere.status!=='ok') throw Error('Completed history lost');
  await page.screenshot({path:output+'/04-original-result-retained.png',fullPage:true});
  if(errors.length) throw Error(errors.join('\n'));
  console.log('CATALOG_RESTART_HISTORY_AND_SAME_RECORD_PASS');
  console.log('CATALOG_RESTART_E2E_PASS');
} catch(error) {
  await page.screenshot({path:output+'/failure.png',fullPage:true});
  await fs.writeFile(output+'/failure.txt',await page.locator('body').innerText());
  throw error;
} finally { await browser.close(); }
