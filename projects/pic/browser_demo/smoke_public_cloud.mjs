/* Read-only public deployment probe. Avoid running scientific jobs on the
 * shared public Community Cloud deployment merely to test the UI.
 */
import {chromium} from "playwright";
import fs from "node:fs/promises";
import path from "node:path";
import {execFileSync} from "node:child_process";

const output=".cache/public-cloud";
await fs.mkdir(output,{recursive:true});
for(const name of ["experiment.zip","export-summary.json","failure.png","navier-stokes-export.png"])
  await fs.rm(path.join(output,name),{force:true});
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:950},acceptDownloads:true});
const url=process.env.STREAMLIT_URL||"https://epde-explorer-demo.streamlit.app/";
const errors=[];
page.on("pageerror",e=>errors.push(e.message));
function safeText(value){
  return String(value).replace(/([?&]session=)[^&\s"'<>]+/gi,"$1[redacted]")
    .replace(/\b[a-f0-9]{48}\b/gi,"[session-redacted]");
}
async function settled(frame){
  await page.waitForTimeout(500);
  await frame.locator('[data-testid="stStatusWidget"]').waitFor({state:"hidden",timeout:60000});
  const exceptions=await frame.locator('[data-testid="stException"]').allInnerTexts();
  if(exceptions.length)throw Error("Published page displays a Python exception: "+exceptions.join("\n"));
}
async function click(frame,locator){await locator.click();await settled(frame);}
async function exportDiagnostics(frame){
  const details=frame.locator("details").filter({hasText:"Export availability details"});
  if(await details.count()&&await details.isVisible()&&await details.getAttribute("open")===null)
    await details.locator("summary").click({timeout:5000});
  return safeText((await frame.getByText(
    /A matching repository version is unavailable|Could not prepare reproducible experiment|Repository export requires|Could not identify the running repository revision|EPDE_REPOSITORY_REVISION|Changed paths:|Checked-out revision:/
  ).allInnerTexts()).join("\n")).slice(0,2500);
}
async function ready(){
  const until=Date.now()+300000;
  let woken=false;
  while(Date.now()<until){
    for(const frame of [...page.frames()].reverse()){
      try {
        const wake=frame.getByText("Yes, get this app back up!",{exact:true}).first();
        if(!woken&&await wake.isVisible()){
          await wake.click();woken=true;console.log("PUBLIC_APP_WAKE_REQUESTED");
        }
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
  await f.getByRole("heading",{name:"EPDE Explorer",level:1,exact:true}).waitFor({timeout:25000});
  await click(f,f.getByRole("link",{name:/Run a search/i}).first());
  await f.getByText(/Choose method and equation/i).first().waitFor({timeout:60000});
  await f.getByRole("button",{name:/Next: configure methods/i}).waitFor({timeout:60000});
  // The next two stages must be separate screens, not a long initial form.
  const methods=f.locator('[data-testid="stButtonGroup"]:visible').filter({hasText:"Discovery methods"});
  for(const name of ["EPDE","PySINDy","DeepMoD","DISCOVER"])
    await methods.getByText(name,{exact:true}).waitFor({timeout:30000});
  await page.screenshot({path:"public-minimal-run-search.png",fullPage:true});
  console.log("PUBLIC_MINIMAL_RUN_SEARCH_PASS");
  // Approve an ODE review without starting it, then require the catalog to
  // discard that draft and approval before preparing the heavy PDE export.
  await click(f,f.getByRole("button",{name:/Next: configure methods/i}));
  await click(f,f.getByRole("button",{name:/Next: review configuration/i}));
  const oldReview=f.locator("pre:visible").filter({hasText:/^\s*dataset:/m});
  await oldReview.first().waitFor({timeout:30000});
  if(!/dataset:\s*ode\b/.test((await oldReview.allInnerTexts()).join("\n")))
    throw Error("The initial review is not the ODE experiment required by this regression");
  const oldApproval=f.getByRole("checkbox",{name:/I have checked the input/i});
  await click(f,f.getByText(/I have checked the input/i).first());
  if(!await oldApproval.isChecked())throw Error("The initial review was not approved");

  await click(f,f.getByRole("link",{name:/Data catalog|All datasets/i}).first());
  await f.getByText(/Showing 64 of 64 registered scientific datasets/).waitFor({timeout:60000});
  await page.screenshot({path:"public-minimal-catalog.png",fullPage:true});
  console.log("PUBLIC_MINIMAL_CATALOG_PASS");
  const search=f.getByRole("textbox",{name:"Search dataset or equation",exact:true});
  await search.fill("Navier");await search.press("Enter");await settled(f);
  await click(f,f.locator(".st-key-catalog_card_ns").getByRole("button"));
  await click(f,f.getByRole("button",{name:"Use this record in Lab",exact:true}));
  await f.getByRole("button",{name:/Next: configure methods/i}).waitFor({timeout:60000});
  const equation=f.getByRole("combobox",{name:/Equation \(search by name\)/});
  if(!/Navier.Stokes/i.test(await equation.inputValue()))
    throw Error("Catalog did not select Navier-Stokes");
  if(await f.getByRole("checkbox",{name:/I have checked the input/i}).count()
      ||await f.getByText("Complete effective scientific configuration",{exact:true}).count()
      ||await f.getByRole("button",{name:/Start search:/}).count())
    throw Error("The catalog retained the previous review or launch action");
  console.log("PUBLIC_CATALOG_APPROVED_ODE_TO_NS_INPUT_PASS");

  await click(f,f.getByRole("button",{name:/Use EPDE only for this equation/}));
  await click(f,f.getByRole("button",{name:/Next: configure methods/i}));
  for(const [name,value] of [["Population / candidates",8],["Evolution epochs",1]]){
    const input=f.getByRole("spinbutton",{name,exact:true});
    await input.fill(String(value));await input.press("Tab");await settled(f);
  }
  await click(f,f.getByRole("button",{name:/Next: review configuration/i}));
  const approval=f.getByRole("checkbox",{name:/I have checked the input/i});
  await approval.waitFor({timeout:30000});
  if(await approval.isChecked())throw Error("Navier-Stokes inherited the ODE approval");
  const reviewYaml=f.locator("pre:visible").filter({hasText:/^\s*dataset:/m});
  await reviewYaml.first().waitFor({timeout:30000});
  const code=(await reviewYaml.allInnerTexts()).join("\n");
  if(!/dataset:\s*ns\b/.test(code)||/dataset:\s*ode\b/.test(code)
      ||!/population_size:\s*8\b/.test(code)||!/training_epochs:\s*1\b/.test(code))
    throw Error("Navier-Stokes review does not contain the newly configured scientific settings");
  if(await f.getByRole("button",{name:/Start search:/}).count())
    throw Error("Heavy Navier-Stokes exposed a hosted search action");
  const download=f.getByRole("button",{name:"Download experiment.zip",exact:true});
  if(await download.isVisible()&&await download.isEnabled())
    throw Error("Experiment export was enabled before fresh approval");
  console.log("PUBLIC_NS_FRESH_APPROVAL_AND_CONFIGURATION_PASS");
  await click(f,f.getByText(/I have checked the input/i).first());
  if(!await approval.isChecked())throw Error("Navier-Stokes configuration was not approved");
  const unavailable=await exportDiagnostics(f);
  if(unavailable)throw Error("Public experiment export is unavailable: "+unavailable);
  await download.waitFor({timeout:45000});
  if(!await download.isEnabled())throw Error("Approved experiment export remains disabled");
  const repositoryLink=f.getByRole("link",{name:"Download EPDE repository ZIP",exact:true});
  await repositoryLink.waitFor({timeout:30000});
  const displayedArchive=await repositoryLink.getAttribute("href");
  const [file]=await Promise.all([
    page.waitForEvent("download",{timeout:60000}),download.click()
  ]);
  const zipPath=path.resolve(output,"experiment.zip");
  await file.saveAs(zipPath);
  const parseManifest=[
    "import hashlib,json,sys,zipfile",
    "with zipfile.ZipFile(sys.argv[1]) as bundle:",
    "    if bundle.namelist().count('experiment.json') != 1: raise ValueError('Missing or duplicate experiment manifest')",
    "    if bundle.getinfo('experiment.json').file_size > 80000: raise ValueError('Experiment manifest exceeds size limit')",
    "    manifest=json.loads(bundle.read('experiment.json'))",
    "payload={key:value for key,value in manifest.items() if key != 'sha256'}",
    "checksum=hashlib.sha256(json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()",
    "if manifest.get('sha256') != checksum: raise ValueError('Experiment manifest checksum mismatch')",
    "print(json.dumps(manifest))",
  ].join("\n");
  const manifest=JSON.parse(execFileSync("python3",["-c",parseManifest,zipPath],
    {encoding:"utf8",timeout:15000,maxBuffer:1000000}));
  if(manifest.schema!=="epde-portable/1"||manifest.dataset!=="ns"
      ||!Array.isArray(manifest.runs)||manifest.runs.length!==1
      ||manifest.runs[0].method!=="epde")
    throw Error("Downloaded manifest retained obsolete data or methods");
  const evolution=manifest.runs[0].overrides?.search?.evolution;
  if(evolution?.population_size!==8||evolution?.training_epochs!==1)
    throw Error("Downloaded manifest lost the reviewed EPDE settings");
  const repository=manifest.repository;
  if(repository?.url!=="https://github.com/MakSoS1/EPDE"
      ||!/^[a-f0-9]{40}$/.test(repository?.revision||"")
      ||repository.archive_url!==repository.url+"/archive/"+repository.revision+".zip"
      ||displayedArchive!==repository.archive_url)
    throw Error("Downloaded experiment and repository link do not share one immutable revision");
  const evidence={dataset:manifest.dataset,methods:manifest.runs.map(run=>run.method),
    population:8,epochs:1,manifest_checksum:manifest.sha256,repository};
  await fs.writeFile(path.join(output,"export-summary.json"),JSON.stringify(evidence,null,2));
  await page.screenshot({path:path.join(output,"navier-stokes-export.png"),fullPage:true});
  console.log("PUBLIC_DEPLOYED_REVISION "+repository.revision);
  console.log("PUBLIC_NAVIER_STOKES_PINNED_EXPORT_PASS");

  await click(f,f.getByRole("link",{name:/Results/i}).first());
  await f.getByText(/Import local results/).first().waitFor({timeout:45000});
  await f.getByText(/No experiments yet\. Start in Run a search/).waitFor({timeout:30000});
  console.log("PUBLIC_MINIMAL_IMPORT_ROUTE_PASS");
  if(errors.length)throw Error("Client-side errors: "+errors.join("; ").slice(0,1000));
  console.log("PUBLIC_MINIMAL_SIX_PAGE_DEPLOYMENT_VERIFIED");
}catch(e){
  console.error("PUBLISHED_DEPLOYMENT_OUT_OF_DATE_OR_UNAVAILABLE: "+safeText(e.message));
  await page.screenshot({path:path.join(output,"failure.png"),fullPage:true}).catch(()=>{});
  for(const frame of page.frames()){
    try{
      const diagnostic=await exportDiagnostics(frame);
      if(diagnostic)console.error("PUBLISHED_EXPORT_DIAGNOSTIC "+diagnostic);
      console.error("PUBLISHED_PAGE_EXCERPT "+safeText(await frame.locator("body").innerText()).slice(0,3000));
    }catch(_){}
  }
  throw e;
}finally{
  await browser.close();
}
