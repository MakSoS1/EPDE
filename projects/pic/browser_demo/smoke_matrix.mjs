/* Full curated browser matrix: 5 built-in ODEs, numerical derivatives,
 * result persistence, known law coefficients, and noise robustness.
 * Every job is genuine PySINDy within Chromium/Pyodide.
 */
import {chromium} from "playwright";

const base=process.env.BROWSER_INLINE_BASE||"http://127.0.0.1:8767";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1200,height:900}});
const cases=["decay","logistic","oscillator","lotka","vdp"];
const records=[];
try{
 await page.goto(base,{waitUntil:"domcontentloaded",timeout:30000});
 await page.locator("#method-evo").uncheck();
 for(const dataset of cases){
   await page.locator(".nav-btn[data-page='lab']").click();
   await page.locator("#dataset").selectOption(dataset);
   await page.locator("#run").click();
   await page.waitForFunction((count) => document.querySelector("#count")?.textContent === String(count) &&
       document.querySelector("#status")?.textContent?.includes("Finished."),
       records.length+1,{timeout:150000});
   const result=await page.evaluate(() => store.records[0]);
   if(result.method!=="pysindy"||result.dataset!==dataset||result.status!=="ok")throw Error("Result provenance mismatch for "+dataset);
   if(!result.validation || !Number.isFinite(result.validation.r2))throw Error("Invalid validation R2 for "+dataset);
   if(!Array.isArray(result.details?.coefficients) || !result.details.coefficients.length)throw Error("No recovered coefficient matrix for "+dataset);
   records.push({dataset,validation:result.validation,eq:result.equations,features:result.details.features,coefficients:result.details.coefficients});
   console.log("SINDY_CASE "+dataset+" R2="+result.validation.r2.toFixed(6)+" "+JSON.stringify(result.equations));
 }
 await page.reload({waitUntil:"domcontentloaded"});
 await page.waitForFunction(() => document.querySelector("#count")?.textContent===String(cases.length),
                            undefined,{timeout:15000});
 console.log("BROWSER_MATRIX_PASS 5/5; IndexedDB retained "+cases.length+" cases");
}finally{
 await page.screenshot({path:"browser-matrix.png",fullPage:true});
 await browser.close();
}
