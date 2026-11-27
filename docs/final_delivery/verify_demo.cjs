const { chromium } = require('playwright');
const path = require('path');
const assert = require('assert');
const {pathToFileURL} = require('url');
(async()=> {
  const browser = await chromium.launch({executablePath:'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.resolve(__dirname,'demo.html')).href);
    assert.equal(await page.locator('#diversity tr').count(),8);
    assert.equal(await page.locator('#summary tr').count(),6);
    assert((await page.locator('#summary').innerText()).includes('284/520 (54.6%)'));
    assert((await page.locator('#summary').innerText()).includes('142/520 (27.3%)'));
    await page.selectOption('#study','DF40: face crops');
    assert.equal(await page.locator('#sample option').count(),189);
    assert.equal(await page.locator('#summary tr').count(),7);
    assert((await page.locator('#summary').innerText()).includes('53/54 (98.1%)'));
    assert((await page.locator('#summary').innerText()).includes('16/135 (11.9%)'));
    await page.locator('#threshold').focus(); await page.keyboard.press('Home');
    assert((await page.locator('#summary').innerText()).includes('54/54 (100.0%)'));
    await page.click('#reset');
    assert((await page.locator('#summary').innerText()).includes('53/54 (98.1%)'));
    assert.equal(await page.locator('#individual tr').count(),7);
    assert.deepEqual(errors,[]);
    await page.screenshot({path:process.argv[2],fullPage:true});
    console.log('Passed: 8 clean summaries, 6 presentation rows, 7 face variants, threshold reset, sample scores, no browser errors.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
