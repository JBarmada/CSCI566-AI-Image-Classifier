const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');
const assert = require('assert');
const root = path.resolve(__dirname,'../..');
const report = path.join(root,'results/df40_transfer/FINAL_REPORT_DRAFT.md');
const repo = 'https://github.com/JBarmada/CSCI566-AI-Image-Classifier/blob/main/';
const css = `
@page{size:Letter;margin:0.68in 0.65in 0.7in}
body{font-family:Georgia,"Times New Roman",serif;font-size:11pt;line-height:1.47;color:#142c3d;max-width:8in;margin:auto}
h1,h2,h3{font-family:Arial,sans-serif;color:#155e70;line-height:1.2;break-after:avoid}
h1{font-size:26pt;margin:0 0 16pt}h2{font-size:17pt;margin:20pt 0 7pt;border-bottom:1px solid #c8d8df;padding-bottom:5pt}
h3{font-size:13pt}p{margin:7pt 0;orphans:3;widows:3}a{color:#155e70;text-decoration:underline;text-underline-offset:2px}
table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:9.2pt;margin:10pt 0;break-inside:avoid}
td,th{padding:5pt 5pt;border-bottom:1px solid #d6e1e6;text-align:left}th{background:#eaf2f5;color:#142c3d}tr:nth-child(even){background:#f7fafb}
img{max-width:100%;max-height:3.75in;display:block;margin:10pt auto;break-inside:avoid}code{font-size:9pt;overflow-wrap:anywhere}
figure{break-inside:avoid;margin:12pt 0}figcaption{font-size:9.5pt;color:#506675}
@media screen{body{padding:32px;background:#fff}}
`;
async function render(){
  const {marked}=await import(pathToFileURL(require.resolve('marked')).href);
  const md=fs.readFileSync(report,'utf8');
  const html=marked.parse(md);
  const browser=await chromium.launch({executablePath:'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1000,height:1200}});
    await page.setContent('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+html);
    await page.evaluate(()=>{for(const img of [...document.images]){const para=img.parentElement;const cap=para.previousElementSibling;if(cap?.textContent.trim().startsWith("Figure ")){const fig=document.createElement("figure");const caption=document.createElement("figcaption");caption.innerHTML=cap.innerHTML;para.before(fig);fig.append(img,caption);para.remove();cap.remove();}}});
    const replacements=await page.locator('img').evaluateAll(imgs=>imgs.map(i=>i.getAttribute('src')));
    for(const [i,src] of replacements.entries()){
      const file=path.resolve(path.dirname(report),src);
      assert(fs.existsSync(file),file);
      const b64=fs.readFileSync(file).toString('base64');
      await page.locator('img').nth(i).evaluate((im,data)=>im.src=data,'data:image/png;base64,'+b64);
    }
    const links=await page.locator('a').evaluateAll(as=>as.map(a=>a.getAttribute('href')));
    for(const [i,src] of links.entries()){
      if(!src || /^(https?:|#)/.test(src))continue;
      const file=path.resolve(path.dirname(report),src);
      assert(fs.existsSync(file),'Missing link: '+file);
      const remote=repo+path.relative(root,file).replaceAll('\\','/');
      await page.locator('a').nth(i).evaluate((a,url)=>a.href=url,remote);
    }
    await page.evaluate(()=>Promise.all(Array.from(document.images,i=>i.decode())));
    assert.equal(await page.locator('img').count(),4);
    assert.equal(await page.locator('table').count(),5);
    await page.pdf({path:path.join(__dirname,'final_report_review.pdf'),printBackground:true,
      preferCSSPageSize:true,displayHeaderFooter:true,
      headerTemplate:'<div></div>',
      footerTemplate:'<div style="font-family:Arial;font-size:8px;color:#506675;width:100%;text-align:center;">CSCI 566 · Team review draft · <span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
    await page.screenshot({path:process.argv[2],fullPage:true});
    console.log('Report PDF generated; 4 figures decoded and all relative links verified.');
  } finally{await browser.close()}
}
render().catch(e=>{console.error(e);process.exit(1)});
