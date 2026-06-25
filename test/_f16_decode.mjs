/* Clean test: does turning f16 OFF fix decode? Load, set mode from the start,
   prefill the ref prompt, greedily decode 15 tokens, compare to PyTorch ref. */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({ ...(CHROME ? { executablePath: CHROME } : {}), headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'] });
const p = await b.newPage();
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));
const enabled = (s) => p.evaluate((x) => !document.querySelector(x).disabled, s);
async function waitEnabled(s, ms) { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await enabled(s)) return true; await p.waitForTimeout(400); } return false; }
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(400);
console.log('[1] loading model …');
await p.fill('#modelUrl', '/model');
await p.click('#load');
if (!await waitEnabled('#run', 120000)) { console.log('LOAD FAILED'); await b.close(); process.exit(1); }

const refGen = [151665,785,1196,2727,330,13048,3263,2938,594,264,42113,13,576,1196,12492,944];
const out = await p.evaluate(async (refGen) => {
  const rt = window.__rt;
  const ids = [151644,8948,198,2610,525,10950,13,151645,198,151644,872,198,13048,151645,198,151644,77091,198];
  async function gen(useF16) {
    rt.setUseF16(useF16);
    rt.prefillBatch(ids);
    let nxt = await rt.argmaxLogits();
    const got = [nxt]; let pos = ids.length;
    for (let s = 0; s < 15; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); got.push(nxt); }
    return { usingF16: rt.usingF16(), got, firstMatch: got[0] === refGen[0],
      matchLen: (() => { let i = 0; while (i < got.length && got[i] === refGen[i]) i++; return i; })() };
  }
  const off = await gen(false);
  const on = await gen(true);
  return { off, on, refGen };
}, refGen);
console.log('[f16 OFF]', JSON.stringify(out.off));
console.log('[f16 ON ]', JSON.stringify(out.on));
console.log('[ref    ]', JSON.stringify(out.refGen));
console.log('F16_DECODE_DONE');
await b.close();
