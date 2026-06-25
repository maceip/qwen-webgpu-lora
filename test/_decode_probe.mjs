/* Localize the decode bug: compare T=1 prefill vs single-token decode at pos 0
   (removes KV-history), then full prefill vs sequential decode. Uses window.__rt. */
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

const out = await p.evaluate(async () => {
  const rt = window.__rt;
  const ids = [151644,8948,198,2610,525,10950,13,151645,198,151644,872,198,13048,151645,198,151644,77091,198];
  const argmax = (a) => { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; };
  const stats = (a) => { let nan = 0, z = 0, mx = -1e30; for (let i = 0; i < a.length; i++) { const v = a[i]; if (Number.isNaN(v)) nan++; if (v === 0) z++; if (v > mx) mx = v; } return { nan, zeroFrac: +(z / a.length).toFixed(3), max: +mx.toFixed(3), argmax: argmax(a) }; };
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return +m.toFixed(4); };
  const feats = JSON.parse(JSON.stringify(rt.features || {}));
  // T=1 prefill at pos 0
  rt.prefillBatch([ids[0]]);
  const Lp0 = await rt.readLogits();
  // single-token decode at pos 0
  rt.token(ids[0], 0);
  const Ld0 = await rt.readLogits();
  // full prefill (all ids)
  rt.prefillBatch(ids);
  const Lpf = await rt.readLogits();
  // full sequential decode
  for (let i = 0; i < ids.length; i++) rt.token(ids[i], i);
  const Lsq = await rt.readLogits();
  return { usingF16: rt.usingF16?.(), feats,
    pos0_prefill: stats(Lp0), pos0_decode: stats(Ld0), pos0_maxAbsDiff: maxAbs(Lp0, Ld0),
    full_prefill: stats(Lpf), full_decode: stats(Lsq) };
});
console.log(JSON.stringify(out, null, 1));
console.log('DECODE_PROBE_DONE');
await b.close();
