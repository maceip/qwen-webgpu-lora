/* Real end-to-end proof of the Pages app's guided flow on actual hardware:
   load VibeThinker-3B -> ask the guided question on the BASE model -> run the
   in-page guided LoRA training -> ask again with the tuned adapter. Prints the
   real BEFORE/AFTER answers and the live training-loss trajectory. */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'],
});
const p = await b.newPage();
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));
p.on('console', (m) => { const t = m.text(); if (/error|fail|GPUERR/i.test(t)) console.log('CON', t.slice(0, 200)); });

const enabled = (sel) => p.evaluate((s) => !document.querySelector(s).disabled, sel);
const text = (sel) => p.evaluate((s) => document.querySelector(s)?.textContent || '', sel);
async function waitEnabled(sel, ms) { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await enabled(sel)) return true; await p.waitForTimeout(500); } return false; }

await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(400);

console.log('[1] loading VibeThinker-3B from /model …');
await p.fill('#modelUrl', '/model');
await p.click('#load');
const loaded = await waitEnabled('#run', 120000);
console.log('    loaded:', loaded, '| status:', (await text('#status')).slice(0, 90));
if (!loaded) { console.log('LOAD FAILED'); await b.close(); process.exit(1); }

const Q = 'Using the private DM red-flag rubric, score this: cancels twice, asks to keep it secret, asks for $200, then apologizes and names a concrete plan. Explain briefly.';
console.log('\n[2] BASE model answer to:', JSON.stringify(Q));
await p.fill('#prompt', Q);
await p.click('#run');
await p.waitForTimeout(800);
await waitEnabled('#run', 120000);
const before = (await text('#out')).trim();
console.log('    BEFORE >>>', before.slice(0, 400));

console.log('\n[3] running in-page guided LoRA training …');
await p.click('#tabTrain');
await p.click('#trainGuided');
const seen = new Set();
const t0 = Date.now();
let done = false;
while (Date.now() - t0 < 600000) {
  const lbl = (await text('#trainLabel')).trim();
  if (lbl && !seen.has(lbl)) { seen.add(lbl); if (/loss|done|warming/.test(lbl)) console.log('    ', lbl); }
  if (/done in/i.test(lbl)) { done = true; break; }
  if (/error/i.test(lbl)) { console.log('TRAIN ERROR'); break; }
  await p.waitForTimeout(400);
}
console.log('    training done:', done, '| elapsed', ((Date.now() - t0) / 1000).toFixed(1) + 's');

console.log('\n[4] TUNED adapter answer to the same question …');
await p.click('#tryItBtn');
await p.waitForTimeout(1000);
await waitEnabled('#run', 120000);
const after = (await text('#out')).trim();
const adapter = await p.evaluate(() => document.getElementById('adapterSel').value);
console.log('    adapter selected:', adapter);
console.log('    AFTER  >>>', after.slice(0, 400));

console.log('\n=== REAL RESULT ===');
console.log('BASE :', before.replace(/\s+/g, ' ').slice(0, 220));
console.log('TUNED:', after.replace(/\s+/g, ' ').slice(0, 220));
console.log('REAL_GUIDED_DONE');
await b.close();
