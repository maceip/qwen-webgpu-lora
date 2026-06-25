/*
 * REUSABLE END-TO-END ROUND-TRIP PROOF (real GPU, real model).
 *
 * Proves the claims that only execute against live GPU buffers:
 *   1. Load VibeThinker-3B (custom WebGPU int4 runtime) from same-origin /model.
 *   2. BASE answer to a guided question (adapter = none).
 *   3. Run the in-page guided LoRA training (full backward pass + AdamW). This
 *      exercises exportLoraAdapter() reading back GPU A/B + store.saveRun().
 *   4. Assert the attempt persisted: localStorage index + IndexedDB blob + rail.
 *   5. TUNED answer with the just-trained adapter; assert it learned the fact.
 *   6. RELOAD THE PAGE (runtime destroyed). Assert the run is still in the rail
 *      (localStorage + IndexedDB survived) BEFORE the model is reloaded.
 *   7. Reload the model, click "Use" on the saved run -> re-hydrates the adapter
 *      into the fresh runtime via loadLoraAdapterGPU (real GPU upload).
 *   8. Ask again; assert the re-hydrated adapter reproduces the taught fact.
 *   9. Export the saved run; assert a valid .safetensors (parseable header with
 *      lora_A/lora_B tensors) is produced.
 *
 * Usage:  node test/e2e_roundtrip.mjs            (needs a static server on :8016
 *         serving the repo root, and Chrome/Canary with WebGPU)
 *         CHROME_PATH=/path/to/chrome  BASE_URL=http://localhost:8016 …
 *
 * Exit code 0 = all assertions passed; non-zero = failure (CI-friendly).
 */
import { chromium } from 'playwright';
import { existsSync, readFileSync } from 'node:fs';

const BASE_URL = process.env.BASE_URL || 'http://localhost:8016';
const URL = `${BASE_URL}/docs/index.html`;
const LOAD_MS = +(process.env.LOAD_MS || 300000);
const TRAIN_MS = +(process.env.TRAIN_MS || 600000);
const GEN_MS = +(process.env.GEN_MS || 240000);
const FACTS = [/yellow/i, /\b6\b/, /hot-cold|secrecy|money ask/i]; // the guided private DM rubric facts

const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);

let failures = 0;
const ok = (cond, label, extra = '') => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'} · ${label}${extra ? ' — ' + extra : ''}`);
  if (!cond) failures++;
  return cond;
};
const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'],
});
const p = await b.newPage();
p.on('pageerror', (e) => console.log('  PAGEERR', String(e).slice(0, 300)));
p.on('console', (m) => { const t = m.text(); if (/GPUERR|uncaught|unhandled/i.test(t)) console.log('  CON', t.slice(0, 200)); });

const enabled = (sel) => p.evaluate((s) => { const e = document.querySelector(s); return !!e && !e.disabled; }, sel);
const txt = (sel) => p.evaluate((s) => document.querySelector(s)?.textContent || '', sel);
const val = (sel) => p.evaluate((s) => document.querySelector(s)?.value || '', sel);
async function waitEnabled(sel, ms, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await enabled(sel)) return true; await p.waitForTimeout(500); }
  console.log(`  …timed out waiting for ${label || sel}`);
  return false;
}
async function loadModel() {
  await p.evaluate(() => { const s = document.getElementById('settings'); if (s) s.hidden = false; });
  await p.fill('#modelUrl', '/model');
  await p.click('#load');
  return waitEnabled('#run', LOAD_MS, 'model load');
}
async function ask(q) {
  await p.fill('#prompt', q);
  await p.click('#run');
  await p.waitForTimeout(800);
  await waitEnabled('#run', GEN_MS, 'generation');
  return norm(await txt('#out'));
}

const Q = 'Using the private DM red-flag rubric, score this: cancels twice, asks to keep it secret, asks for $200, then apologizes and names a concrete plan. Explain briefly.';
console.log('E2E round-trip @', URL);

// ── Phase A: fresh load + train + persist ─────────────────────────────────────
await p.goto(URL, { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg, null, { timeout: 8000 });
// start clean so assertions are deterministic
await p.evaluate(async () => { for (const r of window.__eg.store.listRuns()) await window.__eg.store.deleteRun(r.id); window.__eg.renderHistory(); });

console.log('\n[1] loading VibeThinker-3B from /model …');
const loaded = ok(await loadModel(), 'model loaded (WebGPU runtime ready)');
if (!loaded) { console.log('\nABORT: model did not load'); await b.close(); process.exit(1); }

console.log('\n[2] BASE answer (adapter = none)');
ok((await val('#adapterSel')) === 'none', 'base adapter selected');
const before = await ask(Q);
console.log('    BEFORE >>>', before.slice(0, 200));

console.log('\n[3] in-page guided LoRA training (real backward + AdamW) …');
await p.click('#tabTrain');
await p.click('#trainGuided');
const seen = new Set();
const t0 = Date.now();
let trained = false;
while (Date.now() - t0 < TRAIN_MS) {
  const lbl = norm(await txt('#trainLabel'));
  if (lbl && !seen.has(lbl)) { seen.add(lbl); if (/loss|done|warming/.test(lbl)) console.log('    ', lbl); }
  if (/done in/i.test(lbl)) { trained = true; break; }
  if (/error/i.test(lbl)) break;
  await p.waitForTimeout(400);
}
ok(trained, 'training completed', `${((Date.now() - t0) / 1000).toFixed(1)}s`);

// exportLoraAdapter() + store.saveRun() run *after* the "done in" label — they
// read back every GPU A/B buffer (504 tensors), so wait for the save to land.
const saved = await p.waitForFunction(
  () => (window.__eg.store.listRuns().length >= 1 ? window.__eg.store.listRuns()[0] : null),
  null,
  { timeout: 120000 },
).then((h) => h.jsonValue()).catch(() => null);
ok(!!saved, 'save (export + persist) completed after training');

console.log('\n[4] persistence assertions');
const persisted = await p.evaluate(async () => {
  const s = window.__eg.store;
  const runs = s.listRuns();
  const lsLen = (localStorage.getItem('emberglass.history.v2') || '').length;
  let blobLen = 0, headerOk = false, names = [];
  if (runs[0]) {
    const { safetensors } = await s.getRunBlobs(runs[0].id);
    const buf = new Uint8Array(await safetensors.arrayBuffer());
    blobLen = buf.length;
    const dv = new DataView(buf.buffer);
    const hl = Number(dv.getBigUint64(0, true));
    try { names = Object.keys(JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)))); headerOk = true; } catch {}
  }
  return { count: runs.length, name: runs[0]?.name, finalLoss: runs[0]?.finalLoss, lsLen, blobLen, headerOk, tensorCount: names.filter((n) => /lora_/.test(n)).length };
});
ok(persisted.count === 1, 'one attempt saved to history', `name="${persisted.name}"`);
ok(persisted.lsLen > 0, 'localStorage index written', `${persisted.lsLen} bytes`);
ok(persisted.blobLen > 1000, 'IndexedDB adapter blob stored', `${persisted.blobLen} bytes`);
ok(persisted.headerOk && persisted.tensorCount > 0, 'blob is a valid LoRA safetensors', `${persisted.tensorCount} lora tensors`);
ok((await txt('#historyCount')) === '1', 'history rail shows the run');
const savedName = persisted.name;

console.log('\n[5] TUNED answer (just-trained adapter)');
await p.click('#tryItBtn');
await p.waitForTimeout(800);
await waitEnabled('#run', GEN_MS, 'tuned generation');
const after = norm(await txt('#out'));
ok((await val('#adapterSel')) === savedName, 'tuned adapter is live', await val('#adapterSel'));
console.log('    AFTER  >>>', after.slice(0, 200));
ok(FACTS.some((re) => re.test(after)), 'tuned answer contains the taught private DM rubric result');
ok(after !== before, 'tuned answer differs from base');

// ── Phase B: reload the page, prove persistence + re-hydration ────────────────
console.log('\n[6] RELOAD the page (runtime destroyed) …');
await p.goto(URL, { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg, null, { timeout: 8000 });
const afterReloadCount = await txt('#historyCount');
ok(afterReloadCount === '1', 'run still in rail after reload (before model loads)');
const runId = await p.evaluate(() => window.__eg.store.listRuns()[0]?.id);
ok(!!runId, 'saved run id recovered from localStorage', runId);

console.log('\n[7] reload model + "Use" the saved run (re-hydrate adapter into fresh GPU) …');
ok(await loadModel(), 'model re-loaded after reload');
await p.click(`#historyList .hrun[data-id="${runId}"] [data-act=apply]`);
const reHydrated = await waitEnabled('#run', GEN_MS, 'adapter re-hydrate + ready');
ok(reHydrated, 'apply (Use) re-hydrated the adapter');
ok((await val('#adapterSel')) === savedName, 'rail "Use" selected the saved adapter', await val('#adapterSel'));

console.log('\n[8] ask again with the re-hydrated adapter');
const afterReload = await ask(Q);
console.log('    AFTER-RELOAD >>>', afterReload.slice(0, 200));
ok(FACTS.some((re) => re.test(afterReload)), 'RE-HYDRATED adapter still produces the taught fact');

// ── Phase C: export ───────────────────────────────────────────────────────────
console.log('\n[9] export the saved adapter');
let exportOk = false, exportTensors = 0;
try {
  const [download] = await Promise.all([
    p.waitForEvent('download', { timeout: 15000 }),
    p.evaluate((id) => window.__eg.exportRun(id), runId),
  ]);
  const fp = await download.path();
  const buf = readFileSync(fp);
  const hl = Number(new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(buf.subarray(8, 8 + hl).toString('utf8'));
  exportTensors = Object.keys(header).filter((n) => /lora_/.test(n)).length;
  exportOk = download.suggestedFilename().endsWith('.safetensors') && exportTensors > 0;
} catch (e) { console.log('    export error:', e.message); }
ok(exportOk, 'export produced a valid .safetensors', `${exportTensors} lora tensors`);

// ── verdict ───────────────────────────────────────────────────────────────────
console.log('\n=== ROUND-TRIP SUMMARY ===');
console.log('BASE        :', before.slice(0, 160));
console.log('TUNED       :', after.slice(0, 160));
console.log('AFTER RELOAD:', afterReload.slice(0, 160));
console.log(failures === 0 ? '\nALL PASS · E2E_ROUNDTRIP_DONE' : `\n${failures} ASSERTION(S) FAILED · E2E_ROUNDTRIP_FAILED`);
await b.close();
process.exit(failures === 0 ? 0 : 1);
