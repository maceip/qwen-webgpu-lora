/* Verify the simplified installer-style UI loads with no JS errors and the
   right elements are present/absent. Screenshots both panes. */
import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 820, height: 1100 }, deviceScaleFactor: 2 });
const errs = [];
p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
p.on('pageerror', (e) => errs.push('PAGEERROR: ' + e.message));
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(500);

const r = await p.evaluate(() => {
  const has = (id) => !!document.getElementById(id);
  const vis = (id) => { const e = document.getElementById(id); if (!e) return false; const s = getComputedStyle(e); return s.display !== 'none' && s.visibility !== 'hidden'; };
  return {
    removed: { ctaTrain: has('ctaTrain'), laneGuided: has('laneGuided'), laneOwn: has('laneOwn'), guidedLane: has('guidedLane'), ownLane: has('ownLane') },
    present: { loadHF: has('loadHF'), hfRepo: has('hfRepo'), load: has('load'), modelFiles: has('modelFiles'), run: has('run'), trainGuided: has('trainGuided'), trainOwn: has('trainOwn'), ownText: has('ownText') },
    adapterHidden: !vis('adapterWrap'),
    askFolded: document.getElementById('askSection')?.classList.contains('folded'),
    askLockedVisible: vis('askLocked'),
  };
});
console.log('REMOVED(should all be false):', JSON.stringify(r.removed));
console.log('PRESENT(should all be true): ', JSON.stringify(r.present));
console.log('adapterHidden:', r.adapterHidden, '| askFolded:', r.askFolded, '| askLockedVisible:', r.askLockedVisible);
console.log('CONSOLE ERRORS:', errs.length ? JSON.stringify(errs) : 'none');

await p.locator('#paneInfer').screenshot({ path: '/tmp/simpl_infer.png' });
// open the advanced details to confirm it expands
await p.evaluate(() => document.querySelector('#paneInfer details.cfg')?.setAttribute('open', ''));
await p.waitForTimeout(150);
await p.locator('#paneInfer').screenshot({ path: '/tmp/simpl_infer_open.png' });
await p.evaluate(() => document.getElementById('tabTrain').click());
await p.waitForTimeout(200);
await p.locator('#paneTrain').screenshot({ path: '/tmp/simpl_train.png' });
await b.close();
console.log('SIMPLIFY_CHECK_DONE');
