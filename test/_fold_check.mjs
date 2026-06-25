import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 740, height: 760 }, deviceScaleFactor: 2 });
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(400);
const s = await p.evaluate(() => ({
  folded: document.getElementById('askSection').classList.contains('folded'),
  lockShown: getComputedStyle(document.getElementById('askLocked')).display !== 'none',
  askH: Math.round(document.getElementById('askSection').getBoundingClientRect().height),
}));
console.log('FOLD', JSON.stringify(s));
await p.locator('#paneInfer').screenshot({ path: '/tmp/fold.png' });
await b.close();
console.log('FOLD_DONE');
