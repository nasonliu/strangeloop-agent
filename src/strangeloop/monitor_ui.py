"""Local, dependency-free UI served by :mod:`strangeloop.monitor`."""

from __future__ import annotations


def dashboard_html() -> str:
    """Return a self-contained dashboard.

    All dynamic values are assigned through ``textContent``; the page never
    interprets ledger data as HTML and makes no network request except to its
    own local JSON endpoints.
    """
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strangeloop cognitive monitor</title><style>
:root{color-scheme:dark;--bg:#101419;--card:#1b232c;--ink:#edf3f7;--muted:#a9b7c2;--line:#344451;--accent:#75d6a4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}.wide{grid-column:1/-1}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}h1{margin:0 0 4px;font-size:24px}h2{font-size:15px;margin:0 0 10px;color:var(--accent)}p{margin:6px 0}.muted{color:var(--muted)}ul{padding-left:20px;margin:6px 0;max-height:250px;overflow:auto}.event{border-top:1px solid var(--line);padding:7px 0}.status{float:right;color:var(--accent)}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}@media(max-width:500px){main{padding:12px}h1{font-size:20px}}</style></head>
<body><main><h1>Strangeloop cognitive monitor <span id="health" class="status">connecting</span></h1>
<p class="muted">Local inspection surface for observable software records. Hidden chain-of-thought is neither stored nor shown.</p>
<section class="grid">
<article class="card"><h2>Public goal and turn</h2><p id="goal">Loading…</p><p class="muted" id="turn"></p></article>
<article class="card"><h2>DMN loop and budgets</h2><pre id="loop">Loading…</pre></article>
<article class="card"><h2>Unattended research</h2><p class="muted">Read-only public-HTTPS/repository profile telemetry. A report is represented only by digest/count; no focus text, prompt, URL, path, page body, finding text, cookie, key, or hidden reasoning is shown.</p><pre id="unattended">Loading…</pre></article>
<article class="card"><h2>Expedition / frontier learning</h2><p class="muted">Bounded expedition telemetry only. Frontier learning shows only mode, spec digest, fixed ranking reason, a bounded opaque strategy-arm digest/version, post-reward selection count/reason, duplicate-result count, duplicate suppression and aggregate reward/TD counts. `task_terminal_coverage` is scheduler-terminal bookkeeping, not quality or research progress. A strategy arm credits only its authorized experiment kind, never a task instance. Hard gates still run first; there is no cross-run persistence. Offline experiments show only fixed kind/status/count/reproducible/control-valid/result digest/reward qualification. No goal, query, URL, candidate, task ID, hypothesis, raw metric, seed, path, output, page material, planner text, vector values, credentials, or hidden reasoning is shown.</p><pre id="expedition">Loading…</pre></article>
<article class="card"><h2>K3 quota / conservation</h2><p class="muted">Provider-reported K3 Code Plan capacity when available. Resource conservation is not a reward, drive, or consciousness signal; local observed calls/tokens are not the plan balance.</p><pre id="quota">Loading…</pre></article>
<article class="card"><h2>Quota sleep / wake</h2><p class="muted">Resource-management state only; not biological sleep, consciousness, or survival.</p><pre id="sleep">Loading…</pre><ul id="archives"></ul></article>
<article class="card"><h2>K3 model invocation</h2><ul id="models"></ul></article>
<article class="card"><h2>Tool authorization stages</h2><ul id="tools"></ul></article>
<article class="card"><h2>Seeds and self-model claims</h2><ul id="claims"></ul></article>
<article class="card"><h2>自证／证自证镜映</h2><p class="muted">固定深度的可审计互证；不显示推理文本。</p><ul id="mirrors"></ul></article>
<article class="card"><h2>TD / RPE and stops</h2><ul id="learning"></ul></article>
<article class="card wide"><h2>Public event timeline</h2><ul id="events"></ul></article></section>
</main><script>
const byId=id=>document.getElementById(id), put=(id,items)=>{const el=byId(id);el.replaceChildren();(items||[]).forEach(item=>{const li=document.createElement('li');li.textContent=typeof item==='string'?item:JSON.stringify(item);el.appendChild(li)})};
function line(e){const p=e.payload||{},summary=p.public_summary||p.summary||p.outcome||'';return '#'+e.sequence+' '+e.kind+(summary?' — '+summary:'')}
function mirrorLine(e){const p=e.payload||{},codes=Array.isArray(p.check_codes)?p.check_codes.join(', '):'—';return '#'+e.sequence+' '+(p.mirror_id||'—')+' | self='+ (p.self_status||'—')+' @ '+(p.self_confidence??'—')+' | meta='+(p.meta_status||'—')+' cap='+(p.meta_confidence_cap??'—')+' | '+(p.disposition||'—')+' | checks: '+codes}
function archiveLine(a){return (a.archive_id||'—')+' | epoch '+(a.epoch_id||'—')+' | digest '+(a.archive_digest||'—')+' | events '+(a.event_count??'—')+' | head '+(a.chain_head_hash||'—')+' | manifest '+(a.schema_version??'—')+'/'+(a.chain_head_sequence??'—')+' | quota '+(a.quota_remaining??'—')+'/'+(a.quota_total??'—')+' | window '+(a.quota_window_kind||'—')}
function quotaLine(q){const balance=(Number.isInteger(q.remaining)&&Number.isInteger(q.total))?q.remaining+'/'+q.total+' '+(q.primary_unit||'units'):'unknown';const primaryWindow=(q.primary_window_kind==='weekly'||q.primary_window_kind==='rolling_5h')?q.primary_window_kind:'unknown';return 'K3 Plan remaining: '+balance+'\\nprimary window: '+primaryWindow+'\\nreset: '+(q.reset_at||'unknown')+'\\nobserved: '+(q.observed_at||'unknown')+'\\nauthority/freshness: '+(q.authority||'unknown')+' / '+(q.freshness||'unknown')+'\\nallow call: '+(q.allow_call===true?'yes':'no')+'\\nreason: '+(q.reason||'unknown')+'\\nlocal observed calls/tokens (not plan): '+(q.local_observed_calls??'—')+' / '+(q.local_observed_tokens??'—')}
async function refresh(){try{const [state,events,health]=await Promise.all([fetch('/api/state').then(r=>r.json()),fetch('/api/events?after_sequence=0').then(r=>r.json()),fetch('/api/health').then(r=>r.json())]);byId('health').textContent=health.status==='ok'?'local / healthy':'error';byId('goal').textContent=state.current_goal||'No public goal recorded.';byId('turn').textContent=state.current_turn?'Turn: '+state.current_turn:'No public turn recorded.';byId('loop').textContent=JSON.stringify(state.dmn_loop||{},null,2);byId('unattended').textContent=JSON.stringify(state.unattended||{},null,2);byId('expedition').textContent=JSON.stringify(state.expedition||{},null,2);byId('quota').textContent=quotaLine(state.quota||{});const sleep=state.sleep_wake||{};byId('sleep').textContent=JSON.stringify(sleep.coordinator||{},null,2);put('archives',(sleep.archives||[]).map(archiveLine));put('models',(state.model_invocations||[]).map(line));put('tools',(state.tool_stages||[]).map(line));put('claims',(state.seeds_and_claims||[]).map(line));put('mirrors',(state.metacognitive_mirrors||[]).map(mirrorLine));put('learning',(state.learning_and_stops||[]).map(line));put('events',(events.events||[]).map(line))}catch(e){byId('health').textContent='unavailable'}}refresh();setInterval(refresh,1500);
</script></body></html>"""
