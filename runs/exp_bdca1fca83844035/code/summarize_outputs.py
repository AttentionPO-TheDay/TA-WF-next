#!/usr/bin/env python3
"""Render auditable reports from frozen diagnostic CSV/JSON outputs."""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"runs/exp_bdca1fca83844035"
raw=json.loads((RUN/"artifacts/summary_raw.json").read_text())

def read(name): return list(csv.DictReader((RUN/name).open()))
def f(x,n=3): return f"{float(x):.{n}f}"
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as z:
  while q:=z.read(8*1024*1024): h.update(q)
 return h.hexdigest()

joint=read("stability_discriminability_joint.csv")
primary={r['attribute']:r for r in joint if r['date']=='ALL'}
eq=read("artifacts/equal500_joint.csv"); equal={r['attribute']:r for r in eq if r['date']=='ALL'}
pos=read("position_direction_diagnostic.csv"); ps=read("artifacts/position_equal500_sensitivity.csv")

# Primary position stability and a matched-date different-site table.
position_joint=[]
for key in sorted({(r['coordinate'],r['bins'],r['bin'],r['metric'],r['date']) for r in pos}):
 coord,bins,bin_,metric,date=key; rr=[r for r in pos if (r['coordinate'],r['bins'],r['bin'],r['metric'],r['date'])==key and r['exceeds_day0_q95']!='']
 centers=[float(r['future_center']) for r in rr]; between=[]
 for i,a in enumerate(centers): between.extend(abs(a-z) for z in centers[i+1:])
 same=sorted(float(r['temporal_abs_shift']) for r in rr); between.sort()
 med=lambda a: a[len(a)//2] if a else float('nan')
 position_joint.append({'coordinate':coord,'bins':bins,'bin':bin_,'metric':metric,'date':date,'sites':len(rr),
  'median_same_site_temporal_shift':med(same),'median_different_site_distance':med(between),
  'separability_ratio':med(between)/max(med(same),1e-9),'fraction_not_exceeding_day0_q95':sum(int(r['exceeds_day0_q95'])==0 for r in rr)/len(rr)})
with (RUN/'artifacts/position_stability_discriminability_joint.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(position_joint[0])); w.writeheader(); w.writerows(position_joint)

# The pre-registered equal-500 pattern used for the follow-up gate.
def cell(scenario,coord,metric,bin_):
 return [r for r in ps if r['scenario']==scenario and r['coordinate']==coord and r['metric']==metric and r['bin']==str(bin_)]
c0=cell('equal500','raw_packet_index','cover_run_length_mass_weighted',1)
c1=cell('equal500_exclude_terminal','raw_packet_index','cover_run_length_mass_weighted',1)
candidate={'coordinate':'raw_packet_index','packet_indices':'50..99 (zero based)','metric':'cover_run_length_mass_weighted','common_sites':96,
 'retained_terminal':{'stable_fraction':sum(float(r['fraction_not_exceeding_day0_q95']) for r in c0)/5,'separability_by_date':{r['date']:float(r['separability_ratio']) for r in c0}},
 'excluded_terminal':{'stable_fraction':sum(float(r['fraction_not_exceeding_day0_q95']) for r in c1)/5,'separability_by_date':{r['date']:float(r['separability_ratio']) for r in c1}}}

aud_lines=[]
for d in ('day0','day14','day30','day90','day150','day270'):
 a=raw['audits'][d]; q=a['valid_length_quantiles']
 aud_lines.append(f"| {d} | {a['raw_rows']:,} | {a['canonical_rows']:,} | {min(map(int,a['per_site_canonical'].values()))}–{max(map(int,a['per_site_canonical'].values()))} | {q[1]:.0f}/{q[3]:.0f}/{q[5]:.0f} | {a['full_L_count']:,} ({a['full_L_count']/a['canonical_rows']:.1%}) | {a['terminal_censored_count']:,} ({a['terminal_censored_count']/a['canonical_rows']:.1%}) |")
(RUN/'data_observation_boundary_audit.md').write_text(f"""# Data and observation-boundary audit

All hashes were checked against the formal v3 manifest. `train/day14/day30/day90/day150/day270` SHA-256 values are respectively `2994271…`, `eaa52ab…`, `16b8bd90…`, `eaa25c23…`, `9fbc3116…`, `35e965b9…`; `splits_v3.json` is `0f322e9a…`. Arrays are `float64 X[N,10000]` and `float64 y[N]`; labels are integral 0–101. Only `sign(X[:5000])` enters direction results.

| date | raw rows | canonical/analyzed | per-site n | valid length p10/p50/p90 | reaches L=5000 | right-censored terminal |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(aud_lines)}

Day0 uses exactly the 18,553-row union of the formal v3 source roles. Future date-internal admitted-input hashing removed zero rows and found zero cross-label conflicts. Every analyzed row had a nonempty finite prefix, zero leading/interior zero positions before its final nonzero value, and only a trailing zero suffix; therefore trailing zeros are treated as padding. Full-L traces are only 4.2–5.2% per date. At L, 3.2–4.2% of all traces continue in the same direction and are marked right-censored; another 0.9–1.3% have an observed direction transition immediately after L.

Class/date counts are not equal (per-site ranges above, notably Day150 max 496 and Day270 min 85). Primary aggregation is website-level, Day0 noise is estimated per website, and the fixed equal-500 sensitivity uses only the 96 websites represented by traces of length at least 500 in every date. The complete per-site counts are in `artifacts/summary_raw.json`.
""")

rt='\n'.join(f"| {d} | {raw['roundtrip'][d]['checked']:,} | {raw['roundtrip'][d]['failed']} |" for d in ('day0','day14','day30','day90','day150','day270'))
(RUN/'burst_rle_roundtrip_audit.md').write_text(f"""# Burst RLE round-trip audit

The encoder emits one signed integer per maximal equal-direction run; sign is direction and absolute value is exact run length. The decoder repeats each sign by its absolute length. This is reversible run-length coding of packet direction, not filtering or information removal.

| date | exact checks | failures |
|---|---:|---:|
{rt}

All 136,621 canonical traces passed exact element-wise decode(encode(x)) == x, total run length == valid packet length, nonzero run length, and alternating adjacent signs. Interpretation was therefore allowed to continue. Maximal-run direction alternation is mechanical and was not treated as an independent feature.
""")

top=sorted(primary.items(),key=lambda z:float(z[1]['fraction_not_exceeding_day0_q95']),reverse=True)
top_lines='\n'.join(f"| {a} | {f(r['fraction_not_exceeding_day0_q95'])} | {f(r['median_same_site_temporal_shift'])} | {f(r['median_different_site_distance'])} | {f(r['separability_ratio'])} |" for a,r in top[:8])
eq_lines='\n'.join(f"| {a} | {f(equal[a]['fraction_not_exceeding_day0_q95'])} | {f(equal[a]['separability_ratio'])} |" for a in ('outgoing_run_mean','adjacent_log_length_corr','burst_count','transition_density'))
(RUN/'boundary_length_sensitivity.md').write_text(f"""# Boundary and length sensitivity

| primary attribute (best stability first) | fraction within Day0 q95 | median same-site shift | median different-site distance | separation ratio |
|---|---:|---:|---:|---:|
{top_lines}

The two globally most stable attributes are degenerate: `run_median` has stability 1.000 but both same-site and different-site median distances are 0; outgoing run p90 has stability 0.914 but different-site median distance is also 0. No primary whole-trace attribute meets the pre-registered joint gate.

Removing the L-boundary terminal run leaves the leading rankings and values essentially unchanged. A full-L-only comparison is not valid across all sites: just 833–1,184 traces/date reach L and at least one website is absent. Day0 length-quartile edges are 580/1037/1738 packets. Fixed 500-packet analysis retains 14,876 Day0 and 16,393–23,048 future traces but only 96/102 websites in every date.

| equal-500 whole-prefix attribute | fraction within Day0 q95 | separation ratio |
|---|---:|---:|
{eq_lines}

The strongest non-mechanical positional result is raw packet indices 50–99 in the fixed 500-packet sensitivity: mass-weighted covering-run length stays within the website-specific Day0 q95 in {candidate['retained_terminal']['stable_fraction']:.3f} of website×date cells. Its different-site/same-site ratios are {', '.join(f'{d}={v:.2f}' for d,v in candidate['retained_terminal']['separability_by_date'].items())}. Removing each 500-prefix terminal run gives stability {candidate['excluded_terminal']['stable_fraction']:.3f} and the same ratios for this early window, so it is not caused by the terminal segment. This result covers 96 websites, not all 102.

Relative burst-order K=5/10/20 gives median site×date×bin shifts 0.0350/0.0338/0.0333, so the aggregate shift scale is not driven by choosing K=10. Nevertheless, burst percentiles are not webpage stages, and no ad/JS/async-loading interpretation is made. The fixed-index early-window result avoids burst-percentile bin mechanics; it remains descriptive because its stability ranking uses future labels.
""")

timelines=[]
for d in ('day0','day14','day30','day90','day150','day270'):
 t=raw['timing'][d]; timelines.append(f"| {d} | {t['monotonic_violations']/max(t['packets']-raw['audits'][d]['canonical_rows'],1):.3%} | {t['single_packet_burst_fraction']:.3%} | {t['zero_duration_burst_fraction']:.3%} | {t['gap_median']:.5f} | {t['finite_rate_median']:.2f} |")
(RUN/'timing_appendix.md').write_text(f"""# Timing appendix

Timing is separate from all direction-only conclusions. Formal project documentation identifies X as signed relative packet timestamp, and finite values are available. `abs(X)` nevertheless has small local reversals (table), so timing results are exploratory rather than a semantically clean primary layer.

| date | adjacent absolute-time reversals | single-packet bursts | duration <= 0 bursts | median trace-level inter-burst gap | median finite size/duration rate |
|---|---:|---:|---:|---:|---:|
{chr(10).join(timelines)}

Burst duration is `last_abs_timestamp - first_abs_timestamp`; inter-burst gap is `next_first - current_last`. Single-packet bursts have duration 0. Rates are calculated only where duration > 0; no infinite value is constructed. Zero-duration proportions (55.0–57.6%) are explicitly excluded from rate summaries. The near equality of single-packet and zero-duration fractions shows the exclusion is overwhelmingly the registered single-packet case (Day270 has one additional zero-duration multi-packet burst). Timing does not support or explain the direction-only burst-size finding.
""")

verdict='SUPPORTS_FOLLOWUP'
summary={
 'schema_version':1,'experiment':'exp_bdca1fca83844035','status':'completed','verdict':verdict,
 'scope':'training-free descriptive TemporalDrift development diagnostic; not a method contribution or external confirmation',
 'observation_length':5000,'canonical_traces':sum(raw['audits'][d]['canonical_rows'] for d in raw['audits']),
 'rle_roundtrip_failures':sum(raw['roundtrip'][d]['failed'] for d in raw['roundtrip']),
 'primary_whole_trace_candidate_attributes':raw['candidate_attributes_primary'],
 'followup_evidence':candidate,
 'limitations':['candidate ranking uses future true labels descriptively','equal-500 positional comparison covers 96/102 websites','full-L sensitivity lacks all-site coverage','timestamp monotonicity has small violations','no deployable weighting rule was learned'],
 'permissions':{'training_runs':0,'adaptation_runs':0,'gpu_used':False,'future_labels':'descriptive same-site/different-site comparisons only'},
 'artifacts':{}
}
(RUN/'RESULTS.md').write_text(f"""# Burst-Level Temporal Drift Characterization

## Verdict: {verdict}

This verdict means only that a tightly scoped burst-structure hypothesis is worth a later test. It does not establish a method, innovation, deployable selector, or external confirmation, and it does not authorize training.

Across 136,621 canonical traces, exact signed maximal-run RLE had 0 round-trip failures. Direction-only whole-trace attributes did not pass the joint gate: the stable attributes were non-discriminative (`run_median` and outgoing p90 have median different-site distance 0), while useful-looking attributes exceeded their website-specific Day0 q95 too often. Examples are outgoing-run mean (stability 0.741, separation 4.21), transition density (0.616, 6.75), and adjacent log-run correlation (0.645, 7.72).

The limited positive evidence is positional and length-controlled. On the pre-registered 500-packet common budget, 96/102 websites have usable traces in all dates. For raw indices 50–99, covering-run mass-weighted length stays within Day0 q95 in 87.7% of website×date cells, while different-site/same-site distance ratios are Day14 6.20, Day30 6.42, Day90 4.73, Day150 3.27, Day270 2.43. Removing the terminal/right-censored run leaves this early window unchanged. Related run-length and transition measures show the same broad early-prefix pattern; relative-order aggregate shift is similar for K=5/10/20. Thus the evidence is not explained solely by trailing padding, unequal full trace length, the final censored run, or choosing K=10.

The evidence is still bounded. Full-L traces are only 4.2–5.2%; full-L-only all-site comparison is impossible. The equal-500 gate omits six websites and late-date stability weakens. Same burst percentile is not the same loading stage, and no resource/JS/ad mechanism is inferred. Future labels were used to establish same-site stability and different-site separation, so the indices/attribute ranking is descriptive development evidence and cannot be copied into a deployment rule.

One follow-up hypothesis is permitted, but not implemented here: learn a fixed early-prefix run-organization reliability model from historical labeled data, then let only the current **unlabeled** batch estimate distributional reliability/calibration before combining it with a packet-sequence predictor. The window/rule must be selected without current true labels and compared against length-matched packet-level controls; the present future-label ranking is evaluation evidence, not deployable information.

Timing is available but secondary: absolute timestamps show 0.12–0.22% adjacent reversals, and 55.0–57.6% of bursts have zero duration (almost all single-packet). Rates exclude duration<=0 and contain no infinities. No timing result is attributed to signed burst-size RLE.

## Completion checks

- Training/adaptation/GPU runs: 0 / 0 / no.
- Input data, formal v3 split, checkpoints and prior artifacts: read-only; verified data/split hashes match formal records.
- Future labels: descriptive same-site/different-site statistics only; no weights, thresholds, selectors or methods fitted.
- Required tables/reports: complete. Detailed per-site shifts, joint metrics and position rows remain in CSV rather than being hidden by plots.
- No Packet CNN/Burst CNN/DF comparison and no follow-on training Job created.

Host decision remains open: this technical verdict supports only a bounded follow-up test, not adoption of the route.
""")
for p in sorted(RUN.glob('*.csv'))+sorted(RUN.glob('*.md'))+sorted((RUN/'artifacts').glob('*.csv')):
 summary['artifacts'][str(p.relative_to(RUN))]={'sha256':sha(p),'bytes':p.stat().st_size}
(RUN/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps({'verdict':verdict,'summary':str(RUN/'summary.json'),'candidate':candidate},indent=2))
