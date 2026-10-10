"""Paper figures and their rate tables, from saved results only; never reruns model experiments.

  python -m soar_nla.icl_edits.figures [--results DIR] [--out DIR] [--only NAME ...] [--formats png pdf]

Reads <results>/<config>/summary.json (run `analysis` first) and the per-row result files.
Figures (written to <results>/figures/ by default):
  redirect_raw    icl_redirect_raw_L32            ICL edits, layer 32, raw text (main result)
  donor_sweep     donor_sweep_raw_L32[_by_task]   donor patching across layers, raw text (appendix)
  redirect_layers icl_redirect_raw_L24 / _L40 / _L24_L40   custom NLAs at layers 24 and 40 (appendix)
  redirect_chat   icl_redirect_chat_L32           ICL edits with the chat template (appendix)
  task_switching  task_switch_raw_L32 / task_switch_chat_L32 / task_switch_raw_chat_L32
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soar_nla.paths import results_dir

from .analysis import load_icl
from .config import load_config, pinned_revision
from .jsonio import read_json
from .figure_template import (FORMATS, METHODS, TASKS, apply_publication_style, render_layer_comparison,
                              render_stacked_task_figure, render_task_figure, save)

MAIN = 'gemma3-12b'
LAYER_CONFIGS = {24: 'gemma3-12b-custom-L24', 40: 'gemma3-12b-custom-L40'}


def _summary(res, name):
    return json.loads((res / name / 'summary.json').read_text())


def _provenance(cfg):
    return {'target': cfg.target, 'target_revision': pinned_revision(cfg.target),
            'av': cfg.av, 'av_revision': pinned_revision(cfg.av),
            'ar': cfg.ar, 'ar_revision': pinned_revision(cfg.ar),
            'backend': cfg.backend, 'layer': cfg.layer}


# ------------------------------------------------------------ layer 32, raw text
def redirect_raw(res, out, formats):
    """Six ICL tasks at the official layer-32 NLA, raw text, single token vs last four tokens."""
    s = _summary(res, MAIN)
    render_task_figure(s['icl_redirect_raw']['per_task'], out/'icl_redirect_raw_L32', formats=formats)
    print(out/'icl_redirect_raw_L32.png')


def donor_sweep(res, out, formats):
    """Raw-text ICL donor-patching layer sweep."""
    cfg = load_config(MAIN)
    apply_publication_style()
    sweep = _summary(res, MAIN)['icl_redirect_raw']['sweep']
    edit_rows, rows = load_icl(res/MAIN, 'redirect', 'raw')
    trials = Counter((r['span'], r['layer']) for r in rows)
    assert len(set(trials.values())) == 1, 'unequal trial counts across the sweep'
    n_trials = next(iter(trials.values()))
    n_tasks = len({r['task'] for r in rows})
    n_items = len({(r['task'], r['idx']) for r in rows}) // n_tasks
    n_seeds = len({r['seed'] for r in rows})
    fig,ax=plt.subplots(figsize=(12,6))
    for span,label,color,marker,style in [
        ('query','Query token','#009E73','s','--'),
        ('last1','Final token (:)','#D55E00','^','-.'),
        ('last4','Last four tokens','#0072B2','o','-')]:
        pts=sorted((int(k.split('|')[1]),v) for k,v in sweep.items() if k.split('|')[0]==span)
        ax.plot([p[0] for p in pts],[p[1] for p in pts],label=label,
                color=color,marker=marker,linestyle=style,linewidth=2.5,markersize=7)
    ax.axvline(cfg.layer,color='#767676',linestyle=':',linewidth=1.5,zorder=0)
    ax.text(cfg.layer+.5,.12,'NLA layer',rotation=90,color='#767676',fontsize=11)
    ax.set_xlabel('Patched layer')
    ax.set_ylabel('Donor target answer rate')
    ax.set_xticks(list(cfg.sweep_layers))
    ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylim(-.025,1.07)
    ax.set_xlim(-1,cfg.n_layers)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3)
    fig.tight_layout()
    save(fig, out/'donor_sweep_raw_L32', formats)
    (out/'donor_sweep_raw_L32_caption.txt').write_text(
        'Donor patching on Gemma-3-12B raw-text ICL prompts without a chat template. '
        'Each curve shows the target answer rate after replacing the indicated positions with '
        'donor activations at each sampled layer. Results are averaged over '
        f'{n_tasks} tasks, {n_items} items per task and {n_seeds} seeds '
        f'({n_trials} trials per layer and span). '
        f'The vertical line marks layer {cfg.layer}, used in the NLA experiments. '
        f'These sweep results use {n_seeds} seeds, whereas the layer-{cfg.layer} edit comparison uses '
        f"{len({r['seed'] for r in edit_rows})}.\n")
    print(out/'donor_sweep_raw_L32.png')

    # Disaggregate the sweep by task. Final-token patching gives one curve/task.
    acc=defaultdict(list)
    for r in rows:
        if r['span']=='last1':
            acc[(r['task'],r['layer'])].append(r['hit'])
    tasks=[('antonym','Antonym','#0072B2','o','-'),
           ('capitalize','Capitalize','#D55E00','s','--'),
           ('country-capital','Country → capital','#009E73','^','-.'),
           ('english-french','English → French','#CC79A7','D',':'),
           ('present-past','Present → past','#E69F00','v','--'),
           ('singular-plural','Singular → plural','#000000','P','-.')]
    fig,ax=plt.subplots(figsize=(12,6))
    for task,label,color,marker,style in tasks:
        layers=sorted(L for t,L in acc if t==task)
        rates=[sum(acc[task,L])/len(acc[task,L]) for L in layers]
        ax.plot(layers,rates,label=label,color=color,marker=marker,
                linestyle=style,linewidth=2.3,markersize=7)
    ax.axvline(cfg.layer,color='#767676',linestyle=':',linewidth=1.5,zorder=0)
    ax.set_xlabel('Patched layer')
    ax.set_ylabel('Donor target answer rate')
    ax.set_xticks(list(cfg.sweep_layers))
    ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylim(-.025,1.07);ax.set_xlim(-1,cfg.n_layers)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=3,fontsize=11)
    fig.tight_layout()
    save(fig, out/'donor_sweep_raw_L32_by_task', formats)
    (out/'donor_sweep_raw_L32_by_task_caption.txt').write_text(
        'Final-token donor patching on raw-text ICL prompts without a chat template. '
        f'Each line is one task, averaged over {n_items} items and {n_seeds} seeds per layer. '
        f'The patched position is the final colon; the dotted vertical line marks layer {cfg.layer}.\n')
    print(out/'donor_sweep_raw_L32_by_task.png')


# ------------------------------------------------------- layers 24 and 40, raw text
LAYER_SPANS=[('query','Query token (single)','.'),('last1','Final token (single)','//'),('last4','Four tokens',None)]


def _read_layer(res, name):
    rows,done=[],[]
    for tag in ['a','b']:
        d=read_json(res/name/f'icl_redirect_raw-{tag}.json')
        rows+=d['rows'];done+=d['done']
    assert set(done)=={t for t,_ in TASKS},f'{name}: incomplete tasks'
    assert len(rows)==600,(name,len(rows))
    assert len({(r['task'],r['idx'],r['seed']) for r in rows})==600
    stats={}
    for task,_ in TASKS:
        rr=[r for r in rows if r['task']==task]
        assert len(rr)==100 and {r['seed'] for r in rr}==set(range(10))
        stats[task]={sp:{a:float(np.mean([r[f'{sp}|{a}|hit'] for r in rr]))
                           for a,_,_ in METHODS} for sp,_,_ in LAYER_SPANS}
    stats['overall']={sp:{a:float(np.mean([stats[t][sp][a] for t,_ in TASKS]))
                         for a,_,_ in METHODS} for sp,_,_ in LAYER_SPANS}
    stats['controls']={'native_gold_rate':float(np.mean([r['baseline|gold'] for r in rows])),
        **{sp:{'roundtrip_gold_rate':float(np.mean([r[f'{sp}|roundtrip|gold'] for r in rows])),
               'roundtrip_target_rate':float(np.mean([r[f'{sp}|roundtrip|hit'] for r in rows])),
               'noop_target_rate':float(np.mean([r[f'{sp}|noop|hit'] for r in rows])),
               'noop_target_hits':int(sum(r[f'{sp}|noop|hit'] for r in rows)),
               'trials':len(rows)}
           for sp,_,_ in LAYER_SPANS}}
    return stats


def _layer_prose(layer,s):
    o=s['overall']
    rates=lambda sp:', '.join(f"{o[sp][a]:.3f}" for a in ['substitute_only','rewrite_only','edited'])
    if layer==24:
        return (f"At layer 24, query-and-answer substitutions redirect answers reliably when patched "
            f"at the query token, reaching {o['query']['substitute_only']:.3f}, close to donor patching "
            f"({o['query']['donor']:.3f}). Adding the final-unit rewrite reduces the target rate to "
            f"{o['query']['edited']:.3f}, while the rewrite alone reaches {o['query']['rewrite_only']:.3f}. "
            f"Four-token patching shows the same ordering: substitutions reach {o['last4']['substitute_only']:.3f}, "
            f"combined edits {o['last4']['edited']:.3f}, and the rewrite alone {o['last4']['rewrite_only']:.3f}. "
            f"Patching the final token is nearly ineffective, with combined edits reaching "
            f"{o['last1']['edited']:.3f}. Thus, NLA edits can redirect the answer at the early query position, "
            f"and the standardized prediction clause lowers aggregate performance at this layer.")
    return (f"At layer 40, NLA edits redirect answers at the end of the prompt, while query-token "
        f"patching produces no target answers. At the final token, substitutions, final-unit rewrites, "
        f"and combined edits reach {rates('last1')}, respectively, compared with "
        f"{o['last1']['donor']:.3f} for donor patching. Four-token patching raises these rates to "
        f"{rates('last4')}, with combined edits approaching the donor rate of {o['last4']['donor']:.3f}. "
        f"Capitalize remains weak under single-token combined edits ({s['capitalize']['last1']['edited']:.2f}) "
        f"despite successful donor patching ({s['capitalize']['last1']['donor']:.2f}), and improves to "
        f"{s['capitalize']['last4']['edited']:.2f} with four-token edits. At this layer, patching multiple "
        f"positions improves NLA redirection even though the final token alone is sufficient for the donor.")


def _controls_prose(all_stats):
    """Controls paragraph, computed from the per-row results."""
    c={L:s['controls'] for L,s in all_stats.items()}
    spans=[sp for sp,_,_ in LAYER_SPANS]
    native=sorted({round(x['native_gold_rate'],3) for x in c.values()})
    rt_max=max(x[sp]['roundtrip_target_rate'] for x in c.values() for sp in spans)
    noop=[(L,sp,x[sp]['noop_target_hits'],x[sp]['trials'],x[sp]['noop_target_rate'])
          for L,x in c.items() for sp in spans if x[sp]['noop_target_hits']]
    span_name={'query':'query-token','last1':'final-token','last4':'four-token'}
    text=('The unpatched model gives the base answer in '
          + ' and '.join(f'{v:.3f}' for v in native) + ' of trials. ')
    text+=('Round-trip target rates are zero at both layers and all three evaluated spans. ' if rt_max==0 else
           f'Round-trip target rates reach at most {rt_max:.4f}. ')
    if noop:
        text+=('No-op target rates are zero except for '
               + '; '.join(f'{h} of {n} {span_name[sp]} trials at layer {L} ({r:.4f})' for L,sp,h,n,r in noop)
               + '. ')
    else:
        text+='No-op target rates are zero at both layers and all three evaluated spans. '
    text+='Round-trip base-answer retention ranges ' + ' and '.join(
        f"from {min(x[sp]['roundtrip_gold_rate'] for sp in spans):.3f} to "
        f"{max(x[sp]['roundtrip_gold_rate'] for sp in spans):.3f} at layer {L}" for L,x in c.items()) + '.'
    return text


def redirect_layers(res, out, formats):
    """Validate the layer-24/40 raw-text runs (custom NLAs); export figures, rates and prose."""
    cfgs={L:load_config(name) for L,name in LAYER_CONFIGS.items()}
    all_stats={L:_read_layer(res,cfg.name) for L,cfg in cfgs.items()}
    provenance={'nla':{L:_provenance(cfg) for L,cfg in cfgs.items()},
        'format':'raw, without chat template','items_per_task':10,'seeds_per_item':10,
        'av_draws_per_position':3,'selection':'text-only edits.icl_select_score',
        'activation_norm':'AR output rescaled to native residual norm',
        'patch_timing':'decoder-block output during prefill only','layers':all_stats}
    out.mkdir(parents=True,exist_ok=True)
    (out/'icl_redirect_raw_L24_L40_rates.json').write_text(json.dumps(provenance,indent=2))
    lines=['# Raw-text ICL NLA edits at layers 24 and 40','',
        'Both layers use custom EasyNLA pairs trained with the same recipe. '
        'The layer-32 figure uses the official released pair, so differences with that figure '
        'can reflect both layer and NLA training. Each layer has six tasks, 10 items per task, and '
        '10 seeds per item. The three edits and donor are evaluated at the query token, final token, '
        'and last four tokens; all prompts are raw text without a chat template.', '']
    for layer,s in all_stats.items():
        render_task_figure(s,out/f'icl_redirect_raw_L{layer}',spans=LAYER_SPANS,formats=formats)
        lines += [f'## Layer {layer}','',_layer_prose(layer,s),'',f'![Layer {layer}](icl_redirect_raw_L{layer}.png)','']
    render_layer_comparison(all_stats,out/'icl_redirect_raw_L24_L40',spans=LAYER_SPANS,formats=formats)
    lines += ['## Controls','',_controls_prose(all_stats),'']
    (out/'icl_redirect_raw_L24_L40.md').write_text('\n'.join(lines))
    print('Validated 600 trials per layer;', out/'icl_redirect_raw_L24_L40.png')


# --------------------------------------------------------- layer 32, chat template
CHAT_SPANS = [('last1', 'Single final token', '//'),
              ('last4', 'Four header tokens', None)]


def _chat_paragraph(chat, raw):
    return (
        'The chat-template results show a larger benefit from the final-paragraph rewrite '
        'than the raw-text results. With the official layer-32 NLA, single-token '
        f"query-and-answer substitutions reach {chat['last1']['substitute_only']:.3f} with "
        f"the template, compared with {raw['last1']['substitute_only'][0]:.3f} in raw text; "
        f"adding the rewrite raises these rates to {chat['last1']['edited']:.3f} and "
        f"{raw['last1']['edited'][0]:.3f}. For four-token patching, substitutions reach "
        f"{chat['last4']['substitute_only']:.3f} with the template and "
        f"{raw['last4']['substitute_only'][0]:.3f} in raw text, while combined edits reach "
        f"{chat['last4']['edited']:.3f} and {raw['last4']['edited'][0]:.3f}. "
        'Final-paragraph rewrites alone have lower average rates than lexical substitutions '
        'in both formats. Donor patching is also less effective with the template, reaching '
        f"{chat['last1']['donor']:.3f} for one token and {chat['last4']['donor']:.3f} for four, "
        f"compared with {raw['last1']['donor'][0]:.3f} and {raw['last4']['donor'][0]:.3f} "
        'in raw text. The four-token spans differ across formats: raw text includes the final '
        'query token and answer prefix, whereas the chat format uses the assistant header.'
    )


def redirect_chat(res, out, formats):
    """Validate and plot the official layer-32 ICL edits with the chat template."""
    cfg = load_config(MAIN)
    summaries = _summary(res, MAIN)
    summary = summaries['icl_redirect']
    assert summary['format'] == 'chat'
    rows, done = [], []
    for tag in ['a', 'b']:
        shard = read_json(res / MAIN / f'icl_redirect-{tag}.json')
        rows += shard['rows']
        done += shard['done']
    assert len(rows) == 600
    assert len({(row['task'], row['idx'], row['seed']) for row in rows}) == 600
    assert set(done) == {task for task, _ in TASKS}

    stats = {}
    for task, _ in TASKS:
        task_rows = [row for row in rows if row['task'] == task]
        assert len(task_rows) == 100
        assert {row['seed'] for row in task_rows} == set(range(10))
        assert len({row['idx'] for row in task_rows}) == 10
        stats[task] = {
            span: {arm: float(np.mean([row[f'{span}|{arm}|hit'] for row in task_rows]))
                   for arm, _, _ in METHODS}
            for span, _, _ in CHAT_SPANS
        }
        for span, _, _ in CHAT_SPANS:
            for arm, _, _ in METHODS:
                assert np.isclose(stats[task][span][arm],
                                  summary['per_task'][task][span][arm], atol=1e-12)
    overall = {
        span: {arm: float(np.mean([stats[task][span][arm] for task, _ in TASKS]))
               for arm, _, _ in METHODS}
        for span, _, _ in CHAT_SPANS
    }
    for span, _, _ in CHAT_SPANS:
        for arm, _, _ in METHODS:
            assert np.isclose(overall[span][arm], summary[span][arm][0], atol=1e-12)
    controls = {
        'native_gold_rate': float(np.mean([row['baseline|gold'] for row in rows])),
        **{span: {arm: float(np.mean([row[f'{span}|{arm}|hit'] for row in rows]))
                  for arm in ['roundtrip', 'noop']}
           for span, _, _ in CHAT_SPANS}
    }

    render_task_figure(stats, out / 'icl_redirect_chat_L32', spans=CHAT_SPANS, formats=formats)
    metadata = {
        **_provenance(cfg), 'pair': 'official released NLA', 'format': 'chat',
        'trials': len(rows), 'items_per_task': 10, 'seeds_per_item': 10,
        'sources': [f'{MAIN}/icl_redirect-a.json', f'{MAIN}/icl_redirect-b.json'],
        'spans': {span: name for span, name, _ in CHAT_SPANS},
        'per_task': stats, 'overall': overall, 'controls': controls,
    }
    (out / 'icl_redirect_chat_L32_rates.json').write_text(json.dumps(metadata, indent=2))
    (out / 'icl_redirect_chat_L32.md').write_text(
        _chat_paragraph(overall, summaries['icl_redirect_raw']) + '\n')
    print('Validated 600 chat trials;', out / 'icl_redirect_chat_L32.png')


# ------------------------------------------------------------------ task switching
SWITCH_TASKS = [('french->spanish', 'French → Spanish'),
                ('capital->currency', 'Capital → currency'),
                ('antonym->comparative', 'Antonym → comparative')]
SWITCH_METHODS = [('answer_only', 'Answer only', '#999999'),
                  ('task_only', 'Task only', '#CC79A7'),
                  ('answer+task', 'Answer + task', '#0072B2'),
                  ('edited', 'Full edit (+ final-unit rewrite)', '#009E73'),
                  ('donor', 'Donor', '#E69F00')]
SWITCH_SPANS = [('last1', 'Single final token', '//'), ('last4', 'Four tokens', None)]


def _read_switch(res, fmt, summary):
    stem = 'icl_switch_raw' if fmt == 'raw' else 'icl_switch'
    data = read_json(res / MAIN / f'{stem}.json')
    rows = data['rows']
    assert len(rows) == 300
    assert len({(r['task'], r['idx'], r['seed']) for r in rows}) == 300
    assert set(data['done']) == {task for task, _ in SWITCH_TASKS}
    assert summary[stem]['format'] == fmt
    stats = {}
    for task, _ in SWITCH_TASKS:
        task_rows = [r for r in rows if r['task'] == task]
        assert len(task_rows) == 100
        assert {r['seed'] for r in task_rows} == set(range(10))
        assert len({r['idx'] for r in task_rows}) == 10
        stats[task] = {
            span: {arm: float(np.mean([r[f'{span}|{arm}|hit'] for r in task_rows]))
                   for arm, _, _ in SWITCH_METHODS}
            for span, _, _ in SWITCH_SPANS
        }
        for span, _, _ in SWITCH_SPANS:
            for arm, _, _ in SWITCH_METHODS:
                assert np.isclose(stats[task][span][arm],
                                  summary[stem]['per_task'][task][span][arm], atol=1e-12)
    overall = {
        span: {arm: float(np.mean([stats[task][span][arm] for task, _ in SWITCH_TASKS]))
               for arm, _, _ in SWITCH_METHODS}
        for span, _, _ in SWITCH_SPANS
    }
    for span, _, _ in SWITCH_SPANS:
        for arm, _, _ in SWITCH_METHODS:
            assert np.isclose(overall[span][arm], summary[stem][span][arm][0], atol=1e-12)
    controls = {
        'native_gold_rate': float(np.mean([r['baseline|gold'] for r in rows])),
        **{span: {arm: float(np.mean([r[f'{span}|{arm}|hit'] for r in rows]))
                  for arm in ['roundtrip', 'noop']}
           for span, _, _ in SWITCH_SPANS}
    }
    return {'per_task': stats, 'overall': overall, 'controls': controls,
            'trials': len(rows), 'source': f'{MAIN}/{stem}.json'}


def _switch_paragraph(results):
    raw, chat = (results[fmt]['overall'] for fmt in ['raw', 'chat'])
    max_single_edit = max(r[span][arm] for r in [raw, chat]
                          for span, _, _ in SWITCH_SPANS for arm in ['answer_only', 'task_only'])
    controls_zero = all(results[fmt]['controls'][span][arm] == 0 for fmt in results
                        for span, _, _ in SWITCH_SPANS for arm in ['roundtrip', 'noop'])
    trials = sorted({results[fmt]['trials'] for fmt in results})
    return (
        'For task switching, combining answer and task substitutions with a final-paragraph '
        'rewrite gives higher average target rates with the chat template than in raw text. '
        'Across French-to-Spanish '
        'translation, capital-to-currency, and antonym-to-comparative switches, answer-only '
        f'or task-only edits give average target rates no higher than {max_single_edit:.3f}. '
        'Editing both the answer and task words reaches '
        f"{raw['last1']['answer+task']:.3f} with single-token patching and "
        f"{raw['last4']['answer+task']:.3f} with four-token patching in raw text, compared with "
        f"{chat['last1']['answer+task']:.3f} and {chat['last4']['answer+task']:.3f} with the chat "
        'template. Adding the final-paragraph rewrite raises the raw-text rates to '
        f"{raw['last1']['edited']:.3f} and {raw['last4']['edited']:.3f}, and the chat-template "
        f"rates to {chat['last1']['edited']:.3f} and {chat['last4']['edited']:.3f}. "
        'With four-token patching, combined edits approach the donor rate in raw text '
        f"({raw['last4']['donor']:.3f}) and exceed it with the chat template "
        f"({chat['last4']['donor']:.3f}). "
        + ('Round-trip and no-op controls produce no target answers across '
           f"{' / '.join(map(str, trials))} trials per condition."
           if controls_zero else 'Round-trip or no-op controls produce some target answers; see the rates file.')
    )


def task_switching(res, out, formats):
    """Validate and plot official layer-32 task-switching edits in raw and chat formats."""
    cfg = load_config(MAIN)
    summary = _summary(res, MAIN)
    results = {fmt: _read_switch(res, fmt, summary) for fmt in ['raw', 'chat']}
    for fmt, result in results.items():
        spans = [('last1', 'Single final token', '//'),
                 ('last4', 'Four header tokens' if fmt == 'chat' else 'Four tokens', None)]
        render_task_figure(result['per_task'], out / f'task_switch_{fmt}_L32',
                           spans=spans, tasks=SWITCH_TASKS, methods=SWITCH_METHODS, xlabel='Task switch',
                           formats=formats)
    render_stacked_task_figure(
        {'Raw text': results['raw']['per_task'],
         'Chat template': results['chat']['per_task']},
        out / 'task_switch_raw_chat_L32', spans=SWITCH_SPANS,
        tasks=SWITCH_TASKS, methods=SWITCH_METHODS, xlabel='Task switch', formats=formats)
    metadata = {
        **_provenance(cfg), 'pair': 'official released NLA', 'items_per_task': 10, 'seeds_per_item': 10,
        'methods': {arm: name for arm, name, _ in SWITCH_METHODS},
        'full_edit': 'answer substitutions + task-description substitutions + final-paragraph rewrite',
        'spans': {span: name for span, name, _ in SWITCH_SPANS},
        'patch_positions': {'raw': 'final colon, or final query token plus newline/A/colon',
                            'chat': 'final header token, or all four assistant-header tokens'},
        'results': results,
    }
    (out / 'task_switch_L32_rates.json').write_text(json.dumps(metadata, indent=2))
    text = _switch_paragraph(results)
    (out / 'task_switch_L32.md').write_text(text + '\n')
    print('Validated 300 trials per format;', out / 'task_switch_raw_L32.png')


FIGURES = {'redirect_raw': redirect_raw, 'donor_sweep': donor_sweep, 'redirect_layers': redirect_layers,
           'redirect_chat': redirect_chat, 'task_switching': task_switching}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='soar_nla.icl_edits.figures')
    ap.add_argument('--results', type=Path, default=results_dir('icl_edits'),
                    help='root holding one directory per config')
    ap.add_argument('--out', type=Path, default=None, help='default: <results>/figures')
    ap.add_argument('--only', nargs='*', choices=list(FIGURES), default=list(FIGURES))
    ap.add_argument('--formats', nargs='+', default=list(FORMATS))
    a = ap.parse_args(argv)
    out = a.out or a.results / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for name in a.only:
        FIGURES[name](a.results, out, tuple(a.formats))


if __name__ == '__main__':
    main()
