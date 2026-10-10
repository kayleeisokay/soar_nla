"""Shared style and layout for the ICL task figures (figures4papers house style)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

FORMATS = ('png', 'pdf')
TASKS=[('antonym','Antonym'),('capitalize','Capitalize'),('country-capital','Country → capital'),
       ('english-french','English → French'),('present-past','Present → past'),('singular-plural','Singular → plural')]
METHODS=[('substitute_only','Query + answer','#0072B2'),
         ('rewrite_only','Final-unit rewrite only','#CC79A7'),
         ('edited','Full edit (both)','#009E73'),('donor','Donor','#E69F00')]
MAIN_SPANS=[('last1','Single token','//'),('last4','Four tokens',None)]
SPAN_STRENGTH={'query': .45, 'last1': .72, 'last4': 1.0}


def apply_publication_style():
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'],
        'font.size': 13, 'axes.linewidth': 2, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig, out_path, formats=FORMATS):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_path.with_suffix('.'+fmt), dpi=300, facecolor='white')
    plt.close(fig)


def span_shade(color, span):
    strength = SPAN_STRENGTH[span]
    return tuple(1 - strength * (1 - channel) for channel in to_rgb(color))


def legend_handles(spans, shade_spans=False, methods=METHODS):
    handles = [Patch(facecolor=color, edgecolor='black', label=name)
               for _, name, color in methods]
    handles += [Patch(facecolor=span_shade('#444444', span) if shade_spans else 'white',
                      edgecolor='black', hatch=None if shade_spans else hatch,
                      label=name) for span, name, hatch in spans]
    return handles


def render_task_figure(stats,out_path,spans=MAIN_SPANS,shade_spans=False,
                       tasks=TASKS,methods=METHODS,xlabel='ICL task',formats=FORMATS):
    apply_publication_style()
    fig,ax=plt.subplots(figsize=(21,7))
    series=[(arm,span,color,hatch) for arm,_,color in methods for span,_,hatch in spans]
    width=.84/len(series)
    for i,(arm,span,color,hatch) in enumerate(series):
        vals=[stats[t][span][arm] for t,_ in tasks]
        vals.append(float(np.mean(vals)))
        x=np.arange(len(tasks)+1)+(i-(len(series)-1)/2)*width
        ax.bar(x,vals,width=width,color=span_shade(color,span) if shade_spans else color,
               hatch=None if shade_spans else hatch,edgecolor='black',linewidth=1)
        label_heights=np.asarray(vals)+.025
        if len(series)>8:
            label_heights+=.035*(i%2)
        for xx,v,yy in zip(x,vals,label_heights):
            ax.text(xx,yy,f'{v:.2f}',ha='center',fontsize=8)
    ax.set_xticks(range(len(tasks)+1),[n for _,n in tasks]+['Overall'],fontsize=12)
    ax.axvline(len(tasks)-.5,color='#CFCECE',linewidth=1,linestyle='--')
    ax.set_ylim(0,1.15);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylabel('Target answer rate');ax.set_xlabel(xlabel,labelpad=12)
    handles=legend_handles(spans,shade_spans,methods)
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.15),
              ncol=len(handles),fontsize=12)
    fig.tight_layout()
    save(fig,out_path,formats)


def render_stacked_task_figure(stats_by_panel, out_path, spans=MAIN_SPANS,
                              shade_spans=False, tasks=TASKS, methods=METHODS,
                              xlabel='ICL task', formats=FORMATS):
    """Stack comparisons vertically with shared task positions and rate scales."""
    apply_publication_style()
    panels = list(stats_by_panel)
    fig, axes = plt.subplots(len(panels), 1, figsize=(21, 6 * len(panels)),
                             sharex=True, sharey=True, squeeze=False)
    series = [(arm, span, color, hatch)
              for arm, _, color in methods for span, _, hatch in spans]
    width = .84 / len(series)
    task_centers = np.arange(len(tasks) + 1)

    for ax, panel in zip(axes[:, 0], panels):
        stats = stats_by_panel[panel]
        for series_index, (arm, span, color, hatch) in enumerate(series):
            vals = [stats[task][span][arm] for task, _ in tasks]
            vals.append(float(np.mean(vals)))
            x = task_centers + (series_index - (len(series) - 1) / 2) * width
            ax.bar(x, vals, width=width,
                   color=span_shade(color, span) if shade_spans else color,
                   hatch=None if shade_spans else hatch,
                   edgecolor='black', linewidth=1)
            for xx, value in zip(x, vals):
                label_offset = .035 * (series_index % 2) if len(series) > 8 else 0
                ax.text(xx, value + .025 + label_offset, f'{value:.2f}',
                        ha='center', fontsize=8)
        ax.text(.01, .97, str(panel), ha='left', va='top',
                fontsize=13, transform=ax.transAxes)
        ax.axvline(len(tasks) - .5, color='#CFCECE', linewidth=1,
                   linestyle='--')
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, .25, .5, .75, 1])

    ax = axes[-1, 0]
    ax.set_xticks(task_centers, [name for _, name in tasks] + ['Overall'],
                  fontsize=12)
    ax.set_xlabel(xlabel, labelpad=12)
    fig.supylabel('Target answer rate', fontsize=13, x=.008)
    handles = legend_handles(spans, shade_spans, methods)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995),
               ncol=len(handles), fontsize=12)
    fig.tight_layout(rect=(.012, 0, 1, .96), h_pad=2)
    save(fig, out_path, formats)


def render_layer_comparison(stats_by_layer, out_path, spans=MAIN_SPANS, shade_spans=False,
                            formats=FORMATS):
    """Stack layers vertically with shared task positions and rate scales."""
    render_stacked_task_figure({f'Layer {layer}': stats
                               for layer, stats in stats_by_layer.items()},
                              out_path, spans=spans, shade_spans=shade_spans, formats=formats)
