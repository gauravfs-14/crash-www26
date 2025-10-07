import os
import json
from typing import Dict, Any, List, Tuple
import math
import numpy as np
import scipy.stats as stats
import networkx as nx

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import SimpleLogger

logger = SimpleLogger(log_file="logs/visualize.log")


# =========================
# Global visualization style
_FONT_SERIF = 'DejaVu Serif'
_COLOR_TEXT = '#2f2f2f'
_COLOR_TITLE = '#1a1a1a'
_COLOR_GRID = '#d1cfc7'
_ACCENT = '#2a6f97'
_SIZE_TITLE = 22
_SIZE_SUBTITLE = 18
_SIZE_AXIS_TITLE = 17
_SIZE_AXIS_LABEL = 13
_SIZE_TICK = 11
_SIZE_LEGEND = 11
_SIZE_FOOTNOTE = 9


def _apply_global_style():
    # Matplotlib/Seaborn
    plt.rcParams.update({
        'font.family': _FONT_SERIF,
        'axes.titlesize': _SIZE_TITLE,
        'axes.labelsize': _SIZE_AXIS_TITLE,
        'xtick.labelsize': _SIZE_TICK,
        'ytick.labelsize': _SIZE_TICK,
        'axes.edgecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
        'text.color': _COLOR_TEXT,
    })
    sns.set_theme(style='white', font=_FONT_SERIF)

    # Plotly default tweaks per chart; we'll apply layout on each fig
    pass


def _apply_plotly_layout(fig, title: str):
    fig.update_layout(
        template='plotly_white',
        title={'text': title, 'font': {'family': _FONT_SERIF, 'size': _SIZE_SUBTITLE, 'color': _COLOR_TITLE}},
        font={'family': _FONT_SERIF, 'size': _SIZE_AXIS_LABEL, 'color': _COLOR_TEXT},
        legend={'font': {'size': _SIZE_LEGEND}},
        xaxis={'gridcolor': _COLOR_GRID, 'griddash': 'dash'},
        yaxis={'gridcolor': _COLOR_GRID, 'griddash': 'dash'},
        margin={'l': 60, 'r': 30, 't': 60, 'b': 50},
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
    )


def load_extractions(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj: Dict[str, Any] = json.loads(line)
            crash_id = obj.get('crash_id')
            events = obj.get('events') or []
            for ev in events:
                rec = {
                    'crash_id': crash_id,
                    'agent_id': ev.get('agent_id'),
                    'agent_type': ev.get('agent_type'),
                    'action': ev.get('action'),
                    'outcome': ev.get('outcome'),
                    'conditions': ", ".join(ev.get('conditions') or []),
                    'mentions_alcohol': bool(ev.get('mentions_alcohol_or_drugs')),
                    'severity_proxy': _severity_proxy(ev.get('outcome') or ""),
                }
                records.append(rec)
    return pd.DataFrame.from_records(records)


def _severity_proxy(outcome: str) -> str:
    """Determine severity based on actual outcome patterns in the data."""
    o = outcome.lower()
    if any(k in o for k in ['fatal', 'death', 'killed', 'died']):
        return 'Fatal'
    elif any(k in o for k in ['injury', 'injured', 'hurt', 'wounded', 'medical', 'hospital', 'treatment']):
        return 'Injury'
    elif any(k in o for k in ['damage', 'damaged', 'property', 'disabled', 'collision', 'struck', 'hit', 'collided']):
        return 'Property_Damage'
    else:
        return 'Minor'


def plot_summary(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Severity distribution
    fig1 = px.histogram(
        df,
        x='severity_proxy',
        color='severity_proxy',
        title='Event Severity Distribution',
        color_discrete_sequence=px.colors.sequential.Blues,
    )
    fig1.update_layout(template='plotly_white', legend_title_text='Severity')
    fig1.write_image(os.path.join(out_dir, 'severity_distribution.png'), scale=2)
    fig1.write_html(os.path.join(out_dir, 'severity_distribution.html'))

    # Removed alcohol involvement per new spec

    # Removed top actions per new spec

    # Removed condition co-occurrence heatmap per new spec


def _classify_condition_type(cond: str) -> str:
    c = cond.lower()
    if any(k in c for k in ['alcohol', 'dwi', 'dui', 'impair']):
        return 'Impairment'
    if any(k in c for k in ['wet', 'rain', 'snow', 'ice', 'slippery']):
        return 'Weather'
    if any(k in c for k in ['dark', 'light', 'dawn', 'dusk', 'night']):
        return 'Lighting'
    if any(k in c for k in ['speed']):
        return 'Speed'
    if any(k in c for k in ['distract', 'phone', 'text']):
        return 'Distraction'
    if any(k in c for k in ['intersection', 'signal', 'lane', 'guardrail', 'barrier', 'shoulder']):
        return 'Infrastructure'
    return 'Other'


def _extract_condition_types(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        conds = [c.strip() for c in str(r.get('conditions') or '').split(',') if c.strip()]
        types = sorted({_classify_condition_type(c) for c in conds})
        rows.append({
            'crash_id': r['crash_id'],
            'severity_proxy': r['severity_proxy'],
            'condition_types': types,
        })
    return pd.DataFrame(rows)


def _compute_motifs(df: pd.DataFrame) -> pd.DataFrame:
    """Derive motif flags from actual data patterns.
    Motif definitions based on real data:
      - Alcohol_Impairment: mentions alcohol or drugs
      - Unsafe_Maneuvers: unsafe driving actions
      - Collision_Patterns: specific collision types
    """
    # Create motif flags based on actual data patterns
    has_alcohol = df['mentions_alcohol'].fillna(False)
    
    # Unsafe maneuvers from actual actions
    unsafe_actions = df['action'].str.contains(
        'veered|backed unsafely|fled|attempting.*turn', 
        case=False, na=False
    )
    
    # Collision patterns from outcomes
    collision_patterns = df['outcome'].str.contains(
        'collided|struck|hit', 
        case=False, na=False
    )
    
    # Alcohol-related conditions
    alcohol_conditions = df['conditions'].apply(
        lambda x: any('alcohol' in str(cond).lower() for cond in x) if isinstance(x, list) else False
    )

    motifs = pd.DataFrame({
        'crash_id': df['crash_id'],
        'severity_proxy': df['severity_proxy'],
        'Alcohol_Impairment': has_alcohol | alcohol_conditions,
        'Unsafe_Maneuvers': unsafe_actions,
        'Collision_Patterns': collision_patterns,
    })
    return motifs


def _odds_ratio_ci(a: int, b: int, c: int, d: int) -> Tuple[float, Tuple[float, float]]:
    """Compute odds ratio and Wald 95% CI for 2x2 table [[a,b],[c,d]].
    a: motif & severe, b: motif & not severe, c: no motif & severe, d: no motif & not severe
    """
    # Add 0.5 Haldane correction to avoid zeros
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a2 * d2) / (b2 * c2)
    se = (1/a2 + 1/b2 + 1/c2 + 1/d2) ** 0.5
    import math
    lo = math.exp(math.log(or_val) - 1.96 * se)
    hi = math.exp(math.log(or_val) + 1.96 * se)
    return or_val, (lo, hi)


def viz1_motif_risk_dashboard(df: pd.DataFrame, out_dir: str):
    """Visualization 1: Motif risk dashboard (forest + frequency) as a single cohesive PNG."""
    motifs = _compute_motifs(df)
    is_severe = df['severity_proxy'].isin(['Injury', 'Fatal'])
    rows: List[Dict[str, Any]] = []
    for motif in ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']:
        flag = motifs[motif]
        a = int(((flag) & (is_severe)).sum())
        b = int(((flag) & (~is_severe)).sum())
        c = int(((~flag) & (is_severe)).sum())
        d = int(((~flag) & (~is_severe)).sum())
        or_val, (lo, hi) = _odds_ratio_ci(a, b, c, d)
        rows.append({'motif': motif, 'or': or_val, 'lo': lo, 'hi': hi, 'freq': int(flag.sum())})
    res = pd.DataFrame(rows).sort_values('or', ascending=False).reset_index(drop=True)

    _apply_global_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3)

    # Left: Forest plot
    ax = axes[0]
    y = range(len(res))
    
    # Draw confidence intervals with thicker lines for better visibility
    ax.hlines(y, res['lo'], res['hi'], color=_ACCENT, linewidth=3, alpha=0.8)
    ax.plot(res['or'], y, 'o', color=_ACCENT, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # Set up axes first
    ax.set_yticks(list(y))
    ax.set_yticklabels(res['motif'], fontsize=_SIZE_AXIS_LABEL)
    ax.set_xlabel('Odds Ratio (Injury/Fatal vs Other)', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    ax.set_title('Risk Motifs – Forest Plot', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4, color=_COLOR_GRID)
    
    # Add OR values with better positioning first
    for i, (lo, hi, orv) in enumerate(zip(res['lo'], res['hi'], res['or'])):
        # Position text to the right with adequate spacing, avoiding OR=1 line
        text_x = max(hi + (hi - lo) * 0.3, 1.5)  # Ensure text is well to the right of OR=1
        ax.text(text_x, i, f"{orv:.2f}", va='center', ha='left', 
                fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Set x-axis limits to accommodate text
    max_x = max(res['hi']) * 1.4
    ax.set_xlim(0, max_x)
    
    # Add OR=1 reference line with better positioning (after setting limits)
    ax.axvline(1.0, color=_COLOR_GRID, linestyle='--', linewidth=2, alpha=0.7, label='OR=1', zorder=1)
    
    # Add legend in upper right to avoid overlap
    ax.legend(loc='upper right', fontsize=_SIZE_AXIS_LABEL, framealpha=0.9)

    # Right: Frequency bar (horizontal, values labeled)
    ax2 = axes[1]
    freq_sorted = res.sort_values('freq', ascending=True)
    max_freq = max(freq_sorted['freq']) if len(freq_sorted) > 0 else 1
    
    # Create bars with better styling
    bars = ax2.barh(freq_sorted['motif'], freq_sorted['freq'], 
                    color='#5fa8d3', edgecolor='#2a6f97', linewidth=1, alpha=0.8)
    
    # Add value labels with smart positioning
    for i, (bar, v) in enumerate(zip(bars, freq_sorted['freq'])):
        bar_width = bar.get_width()
        if bar_width > max_freq * 0.1:  # If bar is wide enough, put text inside
            ax2.text(bar_width/2, bar.get_y() + bar.get_height()/2, f"{int(v)}", 
                    va='center', ha='center', fontsize=_SIZE_AXIS_LABEL, 
                    color='white', weight='bold')
        else:  # Otherwise, put text outside
            ax2.text(bar_width + max_freq * 0.05, bar.get_y() + bar.get_height()/2, f"{int(v)}", 
                    va='center', fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, weight='bold')
    
    # Set up axes with proper limits
    if max_freq > 0:
        ax2.set_xlim(0, max_freq * 1.3)
    else:
        ax2.set_xlim(0, 1)
    ax2.set_xlabel('Frequency', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    ax2.set_title('Motif Frequency', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.4, color=_COLOR_GRID)
    
    # Set y-axis labels properly
    ax2.set_yticks(range(len(freq_sorted)))
    ax2.set_yticklabels(freq_sorted['motif'], fontsize=_SIZE_AXIS_LABEL)

    # Remove main title and unnecessary footnotes
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'viz1_motif_risk.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def viz2_causal_pathways_sankey(df: pd.DataFrame, out_dir: str):
    """Visualization 2: Causal Risk Pathways Sankey (Action → Outcome → Severity).
    Styled per VISUALIZATIONS.md with muted palette and clear labeling.
    """
    _apply_global_style()
    # Prepare category selection with smart bucketing
    action_counts = df['action'].value_counts()
    outcome_counts = df['outcome'].value_counts()

    # If unique categories are few, do not bucket into "Other"
    if len(action_counts) <= 6:
        selected_actions = list(action_counts.index)
    else:
        selected_actions = list(action_counts.head(6).index) + ['Other Action']

    if len(outcome_counts) <= 6:
        selected_outcomes = list(outcome_counts.index)
    else:
        selected_outcomes = list(outcome_counts.head(6).index) + ['Other Outcome']

    # Mapping functions: keep original values for counting; only map label to 'Other *' when needed
    def map_action(a: str) -> str:
        if 'Other Action' in selected_actions and a not in selected_actions:
            return 'Other Action'
        return a

    def map_outcome(o: str) -> str:
        if 'Other Outcome' in selected_outcomes and o not in selected_outcomes:
            return 'Other Outcome'
        return o

    # Create longer, more descriptive labels for Sankey
    def create_short_label(text: str, max_len: int = 35) -> str:
        if not isinstance(text, str):
            return str(text)
        
        # If text is short enough, return as is
        if len(text) <= max_len:
            return text
        
        # For longer text, try to keep the most important words
        words = text.split()
        if len(words) <= 4:
            # If few words, just truncate with ellipsis
            return text[:max_len-3] + '...'
        
        # Try to keep first few words that fit
        result = ''
        for word in words:
            if len(result + ' ' + word) <= max_len - 3:
                result += (' ' + word) if result else word
            else:
                break
        
        return result + '...' if result else text[:max_len-3] + '...'

    df_f = df.copy()
    df_f['action_bucket'] = df_f['action'].apply(map_action)
    df_f['outcome_bucket'] = df_f['outcome'].apply(map_outcome)

    action_uniques_raw = sorted(pd.Series(df_f['action_bucket'].unique()).tolist())
    outcome_uniques_raw = sorted(pd.Series(df_f['outcome_bucket'].unique()).tolist())
    # Build short labels that fit in Sankey nodes
    action_label_map = {raw: create_short_label(raw) for raw in action_uniques_raw}
    outcome_label_map = {raw: create_short_label(raw) for raw in outcome_uniques_raw}
    actions = [action_label_map[a] for a in action_uniques_raw]
    outcomes = [outcome_label_map[o] for o in outcome_uniques_raw]
    severities = ['Minor', 'Property_Damage', 'Injury', 'Fatal']
    nodes = actions + outcomes + severities
    node_index = {name: i for i, name in enumerate(nodes)}

    # Build links: action→outcome
    src, tgt, val = [], [], []
    # Build using buckets, then label with truncated display strings
    for a_raw in action_uniques_raw:
        for o_raw in outcome_uniques_raw:
            v = int(((df_f['action_bucket'] == a_raw) & (df_f['outcome_bucket'] == o_raw)).sum())
            if v > 0:
                src.append(node_index[action_label_map[a_raw]])
                tgt.append(node_index[outcome_label_map[o_raw]])
                val.append(v)
    # outcome→severity
    for o_raw in outcome_uniques_raw:
        for s in severities:
            v = int(((df_f['outcome_bucket'] == o_raw) & (df_f['severity_proxy'] == s)).sum())
            if v > 0:
                src.append(node_index[outcome_label_map[o_raw]])
                tgt.append(node_index[s])
                val.append(v)

    # Multiple colors for different node types
    node_colors = []
    for i, node in enumerate(nodes):
        if i < len(actions):
            node_colors.append('#a6cee3')  # Light blue for actions
        elif i < len(actions) + len(outcomes):
            node_colors.append('#b2df8a')  # Light green for outcomes
        else:
            node_colors.append('#fb9a99')  # Light red for severities

    # Color links based on source node type
    link_colors = []
    for s in src:
        if s < len(actions):
            link_colors.append('rgba(166,206,227,0.6)')  # Blue for action links
        else:
            link_colors.append('rgba(178,223,138,0.6)')  # Green for outcome links

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=25,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=nodes,
            color=node_colors,
            hovertemplate='%{label}<extra></extra>',
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
            color=link_colors,
        ),
    )])
    
    # Apply readable font size for longer labels
    fig.update_layout(
        font=dict(size=9),
        title_font_size=18,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    _apply_plotly_layout(fig, 'Causal Risk Pathways (Action → Outcome → Severity)')
    out_path = os.path.join(out_dir, 'viz2_causal_pathways.png')
    fig.write_image(out_path, scale=2)


def viz3_comprehensive_stats_dashboard(df: pd.DataFrame, out_dir: str):
    """Visualization 3: Comprehensive statistical analysis of causal risk pathways.
    Includes severity distribution, statistical significance, OR distribution, and event complexity.
    """
    _apply_global_style()
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Crash severity distribution
    ax1 = axes[0, 0]
    severity_counts = df['severity_proxy'].value_counts().reindex(['Minor', 'Property_Damage', 'Injury', 'Fatal'], fill_value=0)
    colors = ['#d1d5db', '#93c5fd', '#f59e0b', '#ef4444']
    bars = ax1.bar(severity_counts.index, severity_counts.values, color=colors, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Crash Severity', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    ax1.set_ylabel('Count', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    ax1.set_title('Crash Severity Distribution', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
    
    # Add value labels inside bars
    for bar, value in zip(bars, severity_counts.values):
        if value > 0:
            # Position text inside the bar
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    str(value), ha='center', va='center', fontsize=_SIZE_AXIS_LABEL, 
                    color='white', weight='bold')
    
    # 2. Statistical significance distribution (motif p-values)
    ax2 = axes[0, 1]
    motifs = _compute_motifs(df)
    is_severe = df['severity_proxy'].isin(['Injury', 'Fatal'])
    p_values = []
    motif_names = []
    
    for col in ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']:
        if col in motifs.columns:
            flag = motifs[col]
            a = int(((flag) & (is_severe)).sum())
            b = int(((flag) & (~is_severe)).sum())
            c = int(((~flag) & (is_severe)).sum())
            d = int(((~flag) & (~is_severe)).sum())
            
            # Ensure we have valid contingency table
            if a + b + c + d > 0 and (a + c) > 0 and (b + d) > 0:
                table = np.array([[a, b], [c, d]])
                try:
                    _, p_val = stats.fisher_exact(table)
                    if p_val > 0:
                        p_values.append(-np.log10(p_val))
                        motif_names.append(col.replace('_', ' '))
                except:
                    # Fallback for edge cases
                    p_values.append(0.5)
                    motif_names.append(col.replace('_', ' '))
    
    if p_values:
        bars = ax2.bar(motif_names, p_values, color='#5fa8d3', edgecolor='white', linewidth=1)
        ax2.set_xlabel('Motif', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax2.set_ylabel('-log10(p-value)', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax2.set_title('Motif Statistical Significance', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.legend(fontsize=_SIZE_AXIS_LABEL)
        
        # Add value labels only if values are meaningful
        for bar, value in zip(bars, p_values):
            if value > 0.1:  # Only show labels for meaningful values
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT)
        
        # Add explanatory text for non-significant results
        if all(v < 0.5 for v in p_values):
            ax2.text(0.5, 0.7, 'No significant associations\nfound in this sample', 
                    ha='center', va='center', transform=ax2.transAxes, 
                    fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    else:
        # Show message if no valid data
        ax2.text(0.5, 0.5, 'Insufficient data for\nstatistical significance', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT)
        ax2.set_xlabel('Motif', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax2.set_ylabel('-log10(p-value)', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax2.set_title('Motif Statistical Significance', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    # 3. Distribution of odds ratios
    ax3 = axes[1, 0]
    or_values = []
    for col in ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']:
        if col in motifs.columns:
            flag = motifs[col]
            a = int(((flag) & (is_severe)).sum())
            b = int(((flag) & (~is_severe)).sum())
            c = int(((~flag) & (is_severe)).sum())
            d = int(((~flag) & (~is_severe)).sum())
            
            # Ensure valid contingency table for odds ratio
            if a + b + c + d > 0 and (a + c) > 0 and (b + d) > 0:
                try:
                    or_val, _ = _odds_ratio_ci(a, b, c, d)
                    if not np.isnan(or_val) and or_val > 0:
                        or_values.append(or_val)
                except:
                    # Skip invalid calculations
                    continue
    
    if or_values:
        # Create a more informative display for odds ratios
        if len(or_values) == 1:
            # Single value - show as bar chart
            ax3.bar(['Odds Ratio'], or_values, color='#5fa8d3', edgecolor='white', alpha=0.7)
            ax3.set_ylabel('Value', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax3.text(0, or_values[0] + 0.05, f'{or_values[0]:.2f}', ha='center', va='bottom', 
                    fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, weight='bold')
        else:
            # Multiple values - show as histogram
            ax3.hist(or_values, bins=min(5, len(or_values)), color='#5fa8d3', edgecolor='white', alpha=0.7)
            ax3.set_ylabel('Frequency', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        
        ax3.set_xlabel('Odds Ratio', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax3.set_title('Distribution of Motif Odds Ratios', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
        ax3.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
        ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='OR=1')
        ax3.legend(fontsize=_SIZE_AXIS_LABEL)
    else:
        # Show message if no valid odds ratios
        ax3.text(0.5, 0.5, 'Insufficient data for\nodds ratio calculation', 
                ha='center', va='center', transform=ax3.transAxes, 
                fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT)
        ax3.set_xlabel('Odds Ratio', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax3.set_ylabel('Frequency', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        ax3.set_title('Distribution of Motif Odds Ratios', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    # 4. Event graph complexity (events per crash)
    ax4 = axes[1, 1]
    events_per_crash = df.groupby('crash_id').size()
    
    if len(events_per_crash.unique()) == 1:
        # Single value - show as bar chart
        unique_events = events_per_crash.iloc[0]
        count = len(events_per_crash)
        ax4.bar([f'{unique_events} Events'], [count], color='#5fa8d3', edgecolor='white', alpha=0.7)
        ax4.set_ylabel('Number of Crashes', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
        # Add value label on top of bar
        ax4.text(0, count + 0.1, f'{count}', ha='center', va='bottom', 
                fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, weight='bold')
    else:
        # Multiple values - show as histogram
        ax4.hist(events_per_crash, bins=min(15, len(events_per_crash.unique())), 
                color='#5fa8d3', edgecolor='white', alpha=0.7)
        ax4.set_ylabel('Number of Crashes', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    
    ax4.set_xlabel('Number of Events per Crash', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
    ax4.set_title('Event Graph Complexity', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
    
    # Add statistics text
    mean_events = events_per_crash.mean()
    ax4.text(0.7, 0.8, f'Mean: {mean_events:.1f}', transform=ax4.transAxes, 
             fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'viz3_comprehensive_stats.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def viz4_correlation_factor_analysis(df: pd.DataFrame, out_dir: str):
    """Visualization 4: Correlation factor analysis dashboard.
    Includes motif co-occurrence matrix, frequency vs OR scatter, and motif distribution by severity.
    """
    _apply_global_style()
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Compute motifs
    motifs = _compute_motifs(df)
    motif_cols = ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']
    
    # 1. Motif co-occurrence correlation matrix
    ax1 = axes[0, 0]
    if len(motif_cols) >= 2 and not motifs[motif_cols].empty:
        corr_matrix = motifs[motif_cols].corr()
        im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(motif_cols)))
        ax1.set_yticks(range(len(motif_cols)))
        ax1.set_xticklabels([col.replace('_', ' ') for col in motif_cols], rotation=45, ha='right')
        ax1.set_yticklabels([col.replace('_', ' ') for col in motif_cols])
        
        # Add correlation values as text
        for i in range(len(motif_cols)):
            for j in range(len(motif_cols)):
                text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=_SIZE_AXIS_LABEL)
        
        ax1.set_title('Motif Co-occurrence Correlation', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
        plt.colorbar(im, ax=ax1, shrink=0.8)
    else:
        ax1.text(0.5, 0.5, 'Insufficient motif data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Motif Co-occurrence Correlation', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    # 2. Motif frequency vs risk association (OR)
    ax2 = axes[0, 1]
    is_severe = df['severity_proxy'].isin(['Injury', 'Fatal'])
    freq_or_data = []
    
    for col in motif_cols:
        if col in motifs.columns:
            flag = motifs[col]
            a = int(((flag) & (is_severe)).sum())
            b = int(((flag) & (~is_severe)).sum())
            c = int(((~flag) & (is_severe)).sum())
            d = int(((~flag) & (~is_severe)).sum())
            
            if a + b + c + d > 0:
                or_val, _ = _odds_ratio_ci(a, b, c, d)
                freq = int(flag.sum())
                freq_or_data.append({
                    'motif': col.replace('_', ' '),
                    'frequency': freq,
                    'or': or_val
                })
    
    if freq_or_data:
        freq_df = pd.DataFrame(freq_or_data)
        
        # Handle case where all OR values are 0 or very small
        if all(freq_df['or'] <= 0.1):
            # Show frequency distribution instead
            bars = ax2.bar(freq_df['motif'], freq_df['frequency'], 
                          color='#5fa8d3', edgecolor='white', alpha=0.7)
            ax2.set_xlabel('Motif', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax2.set_ylabel('Frequency', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax2.set_title('Motif Frequency Distribution', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
            ax2.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
            
            # Add value labels on bars with better positioning
            for bar, value in zip(bars, freq_df['frequency']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(value), ha='center', va='bottom', fontsize=_SIZE_AXIS_LABEL, 
                        color=_COLOR_TEXT, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add explanatory text
            ax2.text(0.5, 0.9, 'No significant risk associations\nfound in this sample', 
                    ha='center', va='center', transform=ax2.transAxes, 
                    fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            # Show scatter plot for meaningful OR values
            scatter = ax2.scatter(freq_df['frequency'], freq_df['or'], 
                                  c=freq_df['or'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
            
            # Add labels for each point with better positioning
            for i, row in freq_df.iterrows():
                # Calculate offset to avoid overlap
                offset_x = 10 if i % 2 == 0 else -10
                offset_y = 10 if i % 2 == 0 else -10
                ax2.annotate(row['motif'], (row['frequency'], row['or']), 
                            xytext=(offset_x, offset_y), textcoords='offset points', 
                            fontsize=_SIZE_AXIS_LABEL, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            
            ax2.set_xlabel('Motif Frequency', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax2.set_ylabel('Odds Ratio', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax2.set_title('Frequency vs Risk Association', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
            ax2.grid(True, linestyle='--', alpha=0.5, color=_COLOR_GRID)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='OR=1')
            ax2.legend(fontsize=_SIZE_AXIS_LABEL)
            plt.colorbar(scatter, ax=ax2, shrink=0.8, label='Odds Ratio')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for analysis', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Frequency vs Risk Association', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    # 3. Motif distribution by crash severity (stacked bar)
    ax3 = axes[1, 0]
    severity_order = ['Minor', 'Property_Damage', 'Injury', 'Fatal']
    motif_severity_data = []
    
    for _, row in motifs.iterrows():
        for motif_col in motif_cols:
            if motif_col in motifs.columns and row[motif_col]:
                motif_severity_data.append({
                    'motif': motif_col.replace('_', ' '),
                    'severity': row['severity_proxy'] if pd.notna(row['severity_proxy']) else 'Unknown'
                })
    
    if motif_severity_data:
        motif_sev_df = pd.DataFrame(motif_severity_data)
        pivot = motif_sev_df.pivot_table(index='motif', columns='severity', aggfunc='size', fill_value=0)
        pivot = pivot.reindex(columns=severity_order, fill_value=0)
        
        # Only show severities that have data
        available_severities = [sev for sev in severity_order if sev in pivot.columns and pivot[sev].sum() > 0]
        
        if available_severities:
            bottom = np.zeros(len(pivot))
            colors = ['#d1d5db', '#93c5fd', '#f59e0b', '#ef4444']
            color_map = {sev: colors[i] for i, sev in enumerate(severity_order) if sev in available_severities}
            
            for sev in available_severities:
                if sev in pivot.columns:
                    ax3.bar(pivot.index, pivot[sev], bottom=bottom, label=sev, color=color_map[sev], alpha=0.8)
                    bottom += pivot[sev]
            
            ax3.set_xlabel('Motif', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax3.set_ylabel('Count', fontsize=_SIZE_AXIS_TITLE, color=_COLOR_TEXT)
            ax3.set_title('Motif Distribution by Severity', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
            ax3.legend(fontsize=_SIZE_AXIS_LABEL)
            ax3.grid(axis='y', linestyle='--', alpha=0.5, color=_COLOR_GRID)
            
            # Rotate x-axis labels for readability
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on top of stacked bars
            for i, motif in enumerate(pivot.index):
                total_height = bottom[i]
                if total_height > 0:
                    ax3.text(i, total_height + 0.1, str(int(total_height)), 
                            ha='center', va='bottom', fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT, weight='bold')
        else:
            ax3.text(0.5, 0.5, 'No severity data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Motif Distribution by Severity', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    else:
        ax3.text(0.5, 0.5, 'No motif-severity data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Motif Distribution by Severity', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    # 4. Motif association network (simplified)
    ax4 = axes[1, 1]
    if len(motif_cols) >= 2 and not motifs[motif_cols].empty:
        # Create a simple network showing motif relationships
        G = nx.Graph()
        for col in motif_cols:
            if col in motifs.columns:
                G.add_node(col.replace('_', ' '))
        
        # Add edges based on co-occurrence
        edges_added = 0
        for i, col1 in enumerate(motif_cols):
            for j, col2 in enumerate(motif_cols):
                if i < j and col1 in motifs.columns and col2 in motifs.columns:
                    co_occur = int(((motifs[col1]) & (motifs[col2])).sum())
                    if co_occur > 0:
                        G.add_edge(col1.replace('_', ' '), col2.replace('_', ' '), weight=co_occur)
                        edges_added += 1
        
        if G.nodes():
            if edges_added > 0:
                # Network with connections
                pos = nx.spring_layout(G, seed=42, k=4, iterations=200)
                nx.draw_networkx_nodes(G, pos, ax=ax4, node_color='lightblue', node_size=1500, alpha=0.8)
                nx.draw_networkx_edges(G, pos, ax=ax4, width=2, alpha=0.6, edge_color='gray')
                # Draw labels with better positioning and background
                for node, (x, y) in pos.items():
                    ax4.text(x, y, node, ha='center', va='center', fontsize=11, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
                ax4.set_title('Motif Association Network', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
            else:
                # No connections - show nodes in a circle with better spacing
                pos = nx.circular_layout(G, scale=2.0)
                nx.draw_networkx_nodes(G, pos, ax=ax4, node_color='lightblue', node_size=1500, alpha=0.8)
                # Draw labels with better positioning and background
                for node, (x, y) in pos.items():
                    ax4.text(x, y, node, ha='center', va='center', fontsize=11, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
                ax4.set_title('Motif Association Network', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
                
                # Add explanatory text with better positioning
                ax4.text(0.5, 0.1, 'No co-occurrences found\nbetween motifs', 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=9, color=_COLOR_TEXT,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax4.text(0.5, 0.5, 'No network connections', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Motif Association Network', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for network', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Motif Association Network', fontsize=_SIZE_SUBTITLE, color=_COLOR_TITLE, loc='left', weight='bold')
    
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'viz4_correlation_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_sankey(df: pd.DataFrame, out_dir: str):
    # Build a Sankey from Action -> Outcome -> Severity (top categories for readability)
    top_actions = df['action'].value_counts().head(8).index.tolist()
    top_outcomes = df['outcome'].value_counts().head(8).index.tolist()

    df_f = df.copy()
    df_f['action_f'] = df_f['action'].where(df_f['action'].isin(top_actions), 'Other Action')
    df_f['outcome_f'] = df_f['outcome'].where(df_f['outcome'].isin(top_outcomes), 'Other Outcome')

    # Nodes
    actions = sorted(df_f['action_f'].unique())
    outcomes = sorted(df_f['outcome_f'].unique())
    severities = ['Minor', 'Property_Damage', 'Injury', 'Fatal']

    nodes = actions + outcomes + severities
    node_index = {name: i for i, name in enumerate(nodes)}

    # Links: action->outcome
    links_s = []
    links_t = []
    links_v = []
    for a in actions:
        for o in outcomes:
            v = int(((df_f['action_f'] == a) & (df_f['outcome_f'] == o)).sum())
            if v > 0:
                links_s.append(node_index[a])
                links_t.append(node_index[o])
                links_v.append(v)

    # Links: outcome->severity
    for o in outcomes:
        for s in severities:
            v = int(((df_f['outcome_f'] == o) & (df_f['severity_proxy'] == s)).sum())
            if v > 0:
                links_s.append(node_index[o])
                links_t.append(node_index[s])
                links_v.append(v)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=16,
            line=dict(color="rgba(0,0,0,0.2)", width=1),
            label=nodes,
            color=["#93c5fd"]*len(nodes)
        ),
        link=dict(
            source=links_s,
            target=links_t,
            value=links_v,
            color="rgba(99,102,241,0.35)"
        )
    )])
    fig.update_layout(title_text="Flow: Action → Outcome → Severity", font_size=12, template='plotly_white')
    fig.write_html(os.path.join(out_dir, 'sankey.html'))
    fig.write_image(os.path.join(out_dir, 'sankey.png'), scale=2)


def plot_stats_dashboard(df: pd.DataFrame, out_dir: str):
    # Statistical significance proxy: z-score approximate p-values for motifs
    motifs = _compute_motifs(df)
    is_severe = df['severity_proxy'].isin(['Injury', 'Fatal'])
    rows = []
    for col in ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']:
        flag = motifs[col]
        a = int(((flag) & (is_severe)).sum())
        b = int(((flag) & (~is_severe)).sum())
        c = int(((~flag) & (is_severe)).sum())
        d = int(((~flag) & (~is_severe)).sum())
        or_val, (lo, hi) = _odds_ratio_ci(a, b, c, d)
        rows.append({'motif': col, 'a': a, 'b': b, 'c': c, 'd': d, 'or': or_val})
    rr = pd.DataFrame(rows)
    # Distribution of ORs
    fig_or = px.histogram(rr, x='or', nbins=20, title='Distribution of Motif Odds Ratios', color_discrete_sequence=['#2a6f97'])
    fig_or.update_layout(template='plotly_white')
    fig_or.write_html(os.path.join(out_dir, 'motif_or_distribution.html'))
    fig_or.write_image(os.path.join(out_dir, 'motif_or_distribution.png'), scale=2)

    # Event graph complexity proxy: events per crash
    events_per_crash = df.groupby('crash_id').size().reset_index(name='events')
    fig_ec = px.histogram(events_per_crash, x='events', nbins=15, title='Event Graph Complexity (Events per Crash)', color_discrete_sequence=['#6a3d9a'])
    fig_ec.update_layout(template='plotly_white')
    fig_ec.write_html(os.path.join(out_dir, 'event_complexity.html'))
    fig_ec.write_image(os.path.join(out_dir, 'event_complexity.png'), scale=2)


def plot_condition_treemap(df: pd.DataFrame, out_dir: str):
    # Removed per VISUALIZATIONS.md
    return


def plot_correlation_analysis(df: pd.DataFrame, out_dir: str):
    motifs = _compute_motifs(df)
    motif_cols = ['Alcohol_Impairment', 'Unsafe_Maneuvers', 'Collision_Patterns']
    if motifs.empty:
        return
    mm = motifs[motif_cols].astype(int)
    corr = mm.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='crest', vmin=-1, vmax=1)
    plt.title('Motif Co-occurrence Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'motif_correlation.png'), dpi=200)
    plt.close()

    # Frequency vs OR scatter
    is_severe = df['severity_proxy'].isin(['Injury', 'Fatal'])
    rows = []
    for col in motif_cols:
        flag = motifs[col]
        a = int(((flag) & (is_severe)).sum())
        b = int(((flag) & (~is_severe)).sum())
        c = int(((~flag) & (is_severe)).sum())
        d = int(((~flag) & (~is_severe)).sum())
        or_val, _ = _odds_ratio_ci(a, b, c, d)
        rows.append({'motif': col, 'freq': int(flag.sum()), 'or': or_val})
    rr = pd.DataFrame(rows)
    fig = px.scatter(rr, x='freq', y='or', text='motif', title='Motif Frequency vs Risk Association (OR)', color='or', color_continuous_scale='Viridis')
    fig.update_traces(textposition='top center')
    fig.update_layout(template='plotly_white')
    fig.write_html(os.path.join(out_dir, 'motif_freq_vs_or.html'))
    fig.write_image(os.path.join(out_dir, 'motif_freq_vs_or.png'), scale=2)


def plot_action_severity_stack(df: pd.DataFrame, out_dir: str):
    # Stacked bar: Top actions by severity distribution
    top_actions = df['action'].value_counts().head(10).index.tolist()
    dff = df[df['action'].isin(top_actions)]
    pivot = dff.pivot_table(index='action', columns='severity_proxy', values='crash_id', aggfunc='count', fill_value=0)
    desired = ['Minor', 'Property_Damage', 'Injury', 'Fatal']
    # Ensure all desired columns exist, filling missing with 0
    pivot = pivot.reindex(columns=desired, fill_value=0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    fig = go.Figure()
    palette = {
        'Minor': '#d1d5db',
        'Property_Damage': '#93c5fd',
        'Injury': '#f59e0b',
        'Fatal': '#ef4444',
    }
    for sev in pivot.columns:
        fig.add_trace(go.Bar(y=pivot.index.tolist(), x=pivot[sev].tolist(), name=sev, orientation='h', marker_color=palette.get(sev, '#9ca3af')))
    fig.update_layout(
        barmode='stack',
        template='plotly_white',
        title='Top Actions × Severity (Stacked)',
        xaxis_title='Count',
        yaxis_title='Action',
        height=600
    )
    fig.write_html(os.path.join(out_dir, 'actions_severity_stacked.html'))
    fig.write_image(os.path.join(out_dir, 'actions_severity_stacked.png'), scale=2)


def _kpi_cards_html(df: pd.DataFrame) -> str:
    total_events = len(df)
    total_crashes = df['crash_id'].nunique()
    alcohol_rate = (df['mentions_alcohol'].sum() / max(total_events, 1)) * 100.0
    top_action = df['action'].value_counts().idxmax() if total_events else 'N/A'
    return f"""
    <div class="kpis">
      <div class="kpi"><div class="kpi-label">Total Events</div><div class="kpi-value">{total_events}</div></div>
      <div class="kpi"><div class="kpi-label">Unique Crashes</div><div class="kpi-value">{total_crashes}</div></div>
      <div class="kpi"><div class="kpi-label">Alcohol Mention Rate</div><div class="kpi-value">{alcohol_rate:.1f}%</div></div>
      <div class="kpi"><div class="kpi-label">Top Action</div><div class="kpi-value">{top_action}</div></div>
    </div>
    """


def build_dashboard(df: pd.DataFrame, out_dir: str):
    # Rich dashboard HTML with KPI cards and multiple analytic views
    parts = {}
    for name in ['severity_distribution', 'sankey', 'motif_frequency', 'motif_or_distribution', 'motif_significance', 'event_complexity', 'motif_freq_vs_or', 'motif_by_severity']:
        html_path = os.path.join(out_dir, f'{name}.html')
        if os.path.exists(html_path):
            with open(html_path, 'r') as fh:
                parts[name] = fh.read()
        else:
            parts[name] = '<div>Figure not available</div>'

    kpis = _kpi_cards_html(df)
    dashboard = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Crash Causal Events Dashboard</title>
        <style>
            :root {{ --border:#e5e7eb; --muted:#6b7280; --card-bg:#ffffff; --shadow:0 1px 2px rgba(0,0,0,0.05); }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; background:#f9fafb; }}
            h1 {{ font-weight: 600; margin-bottom: 8px; }}
            .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
            .kpis {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-bottom:24px; }}
            .kpi {{ background: var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:16px; box-shadow: var(--shadow); }}
            .kpi-label {{ color: var(--muted); font-size:12px; }}
            .kpi-value {{ font-size:24px; font-weight:600; margin-top:4px; }}
            .grid {{ display:grid; grid-template-columns: 1fr; gap:24px; }}
            @media (min-width: 1200px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px; box-shadow: var(--shadow); }}
            .note {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
            img.figure {{ width:100%; border-radius:8px; border:1px solid var(--border); }}
        </style>
    </head>
    <body>
        <h1>Crash Causal Events Dashboard</h1>
        <div class="subtitle">Computational analysis of causal events and risk motifs extracted from crash narratives.</div>
        {kpis}
        <div class="grid">
            <div class="card">{parts['severity_distribution']}</div>
            <div class="card">{parts['sankey']}</div>
            <div class="card"><img class="figure" src="motif_forest.png"/></div>
            <div class="card">{parts['motif_frequency']}</div>
            <div class="card">{parts['motif_or_distribution']}</div>
            <div class="card">{parts['motif_significance']}</div>
            <div class="card">{parts['motif_freq_vs_or']}</div>
            <div class="card">{parts['event_complexity']}</div>
            <div class="card"><img class="figure" src="motif_correlation.png"/></div>
        </div>
        <div class="note">Auto-generated from extractions. Designed for academic-quality visuals.</div>
    </body>
    </html>
    """

    out_html = os.path.join(out_dir, 'dashboard.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(dashboard)
    logger.info(f"Dashboard written to {out_html}")


def generate_visualizations(extractions_path: str, out_dir: str):
    logger.info(f"Loading extractions from {extractions_path}")
    df = load_extractions(extractions_path)
    if df.empty:
        logger.warning("No events found to visualize.")
        return
    # Generate exactly 5 visualizations as specified
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = out_dir
    
    # Visualization 1: Motif risk dashboard (forest + frequency)
    viz1_motif_risk_dashboard(df, viz_dir)
    
    # Visualization 2: Causal risk pathways Sankey
    viz2_causal_pathways_sankey(df, viz_dir)
    
    # Visualization 3: Comprehensive statistics dashboard
    viz3_comprehensive_stats_dashboard(df, viz_dir)
    
    # Visualization 4: Correlation factor analysis dashboard
    viz4_correlation_factor_analysis(df, viz_dir)
    
    # Visualization 5: Event graph networks panel
    viz5_event_graph_networks(df, viz_dir)
    
    logger.info(f"Generated 5 visualizations: viz1_motif_risk.png, viz2_causal_pathways.png, viz3_comprehensive_stats.png, viz4_correlation_analysis.png, viz5_event_graphs.png")


def viz5_event_graph_networks(df: pd.DataFrame, out_dir: str):
    """Visualization 5: Event graph networks dashboard.
    Shows example event graphs for crashes in a dashboard grid layout.
    """
    _apply_global_style()
    
    # Find crashes with multiple events - get up to 4 crashes
    event_counts = df.groupby('crash_id').size().sort_values(ascending=False)
    multi_event_crashes = event_counts[event_counts > 1].head(4)  # Up to 4 crashes
    
    if len(multi_event_crashes) == 0:
        logger.warning("No crashes with multiple events found for network visualization")
        return
    
    # Create dashboard grid layout following style guidelines (14x10 inches)
    n_crashes = len(multi_event_crashes)
    if n_crashes <= 2:
        fig, axes = plt.subplots(1, n_crashes, figsize=(14, 6))
        if n_crashes == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    
    # Define node type colors using academic color palette from VISUALIZATIONS.md
    node_type_info = {
        'crash': {'color': _ACCENT, 'label': 'Crash', 'size': 800},
        'event': {'color': '#1f77b4', 'label': 'Event', 'size': 600},
        'agent': {'color': '#2ca02c', 'label': 'Agent', 'size': 500},
        'action': {'color': '#ff7f0e', 'label': 'Action', 'size': 400},
        'outcome': {'color': '#e31a1c', 'label': 'Outcome', 'size': 400}
    }
    
    for i, (crash_id, event_count) in enumerate(multi_event_crashes.items()):
        ax = axes[i]
        crash_df = df[df['crash_id'] == crash_id].copy()
        
        # Add panel subtitle using matplotlib's title function (18pt)
        ax.set_title(f'Crash {crash_id} - {event_count} Events', 
                    fontsize=_SIZE_SUBTITLE, weight='bold', color=_COLOR_TITLE, pad=20)
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add crash node
        G.add_node('Crash', node_type='crash')
        
        # Add event nodes and relationships using real data
        for idx, (_, row) in enumerate(crash_df.iterrows()):
            event_id = f"Event_{idx+1}"
            agent_id = str(row['agent_id']) if pd.notna(row['agent_id']) else f'Agent_{idx+1}'
            action = str(row['action']) if pd.notna(row['action']) else 'Unknown Action'
            outcome = str(row['outcome']) if pd.notna(row['outcome']) else 'Unknown Outcome'
            agent_type = str(row['agent_type']) if pd.notna(row['agent_type']) else 'Unknown'
            
            # Truncate long text for readability but keep meaningful content
            action_short = action[:20] + '...' if len(action) > 20 else action
            outcome_short = outcome[:20] + '...' if len(outcome) > 20 else outcome
            agent_short = f"{agent_type}: {agent_id[:10]}..." if len(agent_id) > 10 else f"{agent_type}: {agent_id}"
            
            # Add nodes with real data labels
            G.add_node(event_id, node_type='event', label=f"Event {idx+1}")
            G.add_node(agent_id, node_type='agent', label=agent_short)
            G.add_node(f"Action_{idx+1}", node_type='action', label=action_short)
            G.add_node(f"Outcome_{idx+1}", node_type='outcome', label=outcome_short)
            
            # Add edges
            G.add_edge('Crash', event_id, relation='contains')
            G.add_edge(event_id, agent_id, relation='involves')
            G.add_edge(agent_id, f"Action_{idx+1}", relation='performs')
            G.add_edge(f"Action_{idx+1}", f"Outcome_{idx+1}", relation='results_in')
            
            # Add causal relationships between events if multiple
            if idx > 0:
                prev_event = f"Event_{idx}"
                G.add_edge(prev_event, event_id, relation='precedes', style='dashed')
        
        # Draw the network
        if G.nodes():
            # Use spring layout with proper spacing
            pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
            
            # Draw nodes by type with academic colors
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                node_type = G.nodes[node].get('node_type', 'unknown')
                if node_type == 'crash':
                    node_colors.append(_ACCENT)
                    node_sizes.append(800)
                elif node_type == 'event':
                    node_colors.append('#1f77b4')
                    node_sizes.append(600)
                elif node_type == 'agent':
                    node_colors.append('#2ca02c')
                    node_sizes.append(500)
                elif node_type == 'action':
                    node_colors.append('#ff7f0e')
                    node_sizes.append(400)
                elif node_type == 'outcome':
                    node_colors.append('#e31a1c')
                    node_sizes.append(400)
                else:
                    node_colors.append('#9ca3af')
                    node_sizes.append(300)
            
            # Draw nodes with proper styling
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.8, 
                                 edgecolors=_COLOR_TEXT, linewidths=0.7)
            
            # Draw edges with academic styling
            solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') != 'precedes']
            dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'precedes']
            
            if solid_edges:
                nx.draw_networkx_edges(G, pos, edgelist=solid_edges, ax=ax, 
                                      edge_color=_COLOR_GRID, width=0.7, alpha=0.6, 
                                      arrows=True, arrowsize=12)
            if dashed_edges:
                nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, ax=ax, 
                                      edge_color=_ACCENT, width=0.7, alpha=0.8, 
                                      arrows=True, arrowsize=10, style='dashed')
            
            # Draw labels with proper typography (11pt tick labels)
            for node, (x, y) in pos.items():
                node_type = G.nodes[node].get('node_type', 'unknown')
                label = G.nodes[node].get('label', node)
                
                # Use proper font size from style guidelines
                ax.text(x, y, label, ha='center', va='center', 
                       fontsize=_SIZE_TICK, weight='bold', color=_COLOR_TEXT,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               alpha=0.9, edgecolor=_COLOR_GRID, linewidth=0.5))
            
        else:
            ax.text(0.5, 0.5, f'No network data for Crash {crash_id}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=_SIZE_AXIS_LABEL, color=_COLOR_TEXT,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.8, edgecolor=_COLOR_GRID, linewidth=0.5))
        
        # Apply proper axes treatment (remove spines, set background)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add legend following style guidelines (11pt legend, transparent frame)
    legend_elements = []
    for node_type, info in node_type_info.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=info['color'], markersize=6, 
                                         label=info['label']))
    
    # Position legend to not obstruct content
    if n_crashes == 1:
        axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
                      fontsize=_SIZE_LEGEND, title='Node Types', title_fontsize=_SIZE_AXIS_LABEL,
                      frameon=True, fancybox=False, shadow=False, 
                      edgecolor=_COLOR_GRID, facecolor='white')
    elif n_crashes == 2:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                  ncol=5, fontsize=_SIZE_LEGEND, title='Node Types', title_fontsize=_SIZE_AXIS_LABEL,
                  frameon=True, fancybox=False, shadow=False,
                  edgecolor=_COLOR_GRID, facecolor='white')
    elif n_crashes == 3:
        axes[3].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5),
                      fontsize=_SIZE_LEGEND, title='Node Types', title_fontsize=_SIZE_AXIS_LABEL,
                      frameon=True, fancybox=False, shadow=False,
                      edgecolor=_COLOR_GRID, facecolor='white')
        axes[3].set_visible(True)
    else:  # n_crashes == 4
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                  ncol=5, fontsize=_SIZE_LEGEND, title='Node Types', title_fontsize=_SIZE_AXIS_LABEL,
                  frameon=True, fancybox=False, shadow=False,
                  edgecolor=_COLOR_GRID, facecolor='white')
    
    # Hide unused subplots
    if n_crashes == 3:
        pass  # Keep 4th subplot for legend
    else:
        for i in range(n_crashes, len(axes)):
            axes[i].set_visible(False)
    
    # Apply tight layout with balanced margins
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'viz5_event_graphs.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_event_graph_examples(df: pd.DataFrame, out_dir: str, top_n: int = 3):
    counts = df.groupby('crash_id').size().sort_values(ascending=False).head(top_n)
    for crash_id in counts.index:
        sub = df[df['crash_id'] == crash_id]
        import networkx as nx
        G = nx.DiGraph()
        G.add_node(str(crash_id), kind='crash')
        for _, r in sub.iterrows():
            ev_node = f"{r['agent_id']}:{r['action']}"
            G.add_node(ev_node, kind='event', severity=r['severity_proxy'])
            G.add_edge(str(crash_id), ev_node)
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 6))
        node_colors = []
        for n, data in G.nodes(data=True):
            if data.get('kind') == 'crash':
                node_colors.append('#1a1a1a')
            else:
                sev = data.get('severity', 'Minor')
                palette = {'Minor': '#d1d5db', 'Property_Damage': '#93c5fd', 'Injury': '#f59e0b', 'Fatal': '#ef4444'}
                node_colors.append(palette.get(sev, '#9ca3af'))
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=500, arrows=False, edge_color='#d1cfc7')
        for n, (x, y) in pos.items():
            plt.text(x, y, n, fontsize=8, ha='center', va='center', color='#2f2f2f')
        plt.title(f'Event Graph Network – Crash {crash_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'event_graph_{crash_id}.png'), dpi=200)
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate visualizations from extractions')
    parser.add_argument('--extractions', type=str, required=True, help='Path to extractions.jsonl')
    parser.add_argument('--out-dir', type=str, default=os.path.join(os.path.dirname(__file__), 'outputs'), help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generate_visualizations(args.extractions, args.out_dir)


if __name__ == '__main__':
    main()


