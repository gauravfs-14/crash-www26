import os
import json
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import SimpleLogger

logger = SimpleLogger(log_file="logs/visualize.log")


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
    o = outcome.lower()
    if any(k in o for k in ['fatal', 'death', 'killed']):
        return 'Fatal'
    if any(k in o for k in ['injury', 'injured', 'hospital']):
        return 'Injury'
    if any(k in o for k in ['damage', 'collision', 'struck', 'hit']):
        return 'Property_Damage'
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

    # 2) Alcohol involvement
    fig2 = px.histogram(
        df,
        x='mentions_alcohol',
        color='mentions_alcohol',
        title='Alcohol/Drugs Mentioned',
        color_discrete_sequence=px.colors.sequential.Reds,
    )
    fig2.update_layout(template='plotly_white', xaxis_title='Mentions Alcohol/Drugs')
    fig2.write_image(os.path.join(out_dir, 'alcohol_mentions.png'), scale=2)
    fig2.write_html(os.path.join(out_dir, 'alcohol_mentions.html'))

    # 3) Top actions
    top_actions = df['action'].value_counts().head(15).reset_index()
    top_actions.columns = ['action', 'count']
    fig3 = px.bar(
        top_actions,
        x='count',
        y='action',
        orientation='h',
        title='Top Actions',
        color='count',
        color_continuous_scale='Viridis',
    )
    fig3.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
    fig3.write_image(os.path.join(out_dir, 'top_actions.png'), scale=2)
    fig3.write_html(os.path.join(out_dir, 'top_actions.html'))

    # 4) Conditions heatmap (co-occurrence when multiple listed)
    split_conditions = df['conditions'].str.split(',').dropna().apply(lambda x: [c.strip() for c in x if c.strip()])
    all_conds = sorted({c for sub in split_conditions for c in sub})
    matrix = pd.DataFrame(0, index=all_conds, columns=all_conds)
    for conds in split_conditions:
        for i in range(len(conds)):
            for j in range(i, len(conds)):
                a, b = conds[i], conds[j]
                matrix.loc[a, b] += 1
                if a != b:
                    matrix.loc[b, a] += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='Blues', cbar=True)
    plt.title('Condition Co-occurrence Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'conditions_heatmap.png'), dpi=200)
    plt.close()


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


def plot_condition_treemap(df: pd.DataFrame, out_dir: str):
    # Treemap of Condition Type → Condition
    split_conditions = df['conditions'].str.split(',').dropna().apply(lambda x: [c.strip() for c in x if c.strip()])
    rows = []
    for conds in split_conditions:
        for c in conds:
            rows.append({'condition': c, 'type': _classify_condition_type(c)})
    if not rows:
        return
    dd = pd.DataFrame(rows)
    grouped = dd.groupby(['type', 'condition']).size().reset_index(name='count')
    fig = px.treemap(grouped, path=['type', 'condition'], values='count', color='type',
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     title='Conditions Treemap (Type → Specific)')
    fig.update_layout(template='plotly_white')
    fig.write_html(os.path.join(out_dir, 'conditions_treemap.html'))
    fig.write_image(os.path.join(out_dir, 'conditions_treemap.png'), scale=2)


def plot_action_severity_stack(df: pd.DataFrame, out_dir: str):
    # Stacked bar: Top actions by severity distribution
    top_actions = df['action'].value_counts().head(10).index.tolist()
    dff = df[df['action'].isin(top_actions)]
    pivot = dff.pivot_table(index='action', columns='severity_proxy', values='crash_id', aggfunc='count', fill_value=0)
    pivot = pivot[['Minor', 'Property_Damage', 'Injury', 'Fatal']] if set(['Minor','Property_Damage','Injury','Fatal']).issubset(pivot.columns.union(['Minor','Property_Damage','Injury','Fatal'])) else pivot
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
    for name in ['severity_distribution', 'alcohol_mentions', 'top_actions', 'sankey', 'conditions_treemap', 'actions_severity_stacked']:
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
        <div class="subtitle">Computational analysis of causal events extracted from crash narratives.</div>
        {kpis}
        <div class="grid">
            <div class="card">{parts['severity_distribution']}</div>
            <div class="card">{parts['alcohol_mentions']}</div>
            <div class="card">{parts['top_actions']}</div>
            <div class="card">{parts['actions_severity_stacked']}</div>
            <div class="card">{parts['sankey']}</div>
            <div class="card"><img class="figure" src="conditions_heatmap.png"/></div>
            <div class="card">{parts['conditions_treemap']}</div>
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
    plot_summary(df, out_dir)
    plot_action_severity_stack(df, out_dir)
    plot_sankey(df, out_dir)
    plot_condition_treemap(df, out_dir)
    build_dashboard(df, out_dir)


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


