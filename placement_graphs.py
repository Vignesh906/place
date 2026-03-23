import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os, uuid

# ── Palette ────────────────────────────────────────────────────────────────
BG          = '#0B0D14'
CARD        = '#161A28'
CARD2       = '#1C2133'
GRID        = '#252B3B'
TEXT        = '#E2E8F8'
TEXT2       = '#8B95B0'
PLACED_CLR  = '#22C77A'
NOT_CLR     = '#F75A5A'
USER_CLR    = '#F7C948'
ACCENT      = '#4F8EF7'
PURPLE      = '#9B72F7'

def _base_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   CARD,
        'axes.edgecolor':   GRID,
        'axes.labelcolor':  TEXT2,
        'xtick.color':      TEXT2,
        'ytick.color':      TEXT2,
        'text.color':       TEXT,
        'grid.color':       GRID,
        'grid.linewidth':   0.6,
        'font.family':      'DejaVu Sans',
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
    })

def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Academic Score Comparison (user vs dataset percentiles)
# ══════════════════════════════════════════════════════════════════════════
def graph_academic_comparison(user_data: dict, df: pd.DataFrame, out_dir: str) -> str:
    _base_style()

    has_mba = user_data.get('has_mba') == 'yes'
    metrics = [
        ('SSC %',    'ssc_p',    float(user_data.get('ssc_p', 0))),
        ('HSC %',    'hsc_p',    float(user_data.get('hsc_p', 0))),
        ('Degree %', 'degree_p', float(user_data.get('degree_p', 0))),
        ('Etest %',  'etest_p',  float(user_data.get('etest_p', 0))),
    ]
    if has_mba:
        metrics.insert(3, ('MBA %', 'mba_p', float(user_data.get('mba_p', 62))))

    labels      = [m[0] for m in metrics]
    user_vals   = [m[2] for m in metrics]
    placed_avg  = [df[df['status']=='Placed'][m[1]].mean()     for m in metrics]
    notplcd_avg = [df[df['status']=='Not Placed'][m[1]].mean() for m in metrics]

    x  = np.arange(len(labels))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w,   notplcd_avg, w, color=NOT_CLR,    alpha=0.65, label='Avg Not Placed', zorder=3, edgecolor='none')
    b2 = ax.bar(x,       placed_avg,  w, color=PLACED_CLR, alpha=0.65, label='Avg Placed',     zorder=3, edgecolor='none')
    b3 = ax.bar(x + w,   user_vals,   w, color=USER_CLR,   alpha=0.95, label='Your Score',     zorder=3, edgecolor='none')

    for bar, v in zip(b3, user_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{v:.0f}', ha='center', va='bottom', fontsize=9,
                color=USER_CLR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('📚 Your Academic Scores vs Dataset Averages', fontsize=14,
                 fontweight='bold', pad=14, color=TEXT)
    ax.legend(framealpha=0.15, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    ax.yaxis.grid(True, zorder=0, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, 'dyn_academic.png', out_dir)


# ══════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Skill Radar Chart (user vs avg placed)
# ══════════════════════════════════════════════════════════════════════════
def graph_skill_radar(user_data: dict, df: pd.DataFrame, out_dir: str) -> str:
    _base_style()

    placed       = df[df['status'] == 'Placed']
    is_technical = user_data.get('is_technical', 'yes') == 'yes'

    if is_technical:
        categories = ['Communication', 'Technical', 'Coding\n(/10)', 'Internships\n(/3)', 'Projects\n(/4)']
        user_vals  = [
            float(user_data.get('communication', 0)) / 10,
            float(user_data.get('technical', 0))     / 10,
            float(user_data.get('coding_score', 0))  / 100,
            int(user_data.get('internships', 0))     / 3,
            int(user_data.get('projects', 0))        / 4,
        ]
        placed_vals = [
            placed['communication'].mean() / 10,
            placed['technical'].mean()     / 10,
            placed['coding_score'].mean()  / 100,
            placed['internships'].mean()   / 3,
            placed['projects'].mean()      / 4,
        ]
    else:
        # Non-technical — replace coding & technical with etest & degree
        categories = ['Communication', 'Etest\n(/100)', 'Degree\n(%)', 'Internships\n(/3)', 'Projects\n(/4)']
        user_vals  = [
            float(user_data.get('communication', 0)) / 10,
            float(user_data.get('etest_p', 0))       / 100,
            float(user_data.get('degree_p', 0))      / 100,
            int(user_data.get('internships', 0))     / 3,
            int(user_data.get('projects', 0))        / 4,
        ]
        placed_vals = [
            placed['communication'].mean() / 10,
            placed['etest_p'].mean()       / 100,
            placed['degree_p'].mean()      / 100,
            placed['internships'].mean()   / 3,
            placed['projects'].mean()      / 4,
        ]

    N   = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    user_vals   += user_vals[:1]
    placed_vals += placed_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    ax.plot(angles, placed_vals, color=PLACED_CLR, linewidth=2, linestyle='solid')
    ax.fill(angles, placed_vals, color=PLACED_CLR, alpha=0.15)

    ax.plot(angles, user_vals, color=USER_CLR, linewidth=2.5, linestyle='solid')
    ax.fill(angles, user_vals, color=USER_CLR, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], fontsize=8, color=TEXT2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines['polar'].set_color(GRID)

    legend_els = [
        mpatches.Patch(color=PLACED_CLR, alpha=0.7, label='Avg Placed Student'),
        mpatches.Patch(color=USER_CLR,   alpha=0.8, label='Your Profile'),
    ]
    ax.legend(handles=legend_els, loc='upper right', bbox_to_anchor=(1.3, 1.15),
              framealpha=0.15, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    ax.set_title('⚡ Skill Radar: You vs Avg Placed Student',
                 fontsize=13, fontweight='bold', pad=22, color=TEXT)
    plt.tight_layout()
    return _save(fig, 'dyn_radar.png', out_dir)


# ══════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Probability Gauge / Speedometer
# ══════════════════════════════════════════════════════════════════════════
def graph_probability_gauge(probability: float, out_dir: str) -> str:
    _base_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.6, 1.3)
    ax.axis('off')

    # Draw arc segments
    segments = [
        (0,   25,  NOT_CLR,    'Very Low'),
        (25,  50,  '#FF9F45',  'Low'),
        (50,  75,  '#F7C948',  'Moderate'),
        (75,  90,  '#7FD97F',  'High'),
        (90,  100, PLACED_CLR, 'Excellent'),
    ]

    for start, end, color, _ in segments:
        theta1 = 180 - (start / 100) * 180
        theta2 = 180 - (end   / 100) * 180
        arc = mpatches.Wedge(
            center=(0, 0), r=1.1, theta1=theta2, theta2=theta1,
            width=0.28, color=color, alpha=0.85
        )
        ax.add_patch(arc)

    # Needle
    needle_angle = np.radians(180 - (probability / 100) * 180)
    needle_len   = 0.82
    ax.annotate('',
        xy=(needle_len * np.cos(needle_angle), needle_len * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=USER_CLR, lw=2.8,
                        mutation_scale=18))

    # Center dot
    circle = plt.Circle((0, 0), 0.06, color=USER_CLR, zorder=5)
    ax.add_patch(circle)

    # Probability text
    color = PLACED_CLR if probability >= 50 else NOT_CLR
    ax.text(0, -0.28, f'{probability:.1f}%', ha='center', va='center',
            fontsize=32, fontweight='bold', color=color,
            fontfamily='DejaVu Sans')
    ax.text(0, -0.48, 'Placement Probability', ha='center', va='center',
            fontsize=11, color=TEXT2)

    # Zone labels
    for start, end, color, label in segments:
        mid   = (start + end) / 2
        angle = np.radians(180 - (mid / 100) * 180)
        r     = 1.22
        ax.text(r * np.cos(angle), r * np.sin(angle), label,
                ha='center', va='center', fontsize=7.5, color=TEXT2)

    # Scale ticks
    for pct in [0, 25, 50, 75, 100]:
        angle = np.radians(180 - (pct / 100) * 180)
        ax.text(0.72 * np.cos(angle), 0.72 * np.sin(angle),
                f'{pct}', ha='center', va='center', fontsize=8, color=TEXT2)

    ax.set_title('🎯 Placement Probability Gauge', fontsize=14,
                 fontweight='bold', pad=10, color=TEXT)
    plt.tight_layout()
    return _save(fig, 'dyn_gauge.png', out_dir)


# ══════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Weak Areas Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def graph_weak_areas(user_data: dict, df: pd.DataFrame, out_dir: str) -> str:
    _base_style()

    placed       = df[df['status'] == 'Placed']
    has_mba      = user_data.get('has_mba') == 'yes'
    is_technical = user_data.get('is_technical', 'yes') == 'yes'

    items = [
        ('SSC %',        float(user_data.get('ssc_p', 0)),          placed['ssc_p'].mean()),
        ('HSC %',        float(user_data.get('hsc_p', 0)),          placed['hsc_p'].mean()),
        ('Degree %',     float(user_data.get('degree_p', 0)),       placed['degree_p'].mean()),
        ('Etest %',      float(user_data.get('etest_p', 0)),        placed['etest_p'].mean()),
        ('Communication',float(user_data.get('communication', 0))*10, placed['communication'].mean()*10),
        ('Internships',  int(user_data.get('internships', 0))*33.3, placed['internships'].mean()*33.3),
        ('Projects',     int(user_data.get('projects', 0))*25,      placed['projects'].mean()*25),
    ]
    if is_technical:
        items.insert(4, ('Technical', float(user_data.get('technical', 4))*10, placed['technical'].mean()*10))
        items.insert(5, ('Coding Score', float(user_data.get('coding_score', 35)), placed['coding_score'].mean()))
    if has_mba:
        items.insert(3, ('MBA %', float(user_data.get('mba_p', 62)), placed['mba_p'].mean()))

    labels, user_vals, placed_vals = zip(*items)
    gaps   = [p - u for u, p in zip(user_vals, placed_vals)]
    colors = [NOT_CLR if g > 8 else '#F7C948' if g > 0 else PLACED_CLR for g in gaps]

    sorted_items = sorted(zip(gaps, labels, user_vals, placed_vals, colors), reverse=True)
    gaps_s, labels_s, user_s, placed_s, colors_s = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(labels_s))
    w = 0.36

    ax.barh(y + w/2, placed_s, w, color=PLACED_CLR, alpha=0.55,
            label='Avg Placed', edgecolor='none', zorder=3)
    bars = ax.barh(y - w/2, user_s,   w, color=colors_s, alpha=0.90,
                   label='Your Score', edgecolor='none', zorder=3)

    for bar, v, g in zip(bars, user_s, gaps_s):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{v:.0f}' + (f'  ▼{g:.0f}' if g > 0 else '  ✓'),
                va='center', fontsize=9,
                color=NOT_CLR if g > 0 else PLACED_CLR, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels_s, fontsize=10.5)
    ax.set_xlim(0, 120)
    ax.set_xlabel('Score (normalized to 100)', fontsize=11)
    ax.set_title('📋 Your Profile vs Avg Placed Student — Gap Analysis',
                 fontsize=14, fontweight='bold', pad=14, color=TEXT)
    ax.legend(framealpha=0.15, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    ax.xaxis.grid(True, zorder=0, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, 'dyn_weakareas.png', out_dir)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — Generate all 4 dynamic graphs for a prediction session
# ══════════════════════════════════════════════════════════════════════════
def generate_dynamic_graphs(user_data: dict, probability: float) -> dict:
    """
    Generates 4 personalized graphs for the result page.
    Returns a dict of graph filenames (relative to static/).
    """
    df = pd.read_csv('placement_dataset.csv')

    # Each prediction gets its own subfolder so concurrent users don't clash
    session_id = uuid.uuid4().hex[:10]
    out_dir    = os.path.join('static', 'user_graphs', session_id)
    os.makedirs(out_dir, exist_ok=True)

    graph_academic_comparison(user_data, df, out_dir)
    graph_skill_radar(user_data, df, out_dir)
    graph_probability_gauge(probability, out_dir)
    graph_weak_areas(user_data, df, out_dir)

    # Return web-accessible paths
    return {
        'academic':  f'user_graphs/{session_id}/dyn_academic.png',
        'radar':     f'user_graphs/{session_id}/dyn_radar.png',
        'gauge':     f'user_graphs/{session_id}/dyn_gauge.png',
        'weakareas': f'user_graphs/{session_id}/dyn_weakareas.png',
    }