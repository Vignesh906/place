import pandas as pd
import numpy as np
import joblib

def load_model():
    model    = joblib.load('placement_model.pkl')
    scaler   = joblib.load('placement_scaler.pkl')
    features = joblib.load('placement_features.pkl')
    return model, scaler, features

def generate_feedback(data: dict) -> list:
    """Generate smart personalized improvement tips."""
    tips = []
    icons = {
        'comm':   '🗣️', 'tech':   '⚙️', 'intern': '🏢',
        'proj':   '💡', 'code':   '💻', 'etest':  '📝',
        'degree': '🎓', 'mba':    '📊', 'workex': '💼',
        'nontech':'📋',
    }

    is_technical = data.get('is_technical', 'yes') == 'yes'
    comm     = float(data.get('communication', 5))
    code     = float(data.get('coding_score', 35)) if is_technical else 35
    tech     = float(data.get('technical', 4))     if is_technical else 4
    interns  = int(data.get('internships', 0))
    projs    = int(data.get('projects', 0))
    etest    = float(data.get('etest_p', 0))
    degree   = float(data.get('degree_p', 0))
    mba      = float(data.get('mba_p', 62))
    workex   = data.get('workex', 'No')
    has_mba  = data.get('has_mba') == 'yes'

    # Communication — for everyone
    if comm < 4:
        tips.append({'icon': icons['comm'], 'level': 'high',
            'text': 'Communication score is critically low. Join debate clubs, public speaking workshops, or practice mock interviews daily.'})
    elif comm < 6:
        tips.append({'icon': icons['comm'], 'level': 'medium',
            'text': 'Communication skills need improvement. Practice presentations and group discussions regularly.'})
    elif comm < 8:
        tips.append({'icon': icons['comm'], 'level': 'low',
            'text': 'Communication is decent — polishing it further will give you an edge in interviews.'})

    # Technical & Coding — only for technical students
    if is_technical:
        if tech < 4:
            tips.append({'icon': icons['tech'], 'level': 'high',
                'text': 'Technical skills are very weak. Focus on your core subject area and take online courses on Coursera or Udemy immediately.'})
        elif tech < 6:
            tips.append({'icon': icons['tech'], 'level': 'medium',
                'text': 'Improve technical skills through hands-on projects and domain-specific certifications.'})

        if code < 40:
            tips.append({'icon': icons['code'], 'level': 'high',
                'text': 'Coding score is critically low. Practice daily on LeetCode, HackerRank, or CodeChef — even 30 mins/day makes a difference.'})
        elif code < 60:
            tips.append({'icon': icons['code'], 'level': 'medium',
                'text': 'Coding score needs improvement. Focus on data structures, algorithms, and SQL practice.'})
    else:
        # Non-technical specific tips
        tips.append({'icon': icons['nontech'], 'level': 'low',
            'text': 'As a non-technical student, focus on domain knowledge, Excel/data tools, business communication, and industry certifications relevant to your field.'})

    # Internships — for everyone
    if interns == 0:
        tips.append({'icon': icons['intern'], 'level': 'high',
            'text': 'No internship experience — a significant red flag for recruiters. Apply to at least 1 internship immediately (even virtual ones count).'})
    elif interns == 1:
        tips.append({'icon': icons['intern'], 'level': 'low',
            'text': 'One internship is good. Try to get a second to boost your profile significantly.'})

    # Projects — for everyone
    if projs < 2:
        tips.append({'icon': icons['proj'], 'level': 'high' if projs == 0 else 'medium',
            'text': f'Only {projs} project(s) listed. Recruiters want to see at least 3–4 strong projects on your resume or GitHub/portfolio.'})
    elif projs < 4:
        tips.append({'icon': icons['proj'], 'level': 'low',
            'text': 'Add 1–2 more impactful projects aligned with your target role.'})

    # Entrance test — for everyone
    if etest < 50:
        tips.append({'icon': icons['etest'], 'level': 'high',
            'text': 'Entrance test score is very low. Prepare for aptitude tests: quantitative reasoning, logical reasoning, and verbal ability.'})
    elif etest < 65:
        tips.append({'icon': icons['etest'], 'level': 'medium',
            'text': 'Improve your aptitude/entrance test prep — companies heavily filter on this metric.'})

    # Degree percentage — for everyone
    if degree < 55:
        tips.append({'icon': icons['degree'], 'level': 'high',
            'text': 'Degree percentage is below the typical shortlisting threshold. Highlight other strengths to compensate.'})
    elif degree < 65:
        tips.append({'icon': icons['degree'], 'level': 'medium',
            'text': 'Degree percentage is moderate. Try to maintain a stronger CGPA going forward.'})

    # MBA — only if applicable
    if has_mba and mba < 55:
        tips.append({'icon': icons['mba'], 'level': 'medium',
            'text': 'MBA percentage is low — focus on improving your current semester performance.'})

    # No work experience AND no internships
    if workex == 'No' and interns == 0:
        tips.append({'icon': icons['workex'], 'level': 'high',
            'text': 'No work experience or internships detected. Pursue freelance projects, open-source contributions, or part-time roles immediately.'})

    if not tips:
        tips.append({'icon': '🌟', 'level': 'good',
            'text': 'Your profile looks strong! Focus on interview preparation, company research, and tailoring your resume to each role.'})

    return tips


def _safe_float(val, default=0.0):
    """Safely convert a form value to float, returning default if empty/invalid."""
    try:
        if val is None or str(val).strip() == '':
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_int(val, default=0):
    """Safely convert a form value to int, returning default if empty/invalid."""
    try:
        if val is None or str(val).strip() == '':
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default

def prepare_input(form_data: dict, features: list) -> pd.DataFrame:
    """Convert form data to model-ready DataFrame. MBA & technical fields are optional."""

    # MBA fields
    has_mba        = form_data.get('has_mba') == 'yes'
    mba_p_val      = _safe_float(form_data.get('mba_p'), 62.0) if has_mba else 62.0
    specialisation = form_data.get('specialisation') if has_mba and form_data.get('specialisation') else 'Mkt&HR'

    # Technical fields — use realistic non-technical defaults if not applicable
    is_technical   = form_data.get('is_technical', 'yes') == 'yes'
    technical_val  = _safe_float(form_data.get('technical'), 5.0) if is_technical else 4.0
    coding_val     = _safe_float(form_data.get('coding_score'), 50.0) if is_technical else 35.0

    row = {
        'ssc_p':         _safe_float(form_data.get('ssc_p'),        65.0),
        'hsc_p':         _safe_float(form_data.get('hsc_p'),        63.0),
        'degree_p':      _safe_float(form_data.get('degree_p'),     65.0),
        'etest_p':       _safe_float(form_data.get('etest_p'),      60.0),
        'mba_p':         mba_p_val,
        'communication': _safe_float(form_data.get('communication'), 5.0),
        'technical':     technical_val,
        'coding_score':  coding_val,
        'internships':   _safe_int(form_data.get('internships'),     0),
        'projects':      _safe_int(form_data.get('projects'),        0),
        # one-hot
        'gender_F':      1 if form_data.get('gender') == 'F' else 0,
        'gender_M':      1 if form_data.get('gender') == 'M' else 0,
        'hsc_s_Arts':    1 if form_data.get('hsc_s') == 'Arts'     else 0,
        'hsc_s_Commerce':1 if form_data.get('hsc_s') == 'Commerce' else 0,
        'hsc_s_Science': 1 if form_data.get('hsc_s') == 'Science'  else 0,
        'degree_t_Comm&Mgmt': 1 if form_data.get('degree_t') == 'Comm&Mgmt' else 0,
        'degree_t_Others':    1 if form_data.get('degree_t') == 'Others'    else 0,
        'degree_t_Sci&Tech':  1 if form_data.get('degree_t') == 'Sci&Tech'  else 0,
        'workex_No':     1 if form_data.get('workex') == 'No'  else 0,
        'workex_Yes':    1 if form_data.get('workex') == 'Yes' else 0,
        'specialisation_Mkt&Fin': 1 if specialisation == 'Mkt&Fin' else 0,
        'specialisation_Mkt&HR':  1 if specialisation == 'Mkt&HR'  else 0,
    }
    df = pd.DataFrame([row])
    for f in features:
        if f not in df.columns:
            df[f] = 0
    return df[features]


def get_model_stats() -> dict:
    """Return cached accuracy stats for dashboard."""
    return {
        'Random Forest':      {'accuracy': 73.6, 'f1': 77.1},
        'Logistic Regression':{'accuracy': 74.2, 'f1': 76.9},
        'KNN':                {'accuracy': 65.6, 'f1': 68.7},
        'best': 'Random Forest',
        'dataset_size': 1200,
        'features': 22,
        'train_size': 840,
        'test_size': 360,
    }