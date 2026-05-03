from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import os

router = APIRouter()

# ─── LOAD MODELS & ENCODERS ──────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

salary_model  = pickle.load(open(os.path.join(MODELS_DIR, 'salary_model.pkl'),  'rb'))
jobtype_model = pickle.load(open(os.path.join(MODELS_DIR, 'jobtype_model.pkl'), 'rb'))

le_role    = pickle.load(open(os.path.join(MODELS_DIR, 'le_role.pkl'),    'rb'))
le_city    = pickle.load(open(os.path.join(MODELS_DIR, 'le_city.pkl'),    'rb'))
le_exp     = pickle.load(open(os.path.join(MODELS_DIR, 'le_exp.pkl'),     'rb'))
le_company = pickle.load(open(os.path.join(MODELS_DIR, 'le_company.pkl'), 'rb'))
le_jobtype = pickle.load(open(os.path.join(MODELS_DIR, 'le_jobtype.pkl'), 'rb'))

all_features = pickle.load(open(os.path.join(MODELS_DIR, 'features.pkl'), 'rb'))

# ─── SKILL COLUMNS ───────────────────────────────────────
SKILL_COLS = [
    'ReactJS', 'NodeJS', 'Python', 'SQL', 'MongoDB', 'Git',
    'PWA', 'Socket_io', 'Scikit_learn', 'TensorFlow', 'LLM',
    'LangChain', 'Pandas', 'Agile', 'JIRA', 'BA', 'Project_Mgmt',
    'IT_Support', 'Documentation', 'Risk_Mgmt', 'Stakeholder_Mgmt',
    'Figma', 'User_Research', 'Wireframing', 'Prototyping',
    'Adobe_XD', 'Design_Systems', 'Usability_Testing', 'Interaction_Design'
]

# ─── REQUEST SCHEMA ───────────────────────────────────────
class PredictRequest(BaseModel):
    role:         str        # "FullStack", "ITManagement", "UIUXDesign", "AIEngineer"
    city:         str        # "Lahore", "Karachi" etc
    experience:   str        # "Fresh", "6Months", "1Year", "MoreThan1.5Years"
    company_size: str        # "Small", "Medium", "Large"
    job_type:     str        # "Onsite", "Remote"
    skills:       List[str]  # ["ReactJS", "Python", "SQL"] etc

# ─── HELPER — BUILD FEATURE VECTOR ───────────────────────
def build_features(data: PredictRequest):
    role_enc    = le_role.transform([data.role])[0]
    city_enc    = le_city.transform([data.city])[0]
    exp_enc     = le_exp.transform([data.experience])[0]
    company_enc = le_company.transform([data.company_size])[0]
    jobtype_enc = le_jobtype.transform([data.job_type])[0]

    # Skills → 0 or 1
    skill_vector = [1 if skill in data.skills else 0 for skill in SKILL_COLS]

    feature_vector = [role_enc, city_enc, exp_enc, company_enc, jobtype_enc] + skill_vector
    return np.array(feature_vector).reshape(1, -1)

# ─── ENDPOINT: PREDICT ───────────────────────────────────
@router.post("/predict")
def predict(data: PredictRequest):
    features = build_features(data)

    # Salary prediction
    predicted_salary = salary_model.predict(features)[0]
    predicted_salary = max(0, round(predicted_salary / 1000) * 1000)

    # Salary range (±15%)
    salary_min = round((predicted_salary * 0.85) / 1000) * 1000
    salary_max = round((predicted_salary * 1.15) / 1000) * 1000

    # Job type prediction
    jobtype_pred     = jobtype_model.predict(features)[0]
    jobtype_proba    = jobtype_model.predict_proba(features)[0]
    jobtype_label    = le_jobtype.inverse_transform([jobtype_pred])[0]
    jobtype_confidence = round(float(max(jobtype_proba)) * 100, 1)

    # Skill gap — missing important skills for role
    role_skill_map = {
        "FullStack":     ["ReactJS", "NodeJS", "MongoDB", "Git", "SQL"],
        "ITManagement":  ["Agile", "JIRA", "Project_Mgmt", "BA", "Documentation"],
        "UIUXDesign":    ["Figma", "Wireframing", "Prototyping", "User_Research", "Adobe_XD"],
        "AIEngineer":    ["Python", "Pandas", "Scikit_learn", "TensorFlow", "LLM"],
    }
    important_skills = role_skill_map.get(data.role, [])
    skill_gap = [s for s in important_skills if s not in data.skills]

    return {
        "salary": {
            "predicted":  int(predicted_salary),
            "min":        int(salary_min),
            "max":        int(salary_max),
            "currency":   "PKR",
            "period":     "monthly"
        },
        "job_type": {
            "predicted":   jobtype_label,
            "confidence":  jobtype_confidence,
            "onsite_prob": round(float(jobtype_proba[0]) * 100, 1),
            "remote_prob": round(float(jobtype_proba[1]) * 100, 1),
        },
        "skill_gap": skill_gap,
        "input_summary": {
            "role":         data.role,
            "city":         data.city,
            "experience":   data.experience,
            "company_size": data.company_size,
            "job_type":     data.job_type,
            "skills_count": len(data.skills)
        }
    }