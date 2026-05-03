import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ─── PATHS ───────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'careerlens_dataset.csv')
MODELS_DIR  = os.path.dirname(__file__)

# ─── 1. LOAD DATA ────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df)}")

# ─── 2. ENCODE CATEGORICAL COLUMNS ───────────────────────
print("\nEncoding categorical columns...")

le_role     = LabelEncoder()
le_city     = LabelEncoder()
le_exp      = LabelEncoder()
le_company  = LabelEncoder()
le_jobtype  = LabelEncoder()

df['Role_enc']         = le_role.fit_transform(df['Role'])
df['City_enc']         = le_city.fit_transform(df['City'])
df['Experience_enc']   = le_exp.fit_transform(df['Experience'])
df['Company_Size_enc'] = le_company.fit_transform(df['Company_Size'])
df['Job_Type_enc']     = le_jobtype.fit_transform(df['Job_Type'])

print(f"Role encoding:        {dict(zip(le_role.classes_, le_role.transform(le_role.classes_)))}")
print(f"City encoding:        {dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))}")
print(f"Experience encoding:  {dict(zip(le_exp.classes_, le_exp.transform(le_exp.classes_)))}")
print(f"Company encoding:     {dict(zip(le_company.classes_, le_company.transform(le_company.classes_)))}")
print(f"Job_Type encoding:    {dict(zip(le_jobtype.classes_, le_jobtype.transform(le_jobtype.classes_)))}")

# ─── 3. DEFINE FEATURE COLUMNS ───────────────────────────
categorical_features = [
    'Role_enc', 'City_enc', 'Experience_enc',
    'Company_Size_enc', 'Job_Type_enc'
]

skill_features = [
    'ReactJS', 'NodeJS', 'Python', 'SQL', 'MongoDB', 'Git',
    'PWA', 'Socket_io', 'Scikit_learn', 'TensorFlow', 'LLM',
    'LangChain', 'Pandas', 'Agile', 'JIRA', 'BA', 'Project_Mgmt',
    'IT_Support', 'Documentation', 'Risk_Mgmt', 'Stakeholder_Mgmt',
    'Figma', 'User_Research', 'Wireframing', 'Prototyping',
    'Adobe_XD', 'Design_Systems', 'Usability_Testing', 'Interaction_Design'
]

all_features = categorical_features + skill_features

X = df[all_features]

# ─── 4. MODEL 1 — SALARY (LINEAR REGRESSION) ─────────────
print("\n--- Training Salary Model (Linear Regression) ---")

y_salary = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_salary, test_size=0.2, random_state=42
)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)

y_pred = salary_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : PKR {mae:,.0f}")
print(f"R2 Score            : {r2:.4f}")

# ─── 5. MODEL 2 — JOB TYPE (LOGISTIC REGRESSION) ─────────
print("\n--- Training Job Type Model (Logistic Regression) ---")

y_jobtype = df['Job_Type_enc']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_jobtype, test_size=0.2, random_state=42
)

jobtype_model = LogisticRegression(max_iter=1000)
jobtype_model.fit(X_train2, y_train2)

y_pred2  = jobtype_model.predict(X_test2)
accuracy = accuracy_score(y_test2, y_pred2)

print(f"Accuracy: {accuracy * 100:.2f}%")

# ─── 6. SAVE MODELS & ENCODERS ───────────────────────────
print("\nSaving models...")

pickle.dump(salary_model,  open(os.path.join(MODELS_DIR, 'salary_model.pkl'),  'wb'))
pickle.dump(jobtype_model, open(os.path.join(MODELS_DIR, 'jobtype_model.pkl'), 'wb'))

pickle.dump(le_role,     open(os.path.join(MODELS_DIR, 'le_role.pkl'),     'wb'))
pickle.dump(le_city,     open(os.path.join(MODELS_DIR, 'le_city.pkl'),     'wb'))
pickle.dump(le_exp,      open(os.path.join(MODELS_DIR, 'le_exp.pkl'),      'wb'))
pickle.dump(le_company,  open(os.path.join(MODELS_DIR, 'le_company.pkl'),  'wb'))
pickle.dump(le_jobtype,  open(os.path.join(MODELS_DIR, 'le_jobtype.pkl'),  'wb'))

pickle.dump(all_features, open(os.path.join(MODELS_DIR, 'features.pkl'),   'wb'))

print("\n All models saved successfully!")
print("   → salary_model.pkl")
print("   → jobtype_model.pkl")
print("   → le_role.pkl")
print("   → le_city.pkl")
print("   → le_exp.pkl")
print("   → le_company.pkl")
print("   → le_jobtype.pkl")
print("   → features.pkl")
print("\nTraining complete! ")