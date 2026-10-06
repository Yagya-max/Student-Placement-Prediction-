import urllib.request
import json
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test(url, post_data=None):
    if post_data:
        req = urllib.request.Request(
            url, 
            data=json.dumps(post_data).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
    else:
        req = urllib.request.Request(url)
    res = urllib.request.urlopen(req)
    return json.loads(res.read().decode('utf-8'))

print("=== 1. OVERVIEW ===")
ov = test("http://127.0.0.1:6699/api/overview")
print(f"Students: {ov['total_records']}, Mean Salary: {ov['mean_salary']} ({ov['mean_lpa']} LPA), Median: {ov['median_salary']} ({ov['median_lpa']} LPA)")
print(f"Demographic: {ov.get('target_demographic')}")
print(f"Model R²: {ov.get('model_r2')} | MAE: {ov.get('model_mae')}")

print("\n=== 2. EDA CLEANING & IQR ===")
cl = test("http://127.0.0.1:6699/api/eda/cleaning")
print(f"IQR Explanation present: {bool(cl.get('iqr_explanation'))}")
print(f"Snippet: {cl.get('iqr_explanation')[:160]}...")
print(f"Clean records: {cl.get('clean_records')}, Outliers flagged: {cl.get('outliers_detected')}")

print("\n=== 3. ADVISORY ===")
adv = test("http://127.0.0.1:6699/api/advisory/evaluate", {
    "branch": "Computer Science & Engineering",
    "track": "Artificial Intelligence & Machine Learning",
    "cgpa": 8.5,
    "internships": 2,
    "projects": 3,
    "hackathons": 1,
    "certifications": 2,
    "skills": ["PyTorch & Deep Neural Networks", "Natural Language Processing (NLP)", "Large Language Models (LLMs & GenAI)"],
    "communication": 80,
    "aptitude": 85
})
print(f"Predicted Salary: {adv['predicted_salary_formatted']} ({adv['salary_lpa']} LPA) | Tier: {adv['tier']}")
print(f"Is Exceptional (>15L): {adv['is_exceptional']}")
print(f"Specialization Advice: {adv['specialization_advice'][:130]}...")
print(f"Target Companies: {list(adv['target_companies'].keys())}")
for category, companies in adv['target_companies'].items():
    print(f"   * {category}: {', '.join(companies[:4])}")

print("\n=== 4. SEMANTIC SEARCH (75%+ FILTER & NO ID TAGS) ===")
sr = test("http://127.0.0.1:6699/api/search/semantic", {"query": "PyTorch Deep Neural Networks NLP Large Language Models", "min_similarity": 0.75, "top_k": 5})
print(f"Latency: {sr['latency_ms']} ms | Total returned matching >= 75%: {len(sr['results'])}")
for idx, c in enumerate(sr['results']):
    has_id = "student_id" in c or "IND-" in str(c)
    pct = round(c['similarity_score'] * 100, 1)
    print(f"  #{idx+1} Match: {pct}% | Branch: {c['branch']} | Track: {c['track']} | Salary: {c['salary_formatted']} ({c['salary_lpa']}L) | Has ID tag: {has_id}")
    print(f"      Matched Skills: {', '.join(c['skills'][:4])}")

print("\n=== 5. CAREER PATHWAY & NOVEL EDA (SECTION 10) ===")
pw = test("http://127.0.0.1:6699/api/pathway/career?track=Artificial%20Intelligence%20%26%20Machine%20Learning")
print(f"Track: {pw['track']}")
print(f"Role succession stages: {len(pw['pathway'])}")
for role in pw['pathway']:
    print(f"  - {role['title']} | Time: {role['duration']} | Package: {role['comp_india']}")
    print(f"    Milestone: {role['milestone']}")
print(f"Language Advice: {pw['language_advice']['primary']}")
print(f"Language Reasoning: {pw['language_advice']['reasoning']}")
print(f"Mobility India: {pw['mobility_strategy']['stay_in_india']}")
print(f"Mobility Abroad: {pw['mobility_strategy']['try_abroad']}")
