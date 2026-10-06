import urllib.request
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

base = 'http://127.0.0.1:6699'

queries = [
    'Aerospace engineer specializing in CFD aerodynamics and rocket propulsion',
    'VLSI engineer specializing in Verilog RTL design and FPGA synthesis',
    'Robotics engineer with ROS2, autonomous SLAM and LiDAR perception'
]

for q in queries:
    req = urllib.request.Request(
        f'{base}/api/search/semantic',
        data=json.dumps({'query': q, 'min_similarity': 0.75, 'top_k': 3}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode())
        print(f'Query: "{q}"')
        print(f'  Matches (>= 75%): {data["matched_count"]}')
        for r in data['results']:
            print(f'   -> Track: {r["track"]} | Match: {r["similarity_score"]*100:.1f}% | Package: {r["salary_lpa"]} LPA | Has ID?: {"student_id" in r}')
