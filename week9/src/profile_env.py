import platform
import json
import psutil
import os
import sys
sys.path.append('src')
from salt import get_student_params

cpu_info = {
    "processor": platform.processor(),
    "cores_physical": psutil.cpu_count(logical=False),
    "cores_logical": psutil.cpu_count(logical=True),
    "os": platform.system() + " " + platform.release(),
    "python": sys.version
}

student_params = get_student_params("230103191")

os.makedirs("results", exist_ok=True)
with open("results/host_profile.json", "w") as f:
    json.dump(cpu_info, f, indent=4)
with open("results/salted_params.json", "w") as f:
    json.dump(student_params, f, indent=4)

print(f"Environment profiled. Matrix size N = {student_params['N']}")
