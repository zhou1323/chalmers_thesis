# 1. Read the json
# 2. Get the installSteps
# 3. Get the python version
# 4. Save the command

import sys
import re
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import load_json


def extract_python_verison(text):
    match = re.search(r"pipenv\s+--python\s+(\d+\.\d+)", text)

    if match:
        python_version = match.group(1)  # 提取版本号
        return python_version
    else:
        return ""


# main function
if __name__ == "__main__":
    # read the json
    repos = load_json("data/Pybughive", "pybughive_current.json")
    result = {}
    for repo in repos:
        repo_name = repo["repository"]

        # get the installSteps
        installSteps = repo["installSteps"]
        # get the python version
        python_version = extract_python_verison(installSteps)

        result[repo_name] = {
            "python_version": python_version,
            "install_steps": installSteps,
        }

    print(result)
