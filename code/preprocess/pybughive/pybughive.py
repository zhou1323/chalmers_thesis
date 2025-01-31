# Target information:
# - id: xx
# - source: pybughive_xx
# - description_question: xx
# - description_commit: xx
# - url: xx
# - function_codes_before: xx
# - function_codes_after: xx
# - test_cases_before: Used to train the outdated test case
# - test_cases_after: Used to train the aligned test case
# - relevant_test_cases: Used to train the unaligned test case

# 1. Download different repos based on the pybughive dataset
# 2. Extract the information for each bug in the repo
# 2.1. Get the description of the issue.
# 2.1. Get the location for original and changed code and test cases through the git history. And get the commit log[patches]
# 2.2. Get the source code with entire methods and test cases based on the location [function_codes_before, test_cases_before]
# 2.3. Get the changed code with entire methods and test cases based on the location [function_codes_after, test_cases_after]
# 3. Save the information in a JSON file

import os
from git import Repo
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from commit_utils import (
    get_commit_changes,
    get_commit_log,
    parse_patch,
    apply_and_extract_with_commit,
    find_test_cases_by_python_code_lines,
)

from utils import save_to_json, load_json

# Constants
DATASET_PATH = "data/Pybughive"
VERIFIED_DATASET_FILE = "pybughive_current.json"
UNVERIFIED_DATASET_FILE = "pybughivex_current.json"

OUTPUT_DIR = "data/processed_pybughive"
JSON_OUTPUT_FILE = "pybughive_bugs.json"


# Clone a GitHub repository
def clone_repo(username, repository, output_dir):
    repo_url = f"https://github.com/{username}/{repository}.git"
    local_path = os.path.join(output_dir, repository)
    if not os.path.exists(local_path):
        print(f"Cloning repository {repository}...")
        Repo.clone_from(repo_url, local_path)
    return local_path


def process_bug(repo, issue):
    bug_id = issue["id"]
    print(f"Processing {repo.working_dir} bug {bug_id}...")

    for commit in issue["commits"]:
        commit_hash = commit["hash"]
        patch_content = get_commit_changes(repo, commit_hash)

        changes = parse_patch(patch_content, is_file_name=False, language="python")

        print("Processing the code before the commit...")
        functions_before, test_cases_before = apply_and_extract_with_commit(
            repo, changes, commit["parents"], to_get_bugs=True, language="python"
        )

        print("Processing the code after the commit...")
        functions_after, test_cases_after = apply_and_extract_with_commit(
            repo, changes, commit_hash, language="python"
        )
        description_log = get_commit_log(repo, commit_hash)
        description_question = issue["title"]

        # Get relavant test cases
        function_code_lines = {}
        for file_name, function_codes in functions_after.items():
            for function_code in function_codes:
                if file_name in function_code_lines:
                    function_code_lines[file_name].append(
                        (function_code["start_line"], function_code["end_line"])
                    )
                else:
                    function_code_lines[file_name] = [
                        (function_code["start_line"], function_code["end_line"])
                    ]

        test_case_names = []
        for test_cases in test_cases_after.values():
            for test_case in test_cases:
                if isinstance(test_case, dict) and "name" in test_case:
                    test_case_names.append(test_case["name"])

        test_directory = (
            list(test_cases_before.keys()) + list(test_cases_after.keys())
        )[0].split("/tests/")[0] + "/tests/"

        test_cases = find_test_cases_by_python_code_lines(
            test_directory, function_code_lines, filtered_cases=test_case_names
        )

    # Organize the data
    bug_data = {
        "source": f"pybughive_{repo.working_dir.split("/")[-1]}_{bug_id}",
        "description_commit": description_log,
        "description_question": description_question,
        "function_codes_before": functions_before,
        "function_codes_after": functions_after,
        "test_cases_before": test_cases_before,
        "test_cases_after": test_cases_after,
        "relevant_test_cases": test_cases,
    }

    return bug_data


# Process each issue in a repository
def process_issues(repo_path, issues):
    repo = Repo(repo_path)
    bugs_info = []
    for issue in issues:
        try:
            bug_data = process_bug(repo, issue)
            bug_data["id"] = len(bugs_info)
            bugs_info.append(bug_data)
        except Exception as e:
            print(f"Error processing {repo_path} bug {issue.id}: {e}")
    return bugs_info


# Main function
def main():
    verified_dataset = load_json(DATASET_PATH, VERIFIED_DATASET_FILE)
    unverified_dataset = load_json(DATASET_PATH, UNVERIFIED_DATASET_FILE)

    dataset = verified_dataset + unverified_dataset
    dataset = dataset[:10]  # todo: testing!
    all_bugs_info = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in dataset:
        username = entry["username"]
        repository = entry["repository"]
        issues = entry["issues"]

        print(f"Processing repository {repository}...")
        try:
            repo_path = clone_repo(username, repository, DATASET_PATH)
            bugs_info = process_issues(repo_path, issues)
            all_bugs_info.extend(bugs_info)
        except Exception as e:
            print(f"Error processing repository {repository}: {e}")

        # todo: testing!
        break

    save_to_json(all_bugs_info, os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILE))
    print(f"Saved bug information to {JSON_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
