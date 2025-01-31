import os
import json
import sys
from pathlib import Path
from git import Repo
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from commit_utils import (
    get_commit_changes,
    get_merge_commit_local,
    apply_and_extract_with_commit,
    parse_patch,
    extract_repo_info,
    find_test_cases_by_python_code_lines,
    clone_repo,
)
from utils import save_to_json

BUGL_PATH = "data/BuGL/Python/JSON"
PROJECTS_PATH = "data/BuGL/Python/Projects"
OUTPUT_PATH = "data/processed_BuGL"


# Step1: Filter out all closed issues
def filter_closed_issues(issues):
    return issues["closed_issues"]


# Step2: Filter out all issues with changed files
def filter_specific_files(issues):
    """
    Filter out issues that:
    1. Have changes in .md or .py files
    2. Have changes in test files
    """
    return [
        issue
        for issue in issues.values()
        if len(issue["files_changed"]) != 0
        and all(file[1].endswith((".md", ".py")) for file in issue["files_changed"])
        and any("tests" in file[1].lower() for file in issue["files_changed"])
    ]


# Step3: Go to the repo and extract the information
def process_bug(repo_path, issue):
    repo = Repo(repo_path)

    bug_id = issue["issue_id"]

    # 2016-08-18T00:28:50Z
    fixed_time = datetime.strptime(issue["issue_fixed_time"], "%Y-%m-%dT%H:%M:%SZ")
    since = fixed_time - timedelta(days=2)
    until = fixed_time + timedelta(days=2)

    merge_commit_hash = get_merge_commit_local(repo, issue["fixed_by"], since, until)

    merge_commit = repo.commit(merge_commit_hash)
    parent_commit = merge_commit.parents[0]

    patch_content = get_commit_changes(
        repo, merge_commit_hash, parent_commit_hash=parent_commit.hexsha
    )

    changes = parse_patch(patch_content, is_file_name=False, language="python")

    print("Processing the code before the commit...")
    functions_before, test_cases_before = apply_and_extract_with_commit(
        repo, changes, parent_commit.hexsha, to_get_bugs=True, language="python"
    )

    print("Processing the code after the commit...")
    functions_after, test_cases_after = apply_and_extract_with_commit(
        repo, changes, merge_commit_hash, language="python"
    )

    description_log = issue["pull_request_summary"]
    description_question = issue["issue_description"]

    # Get relavant test cases
    relevant_test_cases = get_relevant_test_cases(
        functions_after, test_cases_before, test_cases_after
    )

    # Organize the data
    bug_data = {
        "source": f"bugl_{repo.working_dir.split("/")[-1]}_{bug_id}",
        "description_commit": description_log,
        "description_question": description_question,
        "function_codes_before": functions_before,
        "function_codes_after": functions_after,
        "test_cases_before": test_cases_before,
        "test_cases_after": test_cases_after,
        "relevant_test_cases": relevant_test_cases,
    }

    return bug_data


def get_relevant_test_cases(functions_after, test_cases_before, test_cases_after):
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

    test_directory = (list(test_cases_before.keys()) + list(test_cases_after.keys()))[
        0
    ].split("/tests/")[0] + "/tests/"

    try:
        test_cases = find_test_cases_by_python_code_lines(
            test_directory, function_code_lines, filtered_cases=test_case_names
        )
    except Exception as e:
        print(f"Error finding test cases: {e}")
        test_cases = []
    return test_cases


def process_bugl(base_path, output_path, projects_path, only_clone_repo=False):

    for json_file in os.listdir(base_path):
        if not json_file.endswith(".json"):
            continue

        all_data = []

        json_path = os.path.join(base_path, json_file)

        with open(json_path, "r") as f:
            issues = json.load(f)

        issues = filter_closed_issues(issues)
        issues = filter_specific_files(issues)

        if (len(issues)) == 0:
            continue

        first_issue = issues[0]
        username, repository = extract_repo_info(first_issue["issue_url"])

        repo_path = clone_repo(username, repository, projects_path)

        if only_clone_repo:
            continue

        for issue in issues:
            bug_data = process_bug(repo_path, issue)
            bug_data["id"] = issue["issue_id"]
            all_data.append(bug_data)

        output_file = os.path.join(output_path, json_file)
        save_to_json(all_data, output_file)


if __name__ == "__main__":
    # 1. Filter out all closed issues
    # 2. Filter out all issues with changed files that end with .md or .py
    # 3. Filter out all issues with changed test files
    # 4. Go to the repo and extract the information

    process_bugl(BUGL_PATH, OUTPUT_PATH, PROJECTS_PATH, only_clone_repo=False)
