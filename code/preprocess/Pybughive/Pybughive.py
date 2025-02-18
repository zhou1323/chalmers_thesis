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
    get_relevant_test_cases,
    clone_repo,
)

from utils import save_to_json, load_json

# Constants
DATASET_PATH = "data/Pybughive"
VERIFIED_DATASET_FILE = "pybughive_current.json"
UNVERIFIED_DATASET_FILE = "pybughivex_current.json"
PROJECT_INFO_CONFIG = "project_info_config.json"

OUTPUT_DIR = "data/processed_Pybughive"


def process_bug(repo, issue, project_info_config, need_relevant_test_cases=True):
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
        relevant_test_cases = (
            get_relevant_test_cases(
                functions_after,
                test_cases_before,
                test_cases_after,
                project_info_config,
            )
            if need_relevant_test_cases
            else {}
        )

    # Organize the data
    bug_data = {
        "source": f"pybughive_{repo.working_dir.split('/')[-1]}_{bug_id}",
        "commit_hash": commit_hash,
        "description_commit": description_log,
        "description_question": description_question,
        "function_codes_before": functions_before,
        "function_codes_after": functions_after,
        "test_cases_before": test_cases_before,
        "test_cases_after": test_cases_after,
        "relevant_test_cases": relevant_test_cases,
    }

    return bug_data


# Process each issue in a repository
def process_issues(
    repo_path, issues, project_info_config, need_relevant_test_cases=True
):
    repo = Repo(repo_path)
    bugs_info = []
    for issue in issues:
        try:
            bug_data = process_bug(
                repo,
                issue,
                project_info_config,
                need_relevant_test_cases=need_relevant_test_cases,
            )
            bug_data["id"] = len(bugs_info)
            bugs_info.append(bug_data)
        except Exception as e:
            print(f"Error processing {repo_path} bug {issue['id']}: {e}")

        # break  # todo: remove this line

    return bugs_info


# Main function
def main():
    verified_dataset = load_json(DATASET_PATH, VERIFIED_DATASET_FILE)
    unverified_dataset = load_json(DATASET_PATH, UNVERIFIED_DATASET_FILE)
    unverified_dataset = []  # todo: use unverified dataset

    only_clone_repo = False

    dataset = verified_dataset + unverified_dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    project_info_config = load_json(
        os.path.join(os.getcwd(), "code/preprocess"), PROJECT_INFO_CONFIG
    )

    for entry in dataset:

        all_bugs_info = []
        username = entry["username"]
        repository = entry["repository"]
        issues = entry["issues"]

        output_file = os.path.join(OUTPUT_DIR, f"{repository}.json")

        print(f"Processing repository {repository}...")

        if repository == "jax":
            continue
        try:
            repo_path = clone_repo(username, repository, DATASET_PATH)

            if only_clone_repo or os.path.exists(output_file):
                continue

            bugs_info = process_issues(
                repo_path, issues, project_info_config, need_relevant_test_cases=False
            )
            all_bugs_info.extend(bugs_info)
        except Exception as e:
            print(f"Error processing repository {repository}: {e}")

        save_to_json(all_bugs_info, output_file)
        print(f"Saved bug information to {output_file}")


if __name__ == "__main__":
    main()
