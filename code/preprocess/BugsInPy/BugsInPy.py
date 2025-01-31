import os
import subprocess
import pandas as pd
from pathlib import Path
from git import Repo

import re
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from commit_utils import (
    parse_patch,
    apply_and_extract_with_commit,
    get_commit_log,
    get_commit_changes,
    get_relevant_test_cases,
    extract_repo_info,
    clone_repo,
)

from utils import save_to_json, load_json

PROJECT_INFO_CONFIG = "project_info_config.json"


def process_bug(
    project,
    bug_id,
    buggy_commit_id,
    fixed_commit_id,
    repo,
    project_info_config,
    requirements_file,
):
    """
    Process a single bug in a project.
    """

    patch_content = get_commit_changes(repo, fixed_commit_id, buggy_commit_id)

    changes = parse_patch(patch_content, is_file_name=False, language="python")

    print("Processing the code before the commit...")
    functions_before, test_cases_before = apply_and_extract_with_commit(
        repo, changes, buggy_commit_id, to_get_bugs=True, language="python"
    )

    print("Processing the code after the commit...")
    functions_after, test_cases_after = apply_and_extract_with_commit(
        repo, changes, fixed_commit_id, language="python"
    )
    description_log = get_commit_log(repo, fixed_commit_id)

    # Get relavant test cases
    relevant_test_cases = get_relevant_test_cases(
        functions_after,
        test_cases_before,
        test_cases_after,
        project_info_config,
        requirements_file,
    )

    # Organize the data
    bug_data = {
        "source": f"bugsinpy_{project}_{bug_id}",
        "description_commit": description_log,
        "description_question": "",
        "function_codes_before": functions_before,
        "function_codes_after": functions_after,
        "test_cases_before": test_cases_before,
        "test_cases_after": test_cases_after,
        "relevant_test_cases": relevant_test_cases,
    }
    return bug_data


def process_bugsInPy(base_path, output_path, only_clone_repo=False):
    projects_path = Path(base_path, "projects")
    projects = [
        name
        for name in os.listdir(projects_path)
        if os.path.isdir(os.path.join(projects_path, name))
    ]

    project_info_config = load_json(
        os.path.join(os.getcwd(), "code/preprocess"), PROJECT_INFO_CONFIG
    )

    for project in projects:
        all_data = []

        project_path = os.path.join(projects_path, project)

        output_file = os.path.join(output_path, f"{project}.json")

        project_info_file = os.path.join(project_path, "project.info")

        with open(project_info_file, "r") as f:
            project_info = f.read().splitlines()
            username, repository = extract_repo_info(project_info[0])
            repo_path = clone_repo(username, repository, project_path)

        if only_clone_repo or os.path.exists(output_file):
            continue

        repo = Repo(repo_path)

        # List all directories in the project
        bug_folders = [
            os.path.join(project_path, "bugs", name)
            for name in sorted(os.listdir(os.path.join(project_path, "bugs")), key=int)
        ]

        for bug_folder in bug_folders:
            # if bug_folder.split("/")[-1] != "1":
            #     continue
            with open(os.path.join(bug_folder, "bug.info"), "r") as f:
                bug_info = f.read()

                buggy_commit_match = re.search(
                    r'buggy_commit_id="([a-f0-9]+)"', bug_info
                )
                fixed_commit_match = re.search(
                    r'fixed_commit_id="([a-f0-9]+)"', bug_info
                )

                buggy_commit_id = (
                    buggy_commit_match.group(1) if buggy_commit_match else None
                )
                fixed_commit_id = (
                    fixed_commit_match.group(1) if fixed_commit_match else None
                )
            try:

                print(f"\nProcessing {bug_folder}...")
                bug_data = process_bug(
                    project,
                    len(all_data),
                    buggy_commit_id,
                    fixed_commit_id,
                    repo=repo,
                    project_info_config=project_info_config,
                    requirements_file=os.path.join(
                        os.getcwd(), bug_folder, "requirements.txt"
                    ),
                )
                bug_data["id"] = len(all_data)
                all_data.append(bug_data)
                print(f"Processed {bug_folder}. \n")
            except Exception as e:
                print(f"Error processing bug {bug_folder}: {e}")

        save_to_json(all_data, output_file)


if __name__ == "__main__":

    bugsInPy_path = "data/BugsInPy"
    output_path = "data/processed_BugsInPy"

    process_bugsInPy(bugsInPy_path, output_path, only_clone_repo=False)
