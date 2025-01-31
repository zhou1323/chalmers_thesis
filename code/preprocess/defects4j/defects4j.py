import os
import subprocess
import pandas as pd
from pathlib import Path
from git import Repo

import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from commit_utils import (
    parse_patch,
    extract_functions_and_classes_by_patch,
    apply_and_extract_with_commit,
    get_commit_log,
    get_commit_changes,
    get_jira_description,
    extract_test_methods_from_file,
)

from utils import save_to_json


def defects4j_checkout(base_path, project, bug_id, output_dir):
    """
    Use defect4j checkout to get the buggy version of a project.
    """
    # Determine whether the output directory exists
    if os.path.exists(output_dir):
        return

    print(f"Checking out {project} bug {bug_id}...")
    subprocess.run(
        [
            f"{base_path}/framework/bin/defects4j",
            "checkout",
            "-p",
            project,
            "-v",
            f"{bug_id}b",  # buggy version
            "-w",
            output_dir,  # Specify the output directory
        ],
        check=True,
    )


def apply_and_extract_with_patch(repo_path, changes, patch_file, to_get_bugs=False):
    """
    Apply the patch and extract the code structure.
    """
    if not os.path.exists(patch_file):
        return {}

    extracted_data = {}

    # If reverse is True, apply the patch in reverse
    try:
        subprocess.run(
            (
                ["git", "apply", "--reverse", patch_file]
                if not to_get_bugs
                else ["git", "apply", patch_file]
            ),
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch {patch_file}: {e}")
    except Exception as e:
        print(f"Common Error applying patch {patch_file}: {e}")

    extracted_data = extract_functions_and_classes_by_patch(
        repo_path, changes, not to_get_bugs
    )

    return extracted_data


def process_bug(base_path, projects_path, project, bug, to_use_patch=False, repo=None):
    """
    Process a single bug in a project.
    """
    bug_id = bug["bug_id"]

    print(f"\nProcessing {project} bug {bug_id}...")

    project_path = os.path.join(projects_path, project)

    if to_use_patch:
        repo_path = os.path.join(project_path, f"{project}_bug_{bug_id}")
        src_patch_file = os.path.join(project_path, "patches", f"{bug_id}.src.patch")
        test_patch_file = os.path.join(project_path, "patches", f"{bug_id}.test.patch")

        # Checkout the buggy version
        defects4j_checkout(base_path, project, bug_id, repo_path)
        # Function code processing
        functions_changes = parse_patch(src_patch_file, is_file_name=True)

        print("Processing the code before the patch...")
        functions_before = apply_and_extract_with_patch(
            repo_path, functions_changes, src_patch_file, to_get_bugs=True
        )
        print("Processing the code after the patch...")
        functions_after = apply_and_extract_with_patch(
            repo_path, functions_changes, src_patch_file
        )

        # Test case processing
        test_cases_changes = parse_patch(src_patch_file, is_file_name=True)

        print("Processing the test cases before the patch...")
        test_cases_before = apply_and_extract_with_patch(
            repo_path, test_cases_changes, test_patch_file, to_get_bugs=True
        )
        print("Processing the test cases after the patch...")

        test_cases_after = apply_and_extract_with_patch(
            repo_path, test_cases_changes, test_patch_file
        )

    else:
        commit_hash = bug["revision_fixed"]
        patch_content = get_commit_changes(repo, commit_hash)

        changes = parse_patch(patch_content, is_file_name=False)

        print("Processing the code before the commit...")
        functions_before, test_cases_before = apply_and_extract_with_commit(
            repo, changes, bug["revision_buggy"], to_get_bugs=True
        )

        print("Processing the code after the commit...")
        functions_after, test_cases_after = apply_and_extract_with_commit(
            repo, changes, commit_hash
        )
        description_log = get_commit_log(repo, bug["revision_fixed"])
        description_question = get_jira_description(bug["url"])

        relavan_test_cases_file = os.path.join(
            project_path, "relevant_tests", str(bug_id)
        )
        relevant_test_cases = get_relevant_test_cases(relavan_test_cases_file, repo)

    # Organize the data
    bug_data = {
        "source": f"defects4j_{project}_{bug_id}",
        "description_commit": description_log,
        "description_question": description_question,
        "function_codes_before": functions_before,
        "function_codes_after": functions_after,
        "test_cases_before": test_cases_before,
        "test_cases_after": test_cases_after,
        "relevant_test_cases": relevant_test_cases,
    }

    print(f"Processed {project} bug {bug_id}.\n")
    return bug_data


def get_relevant_test_cases(file_path, repo):
    """
    Get the relevant test cases.

    Returns:
        Dict: {
            test_case_name: { path: xxx, code: test_case_code },
            ...
    """
    relevant_test_case_files = []
    # Get the relevant test cases
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            relevant_test_case_files = file.readlines()

    # Get the detailed code of test cases
    test_cases = {}
    for test_case_file in relevant_test_case_files:
        test_case_file = test_case_file.strip()

        package_path = test_case_file.replace(".", os.sep) + ".java"

        full_path = os.path.join(repo.working_dir, "src", "test", "java", package_path)
        test_cases.update(extract_test_methods_from_file(full_path))

    return test_cases


def get_active_bugs(project_path):
    """
    Read bug IDs and revision IDs from active-bugs.csv

    Returns:
        List of dicts: [
            {
                'bug_id': str,
                'revision_buggy': str,
                'revision_fixed': str,
                'report_id': str,
                'url': str
            },
            ...
        ]
    """
    csv_path = os.path.join(project_path, "active-bugs.csv")
    try:
        df = pd.read_csv(csv_path)

        # Map CSV columns to desired names
        column_mapping = {
            "bug.id": "bug_id",
            "revision.id.buggy": "revision_buggy",
            "revision.id.fixed": "revision_fixed",
            "report.id": "report_id",
            "report.url": "url",
        }

        # Select and rename columns
        bugs_info = df[list(column_mapping.keys())].rename(columns=column_mapping)

        # Convert to list of dictionaries and sort by bug_id
        return sorted(bugs_info.to_dict("records"), key=lambda x: int(x["bug_id"]))
    except Exception as e:
        print(f"Error reading active-bugs.csv: {e}")
        return []


def process_defects4j(base_path, output_path):
    """
    Handle all defects4j projects and bugs.
    """
    testing = True
    filtering = True

    # Whether to use the patch files
    to_use_patch = False

    projects_path = Path(base_path, "framework", "projects")
    projects = [
        name
        for name in os.listdir(projects_path)
        if os.path.isdir(os.path.join(projects_path, name))
    ]
    for project in projects:
        # # TODO: For testing
        if testing and filtering and project != "Math":
            continue
        all_data = []

        project_path = os.path.join(projects_path, project)
        patch_dir = os.path.join(project_path, "patches")
        output_file = os.path.join(output_path, f"{project}.json")

        repo = (
            None
            if to_use_patch
            else Repo(os.path.join(base_path, "project_repos", f"repo_{project}"))
        )

        if not os.path.exists(patch_dir):
            continue

        # Get bug IDs from active-bugs.csv
        bugs = get_active_bugs(project_path)

        for bug in bugs:
            # # TODO: for testing!
            # if testing and filtering and bug["bug_id"] != 1:
            #     continue
            try:
                bug_data = process_bug(
                    base_path,
                    projects_path,
                    project,
                    bug,
                    to_use_patch=to_use_patch,
                    repo=repo,
                )
                bug_data["id"] = len(all_data)
                all_data.append(bug_data)
            except Exception as e:
                print(f"Error processing {project} bug {bug['bug_id']}: {e}")

        save_to_json(all_data, output_file)

        # todo: For testing
        # if testing and len(bugs) != len(all_data):
        #     print(project)
        #     if project == "Math":
        #         continue
        #     break


if __name__ == "__main__":
    # Target information:
    # - id: xx
    # - source: defects4j_xx
    # - description_question: xx
    # - description_commit: xx
    # - url: xx
    # - function_codes_before: xx
    # - function_codes_after: xx
    # - test_cases_before: Used to train the outdated test case
    # - test_cases_after: Used to train the aligned test case
    # - relevant_test_cases: Used to train the unaligned test case

    defects4j_path = "data/defects4j"
    output_path = "data/processed_defects4j"

    # 1. Iterate through all projects [Names of directories in defects4j/framework]
    # 1.1. Iterate through all bugs [In project's directory/active-bugs.csv]
    # 1.1.1. Get the location for original and changed code and test cases [In project's directory/patches]
    # 1.1.2. Get the source code with entire methods and test cases based on the location [function_codes_before, test_cases_before]
    # 1.1.3. Get the changed code with entire methods and test cases based on the location [function_codes_after, test_cases_after]
    # 1.1.4. Get the url. And based on the information in that url, extract the description of the commit [description_question, url]
    # 2. Save the information in a JSON file
    process_defects4j(defects4j_path, output_path)
