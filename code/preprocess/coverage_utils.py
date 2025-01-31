import coverage
import subprocess
import os
from utils import save_to_json, load_json


def install_requirements(directory):
    """
    Install the requirements for the coverage tool.
    """
    if not os.path.exists(os.path.join(directory, "requirements.txt")):
        subprocess.run(["pip", "install", "."], cwd=directory)
    else:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=directory)


def run_coverage(file_path):
    """
    Runs the coverage tool on the codebase.
    """
    current_dir = os.path.join(os.getcwd(), "code/preprocess")
    coverage_file_dir = os.path.dirname(file_path)

    install_requirements(coverage_file_dir)

    subprocess.run(
        [
            "coverage",
            "run",
            f"--rcfile={os.path.join(current_dir, '.coveragerc')}",
            "-m",
            "pytest",
            "tests/",
        ],
        cwd=coverage_file_dir,
    )


def extract_coverage_report(file_dir, target_dir, redo_extraction=False):
    """
    Extracts the coverage report from the coverage data file.
    Args:
        file_path: The path to the coverage data file.
    """
    target_test2line_file = os.path.join(target_dir, "test_cases_to_lines.json")
    target_line2test_file = os.path.join(target_dir, "line_to_test_cases.json")

    file_path = os.path.join(file_dir, ".coverage")

    # Determine whether the target file exists.
    if (
        os.path.exists(target_test2line_file)
        and os.path.exists(target_line2test_file)
        and not redo_extraction
    ):
        print("The coverage report has already been extracted.")
        return

    # Determine whether the coverage data file exists.
    if redo_extraction or not os.path.exists(file_path):
        run_coverage(file_path)

    cov = coverage.Coverage(data_file=file_path)
    cov.load()
    covdata = cov.get_data()

    contexts = covdata.measured_contexts()
    if not contexts:
        print("There are no measured contexts in the coverage data.")

    # test_cases_to_lines: A dictionary containing the test cases and the lines they covered.
    #  {test_func: {file: [lines]}}
    test_cases_to_lines = {}

    # line_to_test_cases: A dictionary containing the lines and the test cases that covered them.
    # {file: { line: [test_cases] }}
    line_to_test_cases = {}

    for file in covdata.measured_files():
        line_to_contexts = covdata.contexts_by_lineno(file)

        for line, context_set in line_to_contexts.items():
            for context in context_set:
                if not context:
                    continue

                # Initialize line_to_test_cases
                line_to_test_cases.setdefault(file, {}).setdefault(line, [])

                # Initialize test_cases_to_lines
                test_cases_to_lines.setdefault(context, {}).setdefault(file, [])

                test_cases_to_lines[context][file].append(line)
                line_to_test_cases[file][line].append(context)

    save_to_json(test_cases_to_lines, target_test2line_file)
    save_to_json(line_to_test_cases, target_line2test_file)


def load_test2line_json(target_dir):
    """
    Load the test_cases_to_lines file from the target directory.
    """
    return load_json(target_dir, "test_cases_to_lines.json")


def load_line2test_json(target_dir):
    """
    Load the line_to_test_cases file from the target directory.
    """
    return load_json(target_dir, "line_to_test_cases.json")
