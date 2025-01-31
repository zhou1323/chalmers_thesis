import coverage
import subprocess
import os
import re
from utils import save_to_json, load_json


def run_coverage(
    file_path, all_test_files, requirements_file, project_info_config=None
):
    """
    Runs the coverage tool on the codebase.

    Args:
        file_path: The path to the coverage data file.
        all_test_files: A list of all test files to consider.
        requirements_file: The path to the requirements file.
        project_info_config: A dictionary containing the project information.

    Returns:
        None
    """
    current_dir = os.path.join(os.getcwd(), "code/preprocess")
    coverage_file_dir = os.path.dirname(file_path)

    match = re.search(r"/data/([^/]+)", coverage_file_dir)
    dataset = match.group(1) if match else ""
    project_name = coverage_file_dir.split("/")[-1]

    prepare_environment(
        coverage_file_dir, dataset, project_name, requirements_file, project_info_config
    )

    try:
        subprocess.run(
            f'bash -c "source activate base; conda activate {dataset}_{project_name}; coverage run --rcfile={os.path.join(current_dir, ".coveragerc")} -m pytest {" ".join(all_test_files)}"',
            shell=True,
            cwd=coverage_file_dir,
            timeout=60,  # 5 minute timeout
            # stdout=subprocess.PIPE,
        )

    except subprocess.TimeoutExpired:
        print(f"Coverage run timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"Error running coverage: {str(e)}")
        return None


def extract_coverage_report(
    file_dir,
    all_test_files,
    requirements_file,
    redo_extraction=False,
    project_info_config=None,
):
    """
    Extracts the coverage report from the coverage data file.
    Args:
        file_path: The path to the coverage data file.
        all_test_files: A list of all test files to consider.
        test_case_dir: The directory containing the test cases.
        redo_extraction: A boolean indicating whether to redo the extraction.
        project_info_config: A dictionary containing the project information.
    Returns:
        None
    """
    target_test2line_file = os.path.join(file_dir, "test_cases_to_lines.json")
    target_line2test_file = os.path.join(file_dir, "line_to_test_cases.json")

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
        run_coverage(file_path, all_test_files, requirements_file, project_info_config)

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

                parts = context.split(".")
                if len(parts) >= 2:
                    context = f"{parts[-2]}.{parts[-1]}"

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

    Args:
        target_dir: The directory containing the test_cases_to_lines file.

    Returns:
        A dictionary containing the test cases and the lines they covered.
    """
    return load_json(target_dir, "test_cases_to_lines.json")


def load_line2test_json(target_dir):
    """
    Load the line_to_test_cases file from the target directory.

    Args:
        target_dir: The directory containing the line_to_test_cases file.

    Returns:
        A dictionary containing the lines and the test cases that covered them.
    """
    return load_json(target_dir, "line_to_test_cases.json")


def prepare_environment(target_dir, dataset, project_name, requirements_file, config):
    """
    Prepare the environment for the project.

    Args:
        target_dir: The target directory for the project.
        dataset: The dataset name.
        project_name: The project name.
        config: The project configuration.

    Returns:
        A dictionary containing the environment variables.
    """
    print(f"\nSetting up project: {project_name}")
    env_name = f"{dataset}_{project_name}"
    python_version = config[dataset][project_name]["python_version"]

    create_conda_environment(env_name, python_version)

    install_steps = config[dataset][project_name].get("install_steps", None)

    install_requirements(
        env_name, target_dir, requirements_file, install_steps=install_steps
    )


def create_conda_environment(env_name, python_version):
    """Create a Conda environment."""
    try:
        # Check if the environment already exists
        existing_envs = subprocess.check_output(
            ["conda", "env", "list"], universal_newlines=True
        )
        if env_name in existing_envs:
            print(f"Virtual environment '{env_name}' already exists.")
            return

        print(
            f"Creating conda environment '{env_name}' with Python {python_version}..."
        )

        subprocess.run(
            ["conda", "create", "-n", env_name, f"python={python_version}"], check=True
        )
        print(f"Conda environment '{env_name}' created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment '{env_name}': {e}")


def install_requirements(env_name, target_dir, requirements_file, install_steps):
    """
    Install the requirements for the project.

    Args:
        env_name: The name of the virtual environment.
        target_dir: The target directory for the project.
        requirements_file: The path to the requirements file.
        install_steps: A list of install steps.

    Returns:
        None
    """

    if requirements_file and os.path.exists(requirements_file):
        try:
            print(
                f"Installing dependencies for '{env_name}' from {requirements_file}..."
            )
            subprocess.run(
                f'bash -c "source activate base; conda activate {env_name}; pip install -r {requirements_file}; pip install coverage pytest"',
                cwd=target_dir,
                shell=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            print(f"Dependencies installed successfully in '{env_name}'!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies in '{env_name}': {e}")
    elif install_steps:
        try:
            print(f"Installing dependencies for '{env_name}' from command...")
            subprocess.run(
                f'bash -c "source activate base; conda activate {env_name}; {install_steps}; pip install coverage pytest"',
                cwd=target_dir,
                shell=True,
                check=True,
            )
            print(f"Dependencies installed successfully in '{env_name}'!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies in '{env_name}': {e}")
    else:
        try:
            print(f"Installing dependencies for '{env_name}' from setup...")
            subprocess.run(
                f'bash -c "source activate base; conda activate {env_name}; pip install .; pip install coverage pytest"',
                shell=True,
                cwd=target_dir,
                stdout=subprocess.PIPE,
                check=True,
            )
            print(f"Dependencies installed successfully in '{env_name}'!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies in '{env_name}': {e}")
