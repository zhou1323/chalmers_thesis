import subprocess
import javalang
import re
import os
import parso
import requests

from bs4 import BeautifulSoup
from utils import has_intersection_between_code_lines
from coverage_utils import load_line2test_json, extract_coverage_report
from git import Repo


def get_jira_description(url: str) -> str:
    """
    Extract description from JIRA issue page.

    Args:
        url: JIRA issue URL

    Returns:
        str: Description text or empty string if error
    """
    try:
        # Send GET request
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find description element
        desc_div = soup.find("div", {"id": "description-val"})
        if desc_div:
            return desc_div.get_text().strip()

        return ""

    except Exception as e:
        print(f"Error fetching description from {url}: {e}")
        return ""


def apply_and_extract_with_commit(
    repo, changes, commit_hash, to_get_bugs=False, language="java"
):
    """
    Apply the commit and extract the code structure.

    Args:
        repo: GitPython repository object
        changes: List of changed files and their modifications
        commit_hash: Commit hash or revision number
        to_get_bugs: Flag to get buggy or fixed commit
    """

    extracted_data = {}

    if language == "java":
        test_path = "test/"
    elif language == "python":
        test_path = "tests/"
    else:
        test_path = "test/"

    try:
        # Apply the commit
        repo.git.checkout(commit_hash, force=True)
    except subprocess.CalledProcessError as e:
        print(f"Error applying commit: {e}")
    except Exception as e:
        print(f"Common Error applying commit: {e}")

    extracted_data = extract_functions_and_classes_by_patch(
        repo.working_dir, changes, to_get_old_lines=to_get_bugs, language=language
    )

    function_codes, test_cases = {}, {}

    for file_path, data in extracted_data.items():
        if test_path in file_path:
            test_cases[file_path] = data
        else:
            function_codes[file_path] = data

    return function_codes, test_cases


def get_commit_changes(repo: Repo, commit_hash: str) -> list:
    """
    Get changed files and their modifications for a specific commit.

    Args:
        repo: GitPython repository object
        commit_hash: Commit hash or revision number

    Returns:
        list: List of changed files and their modifications
    """
    try:
        # Get commit details
        result = repo.git.show(commit_hash)

        return result

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit changes: {e}")
        return []


def get_commit_log(repo: Repo, commit_hash: str):
    """
    Get commit log message for a specific commit.

    Args:
        repo: GitPython repository object
        commit_hash: Commit hash or revision number

    Returns:
        str: Commit log message or empty string if error
    """
    try:
        commit = repo.commit(commit_hash)
        return commit.message

    except Exception as e:
        print(f"Error getting commit log: {e}")
        return ""


def extract_functions_and_classes_by_patch(
    repo_path, changes, to_get_old_lines=False, language="java"
):
    """
    Extract the complete code structure of the changed files.
    """
    extracted_data = {}
    for change in changes:
        file_path = os.path.join(repo_path, change["file"])
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                code = f.read()
                if to_get_old_lines:
                    code_start_line, code_end_line = (
                        change["old_start"],
                        change["old_end"],
                    )
                else:
                    code_start_line, code_end_line = (
                        change["new_start"],
                        change["new_end"],
                    )

                if language == "java":
                    result = extract_code_structure_java(
                        code, code_start_line, code_end_line
                    )
                elif language == "python":
                    result = extract_code_structure_python(
                        code, code_start_line, code_end_line
                    )
                else:
                    result = None

                if result:
                    extracted_data[file_path] = (
                        [result]
                        if file_path not in extracted_data
                        else extracted_data[file_path] + [result]
                    )
    return extracted_data


def extract_test_methods_from_file(file_path, filter_methods=[]):
    """
    Find all methods with @Test annotation in the given Java source code.

    Returns:
        dict: {method_name: {path: str, code: str}}
    """
    with open(file_path, "r") as f:
        source_code = f.read()

    tokens = list(javalang.tokenizer.tokenize(source_code))
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse()

    lines = source_code.split("\n")

    test_methods_dict = {}

    for type_decl in tree.types:
        if isinstance(type_decl, javalang.tree.ClassDeclaration):
            for method in type_decl.methods:
                has_test_annotation = False
                if method.annotations:
                    for ann in method.annotations:
                        if ann.name.endswith("Test"):
                            has_test_annotation = True
                            break

                if has_test_annotation and method.name not in filter_methods:
                    method_source = _extract_method_source(
                        lines, start_line=method.position.line
                    )
                    if method_source:
                        test_methods_dict[method.name] = {
                            "path": file_path,
                            "code": method_source,
                        }

    return test_methods_dict


def _extract_method_source(lines, start_line):
    if not start_line or start_line <= 0:
        return None

    brace_count = 0
    found_open_brace = False
    method_lines = []

    current_line_index = start_line - 1

    while current_line_index < len(lines):
        line = lines[current_line_index]
        method_lines.append(line)

        if not found_open_brace:
            if "{" in line:
                found_open_brace = True
                brace_count += line.count("{")
                brace_count -= line.count("}")
        else:
            brace_count += line.count("{")
            brace_count -= line.count("}")

        current_line_index += 1

        if found_open_brace and brace_count == 0:
            break

    return "\n".join(method_lines)


def extract_code_structure_java(java_code, start_line, end_line):
    """Extract code structure using a unified approach"""
    try:
        tree = javalang.parse.parse(java_code)
        lines = java_code.splitlines(keepends=True)

        # Map of node types to their structure types
        structure_types = {
            javalang.tree.MethodDeclaration: "method",
            javalang.tree.ConstructorDeclaration: "constructor",
        }

        # Try to extract each type of structure
        for node_type, structure_type in structure_types.items():
            result = _extract_structure(
                tree, lines, start_line, end_line, node_type, structure_type
            )
            if result:
                return result

        # Special handling for class declarations
        if _is_class_change(start_line, end_line, tree, lines):
            return _extract_class_fields(tree, lines, start_line, end_line)

        # Fallback to single line extraction
        return _extract_single_line(lines, start_line, end_line)

    except Exception as e:
        print(f"Error parsing Java code: {e}")
        return None


def _extract_class_fields(tree, lines, start_line, end_line):
    """Extract only field declarations from class"""
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if hasattr(node, "position") and node.position:
            class_start = node.position.line

            # Find first method/constructor position
            first_method_line = float("inf")
            for member in node.body:
                if isinstance(
                    member,
                    (
                        javalang.tree.MethodDeclaration,
                        javalang.tree.ConstructorDeclaration,
                    ),
                ):
                    if hasattr(member, "position") and member.position:
                        first_method_line = min(first_method_line, member.position.line)

            if first_method_line == float("inf"):
                first_method_line = _find_structure_end(lines, class_start)

            if has_intersection_between_code_lines(
                [class_start, first_method_line], [start_line, end_line]
            ):
                return {
                    "type": "class",
                    "code": "".join(lines[class_start - 1 : first_method_line - 1]),
                }
    return None


def _is_class_change(start_line, end_line, tree, lines):
    """Check if the change is within a class declaration"""
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if hasattr(node, "position") and node.position:
            class_start = node.position.line
            class_end = _find_structure_end(lines, class_start)
            if has_intersection_between_code_lines(
                [class_start, class_end], [start_line, end_line]
            ):
                return True
    return False


def _extract_structure(tree, lines, start_line, end_line, node_type, structure_type):
    """Generic structure extractor for methods, constructors and classes"""
    for _, node in tree.filter(node_type):
        if hasattr(node, "position") and node.position:
            struct_start = node.position.line
            struct_end = _find_structure_end(lines, struct_start)

            if has_intersection_between_code_lines(
                [struct_start, struct_end], [start_line, end_line]
            ):
                return {
                    "type": structure_type,
                    "code": "".join(lines[struct_start - 1 : struct_end]),
                    "name": node.name,
                    "start_line": struct_start,
                    "end_line": struct_end,
                }
    return None


def _extract_single_line(lines, start_line, end_line):
    """Extract the single line"""
    if has_intersection_between_code_lines([1, len(lines)], [start_line, end_line]):
        return {
            "type": "single_line",
            "code": "".join(lines[start_line - 1 : end_line]),
            "start_line": start_line,
            "end_line": end_line,
        }
    return None


def _find_structure_end(lines, start_line):
    """Find the end of the class"""
    brace_count = 0
    found_first_brace = False

    for i, line in enumerate(lines[start_line - 1 :], start=start_line):
        if "{" in line:
            found_first_brace = True
            brace_count += line.count("{")
        if "}" in line:
            brace_count -= line.count("}")
        if found_first_brace and brace_count == 0:
            return i
    return len(lines)


def parse_patch_content(patch_content, language="java"):
    changes = []
    current_file = None
    if language == "java":
        regex = r"[+-]{3}\s+[ab]/(.*\.java)"
    elif language == "python":
        regex = r"[+-]{3}\s+[ab]/(.*\.py)"
    for line in patch_content:
        if line.startswith("+++") or line.startswith("---"):
            # It starts with +++ or ---! So it is a file name with proper suffix extension
            match = re.match(regex, line)
            if match:
                current_file = match.group(1)
        elif line.startswith("@@"):
            match = re.match(r"@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@", line)
            if match:
                # Could match some non-java files
                if current_file is None:
                    continue
                old_start = int(match.group(1))
                old_lines = int(match.group(2) or 1)
                new_start = int(match.group(3))
                new_lines = int(match.group(4) or 1)
                changes.append(
                    {
                        "file": current_file,
                        "old_start": old_start,
                        "old_end": old_start + old_lines - 1,
                        "new_start": new_start,
                        "new_end": new_start + new_lines - 1,
                    }
                )
    return changes


def parse_patch(patch, is_file_name=True, language="java"):
    """
    Analyze the patch file and extract the changes.
    """
    if is_file_name:
        with open(patch, "r") as f:
            return parse_patch_content(f)
    else:
        return parse_patch_content(patch.splitlines(), language=language)


import parso


def extract_code_structure_python(python_code, start_line, end_line):
    """Extract code structure for Python code using Parso."""
    try:
        # Parse the Python code using Parso
        tree = parso.parse(python_code)
        lines = python_code.splitlines(keepends=True)

        # Map node types to structure types
        structure_types = {
            "funcdef": "function",
            "async_funcdef": "function",  # Async functions are also functions
            "classdef": "class",
        }

        # Try to extract each type of structure
        for node_type, structure_type in structure_types.items():
            result = _extract_structure_python(
                tree, lines, start_line, end_line, node_type, structure_type
            )
            if result:
                return result

        # Fallback to single line extraction
        return _extract_single_line(lines, start_line, end_line)

    except Exception as e:
        print(f"Error parsing Python code: {e}")
        return None


def _extract_structure_python(
    node, lines, start_line, end_line, node_type, structure_type
):
    """Generic structure extractor for functions and classes."""
    if node.type == node_type:
        # Determine the start and end lines of the structure
        struct_start = node.start_pos[0]
        struct_end = node.end_pos[0]

        if has_intersection_between_code_lines(
            [struct_start, struct_end], [start_line, end_line]
        ):
            return {
                "type": structure_type,
                "name": node.name.value,
                "code": "".join(lines[struct_start - 1 : struct_end]),
                "start_line": struct_start,
                "end_line": struct_end,
            }

    # Recursively check child nodes
    if hasattr(node, "children"):
        for child in node.children:
            result = _extract_structure_python(
                child, lines, start_line, end_line, node_type, structure_type
            )
            if result:
                return result

    return None


def find_test_files(directory):
    """Find all test files in the given directory"""
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files


def find_test_cases_by_python_code_lines(
    test_case_directory: str,
    funciton_code_lines: dict,  # { file_path: [(start_line, end_line), ...] }
    filtered_cases: list = None,
) -> dict:
    """
    Find test cases that call the target methods in the given directory.

    Args:
        directory: the directory to search for test files

        funciton_code_lines: the code lines of the target methods
                             { file_path: [(start_line, end_line), ...] }

        filtered_cases: test cases to skip
                        [test_case_name, ...]

    Returns:
        dict: test cases that call the target methods
              { test_case_name: { "file_path": xxx, "code": xxx}, ...}
    """
    if filtered_cases is None:
        filtered_cases = []

    coverate_file_dir = test_case_directory.split("/tests/")[0]
    extract_coverage_report(coverate_file_dir, coverate_file_dir, redo_extraction=True)
    line_to_test = load_line2test_json(coverate_file_dir)

    test_files = find_test_files(test_case_directory)
    test_function_map = {}
    for file_path in test_files:
        with open(file_path, "r", encoding="utf-8") as f:
            code_str = f.read()
        funcs_info = extract_functions_with_parso(code_str, file_path)

        test_function_map.update(funcs_info)

    results = {}

    for src_file_path, line_ranges in funciton_code_lines.items():

        if src_file_path not in line_to_test:
            continue

        for start_line, end_line in line_ranges:
            for line_no in range(start_line, end_line + 1):
                line_str = str(line_no)
                if line_str in line_to_test[src_file_path]:
                    test_func_names = line_to_test[src_file_path][line_str]
                    for func_name in test_func_names:
                        if (
                            func_name not in filtered_cases
                            and func_name in test_function_map
                        ):
                            if func_name not in results:
                                results[func_name] = test_function_map[func_name]

    return results


def extract_functions_with_parso(code_str: str, file_path: str) -> dict:
    """
    Use Parso to extract functions from Python code.

    Returns:
        {
          "test_func_name": {
              "file_path": str,
              "code": str
          },
          "ClassName.test_method_name": {
              "file_path": str,
              "code": str
          },
          ...
        }
    """
    results = {}

    try:
        module = parso.parse(code_str)
    except Exception as e:
        print(f"Parse error in {file_path}: {e}")
        return results

    def _dfs(node, class_path=None):
        """
        DFS to traverse the parso nodes and collect all functions with the test_ prefix.
        When encountering a class definition, recursively traverse its children nodes.
        """
        if node.type == "funcdef":
            func_name_leaf = node.children[1]  # 'def' [whitespace] 'func_name' ...
            if func_name_leaf.type == "name":
                func_name = func_name_leaf.value
                if func_name.startswith("test_"):
                    full_name = (
                        func_name if not class_path else f"{class_path}.{func_name}"
                    )
                    func_code = node.get_code()
                    results[full_name] = {"file_path": file_path, "code": func_code}
        elif node.type == "classdef":
            class_name_leaf = node.children[1]  # 'class' [whitespace] 'ClassName' ...
            if class_name_leaf.type == "name":
                cls_name = class_name_leaf.value
                new_class_path = (
                    cls_name if not class_path else f"{class_path}.{cls_name}"
                )
                # Iterate over the class body to find test methods
                for sub_node in node.children:
                    _dfs(sub_node, class_path=new_class_path)

        # If the node has children, recursively traverse them
        if hasattr(node, "children"):
            for child in node.children:
                _dfs(child, class_path=class_path)

    # Run the DFS starting from the module
    _dfs(module)
    return results
