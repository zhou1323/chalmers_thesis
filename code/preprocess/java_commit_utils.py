import subprocess
import javalang
import re
import os
import requests
from bs4 import BeautifulSoup
from utils import has_intersection_between_code_lines


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


def apply_and_extract_with_commit(repo_path, changes, bug, to_get_bugs=False):
    """
    Apply the commit and extract the code structure.

    Args:
        repo_path: Path to git repository
        bug: Bug details {bug_id, revision_buggy, revision_fixed, url}
        to_get_bugs: Flag to get buggy or fixed commit
    """
    commit_hash = bug["revision_buggy"] if to_get_bugs else bug["revision_fixed"]

    extracted_data = {}

    try:
        subprocess.run(
            (["git", "checkout", commit_hash]),
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error applying commit: {e}")
    except Exception as e:
        print(f"Common Error applying commit: {e}")

    extracted_data = extract_functions_and_classes_by_patch(
        repo_path, changes, to_get_old_lines=to_get_bugs
    )

    function_codes, test_cases = {}, {}

    for file_path, data in extracted_data.items():
        if "test/" in file_path:
            test_cases[file_path] = data
        else:
            function_codes[file_path] = data

    return function_codes, test_cases


def get_commit_log(repo_path: str, commit_hash: str) -> str:
    """
    Get commit log message for a specific commit.

    Args:
        repo_path: Path to git repository
        commit_hash: Commit hash or revision number

    Returns:
        str: Commit log message or empty string if error
    """
    try:
        # Get commit log with pretty format
        result = subprocess.run(
            ["git", "log", "--format=%B", "-n", "1", commit_hash],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Try different encodings
        for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                return result.stdout.decode(encoding).strip()
            except UnicodeDecodeError:
                continue

        print(f"Failed to decode git log with any encoding for commit {commit_hash}")
        return ""

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit log: {e}")
        return ""


def get_commit_changes(repo_path: str, commit_hash: str) -> dict:
    """
    Get changed files and their modifications for a specific commit.

    Args:
        repo_path: Path to git repository
        commit_hash: Commit hash or revision number

    Returns:
        dict: Changed files and their modifications
    """
    try:
        # Get commit details
        result = subprocess.run(
            ["git", "show", commit_hash],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                return result.stdout.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return {}

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit changes: {e}")
        return {}


def extract_functions_and_classes_by_patch(repo_path, changes, to_get_old_lines=False):
    """
    Extract the complete code structure of the changed files.
    """
    extracted_data = {}
    for change in changes:
        file_path = os.path.join(repo_path, change["file"])
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                java_code = f.read()
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

                result = extract_code_structure(
                    java_code, code_start_line, code_end_line
                )
                if result:
                    extracted_data[file_path] = (
                        [result]
                        if file_path not in extracted_data
                        else extracted_data[file_path] + [result]
                    )
    return extracted_data


def extract_code_structure(java_code, start_line, end_line):
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
                }
    return None


def _extract_single_line(lines, start_line, end_line):
    """Extract the single line"""
    if has_intersection_between_code_lines([1, len(lines)], [start_line, end_line]):
        return {
            "type": "single_line",
            "code": "".join(lines[start_line - 1 : end_line]),
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


def parse_patch_content(patch_content):
    changes = []
    current_file = None
    for line in patch_content:
        if line.startswith("+++") or line.startswith("---"):
            # It starts with +++ or ---! So it is a file name with .java extension
            match = re.match(r"[+-]{3}\s+[ab]/(.*\.java)", line)
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


def parse_patch(patch, is_file_name=True):
    """
    Analyze the patch file and extract the changes.
    """
    if is_file_name:
        with open(patch, "r") as f:
            return parse_patch_content(f)
    else:
        return parse_patch_content(patch.splitlines())
