import os
import sys
import json

DATASETS = ["BugsInPy", "Pybughive"]

PROMPT = "// New Requirement: {requirement} \n // Current Function code: {function_code} \n // Previous Testcase: {testcase} \n // Relevant Testcase: {relevant_testcase} \n"

QUESTION_SUFFIX = "<Problem>"
COMMIT_SUFFIX = "<Commit>"

NO_PREVIOUS_TESTCASE = "<None>"
NO_CURRENT_FUNCTION = "<None>"
NO_RELEVANT_TESTCASE = "<None>"


def load_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as file:
        return json.load(file)


def save_to_json(data, output_file):
    """
    Save data to a JSON file.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write/overwrite file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON file {output_file}: {e}")


def get_requirement(data):
    requirement = ""

    description_commit = data["description_commit"]
    description_question = data["description_question"]
    if description_question:
        requirement += QUESTION_SUFFIX + description_question

    if description_commit:
        requirement += COMMIT_SUFFIX + description_commit

    return requirement


def get_function_code_after(data):
    function_codes_after = data["function_codes_after"]

    if len(function_codes_after.values()) == 0:
        return NO_CURRENT_FUNCTION
    elif len(function_codes_after.values()) == 1:
        function_codes = list(function_codes_after.values())[0]
        return "\n".join([i["code"] for i in function_codes])
    else:
        return ""


def get_testcases(data):
    result = {}
    if len(data.values()) == 0:
        return result
    else:
        for test_file, testcases in data.items():
            single_lines = []
            class_dict = {}
            for testcase in testcases:
                if testcase["type"] == "single_line":
                    single_lines.append(testcase["code"])
                    continue
                class_name = (
                    testcase["full_name"].split(".")[-2]
                    if len(testcase["full_name"].split(".")) > 1
                    else ""
                )
                class_dict[class_name] = class_dict.get(class_name, []) + [
                    testcase["code"]
                ]

            for class_name, codes in class_dict.items():
                final_codes = single_lines + codes
                result[f"{test_file}|{class_name}"] = "\n".join(final_codes)
        return result


def get_relevant_testcase(data):
    relevant_test_cases = data["relevant_test_cases"]
    if len(relevant_test_cases.values()) == 0:
        return NO_RELEVANT_TESTCASE
    else:
        testcases = list(relevant_test_cases.values())[0]
        return "\n".join([i["code"] for i in testcases])


def process_dataset(dataset_dir, start_id, language="python"):
    result = []
    # Read through files in the dataset category
    for file in os.listdir(dataset_dir):
        data = load_json(dataset_dir, file)
        for one in data:
            # Info to be used for the prompt
            requirement = get_requirement(one)
            function_code_after = get_function_code_after(one)

            testcases_before = get_testcases(one["test_cases_before"])
            # Get the output testcase
            testcases_after = get_testcases(one["test_cases_after"])

            all_test_file_classes = list(
                set(list(testcases_before.keys()) + list(testcases_after.keys()))
            )

            # Don't have requirement or function code is changed in multiple files
            if not requirement or not function_code_after:
                continue

            for test_file_class in all_test_file_classes:
                test_file, test_class = test_file_class.split("|")

                if (
                    test_file_class in testcases_before
                    and test_file_class in testcases_after
                ):
                    testcase_before = testcases_before[test_file_class]
                    testcase_after = testcases_after[test_file_class]
                elif test_file_class in testcases_before:
                    testcase_before = testcases_before[test_file_class]
                    testcase_after = NO_PREVIOUS_TESTCASE
                elif test_file_class in testcases_after:
                    testcase_before = NO_PREVIOUS_TESTCASE
                    testcase_after = testcases_after[test_file_class]
                else:
                    continue

                # Get the relavant testcase
                # relevant_testcase = (
                #     get_relevant_testcase(one, test_file, test_class)
                #     if testcase_before == NO_PREVIOUS_TESTCASE
                #     else NO_RELEVANT_TESTCASE
                # )
                relevant_testcase = NO_RELEVANT_TESTCASE  # todo: for now

                prompt = PROMPT.format(
                    requirement=requirement,
                    function_code=function_code_after,
                    testcase=testcase_before,
                    relevant_testcase=relevant_testcase,
                )

                result.append(
                    {
                        "id": len(result) + start_id,
                        "source": os.path.join(dataset_dir, file),
                        "test_file": test_file,
                        "test_class": test_class,
                        "language": language,
                        "fn_name": "call_function",
                        "prompt_code": prompt,  # To write the code? # TODO: Check this
                        "prompt_testcase": prompt,  # To generate the test case
                        "output_testcase": testcase_after,
                        "output_solution": function_code_after,  # Function codes
                    }
                )
    return result


if __name__ == "__main__":
    result = []
    # Go through each dataset and preprocess it
    for dataset in DATASETS:
        if dataset == "defects4j":
            language = "java"
        else:
            language = "python"
        dataset_dir = os.path.join("data", "processed_" + dataset)
        result.extend(process_dataset(dataset_dir, len(result), language))

    # Save the result to a file
    save_to_json(result, "data/train_data.json")

    # Randomly shuffle the data and split into train and test
    import random

    random.shuffle(result, random.seed(42))

    java_data = [i for i in result if i["language"] == "java"]
    python_data = [i for i in result if i["language"] == "python"]

    train_data = (
        java_data[: int(0.8 * len(java_data))]
        + python_data[: int(0.8 * len(python_data))]
    )
    test_data = (
        java_data[int(0.8 * len(java_data)) :]
        + python_data[int(0.8 * len(python_data)) :]
    )

    # Save the result to a csv file
    import pandas as pd

    # Save the result to a csv file
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv("dataset/APPS_new/train_data.csv", index=False)
    test_df.to_csv("dataset/APPS_new/test_data.csv", index=False)
