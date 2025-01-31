import os
import json


def has_intersection_between_code_lines(section1, section2):
    """
    Check if two code sections have intersection.
    """
    return max(section1[0], section2[0]) <= min(section1[1], section2[1])


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


def load_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as file:
        return json.load(file)
