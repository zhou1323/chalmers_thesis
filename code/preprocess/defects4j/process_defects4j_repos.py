import os
import subprocess
import pandas as pd

# Define constants
CSV_FILE = "data/defects4j/project_repos/repos.csv"
REPO_DIR = "data/defects4j/project_repos"


def process_repositories():
    """Process Defects4J repositories from CSV file"""
    # Check if CSV file exists
    if not os.path.isfile(CSV_FILE):
        print(f"CSV file doesn't exist: {CSV_FILE}")
        return

    try:
        # Read CSV file using pandas
        df = pd.read_csv(CSV_FILE, encoding="utf-8")

        # Process each repository
        for _, row in df.iterrows():
            project_name = row[0].strip()
            repo_url = row[1].strip()

            # Extract bare repository name
            repo_name = os.path.basename(repo_url).replace(".git", "")

            # Set paths
            bare_repo_path = os.path.join(REPO_DIR, f"{repo_name}.git")
            new_dir = os.path.join(REPO_DIR, f"repo_{project_name}")

            # Check if bare repository exists
            if os.path.isdir(bare_repo_path):
                print(f"Processing repository: {bare_repo_path}")

                # Create new directory
                os.makedirs(new_dir, exist_ok=True)

                try:
                    # Checkout bare repository to new directory
                    subprocess.run(
                        ["git", "clone", bare_repo_path, new_dir],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(
                        f"Bare repository {repo_name} added to new directory {new_dir}"
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error checking out repository {bare_repo_path}: {e}")
            else:
                print(f"Bare repository not found: {bare_repo_path}, skipping")

    except pd.errors.EmptyDataError:
        print("CSV file is empty")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


if __name__ == "__main__":
    process_repositories()
