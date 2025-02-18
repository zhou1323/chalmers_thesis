import os
import sys
import logging
import pickle
import pandas as pd
from handlers.code_processing import evaluation
from handlers.testing_util_v2 import (
    functional_evaluation,
    split_test_cases,
)
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


class Evaluation:
    """
    Class to handle evaluation of model predictions.
    """

    def __init__(self):
        # Load configuration
        self.config = Config.evaluation
        self.test_dataset_path = self.config["test_dataset_path"]
        self.input_column = self.config["input_column"]
        self.output_column = self.config["output_column"]
        self.result_mode = self.config["result_mode"]
        self.pred_mode = self.config["pred_mode"]
        self.n_pred_rank = self.config["n_pred_rank"]
        self.n_test = self.config["n_test"]
        self.on_save = self.config["on_save"]
        self.output_dir = self.config["output_dir"]
        self.gen_testcase_path = self.config["gen_testcase_path"]

        # Initialize logging
        self._setup_logging()

        # Load predictions
        self.predictions = self._load_predictions()

        # Load test dataset
        self.test_df = self._load_test_dataset()

    def _setup_logging(self):
        """Set up logging configuration."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log_file = (
            self.output_dir
            + f"/prediction_from_file_rank{self.n_pred_rank}_{self.pred_mode}{self.n_test}.log"
        )
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.addHandler(logging.StreamHandler())

    def _load_predictions(self):
        """Load predictions from file."""
        if self.result_mode == "codex-codet":
            gen_testcase = pd.read_json(self.gen_testcase_path, lines=False)
            predictions = list(gen_testcase["samples"])
        elif self.result_mode == "codex-api":
            gen_testcase = pd.read_json(self.gen_testcase_path, lines=False)
            predictions = [
                [
                    gen_testcase["response"][i]["choices"][self.n_pred_rank]["message"][
                        "content"
                    ]
                ]
                for i in range(len(gen_testcase))
            ]
        elif self.result_mode == "baselines":
            gen_testcase = pd.read_json(self.gen_testcase_path, lines=False)
            predictions = list(gen_testcase["response"])
        else:
            predictions = pickle.load(open(self.gen_testcase_path, "rb"))
        logging.info("Loaded predictions successfully.")
        return predictions

    def _load_test_dataset(self):
        """Load the test dataset."""
        test_df = pd.read_csv(self.test_dataset_path)
        logging.info(f"# test data: {len(test_df)}")
        return test_df

    def evaluate_predictions(self):
        """Evaluate the predictions against the expected outputs."""
        test_inputs_text = list(self.test_df[self.input_column])
        test_outputs_text = list(self.test_df[self.output_column])
        fn_names = list(self.test_df["fn_name"])
        prompt_codes = list(self.test_df["prompt_code"])
        output_solutions = list(self.test_df["output_solution"])

        if self.pred_mode == "single":
            self.predictions = self._process_single_predictions()

        first_predictions = [pred[self.n_pred_rank] for pred in self.predictions]

        # Evaluate predictions
        logging.info(f"*None-filtered assertions syntax results :{self.pred_mode}*")
        em, es, mrr, parsable = evaluation(
            self.predictions, test_outputs_text, test_inputs_text, processed=False
        )
        logging.info(f"Exact Match: {em}%")
        logging.info(f"Edit Similarity: {es}%")
        logging.info(f"MRR: {mrr}%")
        logging.info("-----AST Parsable Rate-----")
        for k in parsable:
            logging.info(f"{k}: {parsable[k]}%")

        if self.result_mode != "codex-codet":
            filtered_predictions = self._filter_assertions_syntax(fn_names)
            if self.on_save:
                with open(
                    self.output_dir + f"/filtered_prediction-5{self.output_suffix}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(filtered_predictions, f)
            logging.info(f"*Filtered assertions syntax results :{self.pred_mode}*")
            em, es, mrr, parsable = evaluation(
                filtered_predictions,
                test_outputs_text,
                test_inputs_text,
                processed=False,
            )
            logging.info(f"Exact Match: {em}%")
            logging.info(f"Edit Similarity: {es}%")
            logging.info(f"MRR: {mrr}%")
            logging.info("-----AST Parsable Rate-----")
            for k in parsable:
                logging.info(f"{k}: {parsable[k]}%")

        # Functional evaluation
        self._functional_evaluation(
            prompt_codes, output_solutions, first_predictions, fn_names
        )

    def _process_single_predictions(self):
        """Process predictions for single mode."""
        new_predictions = []
        for preds in self.predictions:
            p = []
            for pred in preds:
                if "\nassert" in pred:
                    pred = "\nassert".join(pred.split("\nassert")[: self.n_test])
                p.append(pred)
            new_predictions.append(p)
        return new_predictions

    def _filter_assertions_syntax(self, fn_names):
        """Filter assertions syntax from predictions."""
        filtered_predictions = []
        for i in range(len(self.predictions)):
            temp = []
            for p in self.predictions[i]:
                checked_assertions, _ = split_test_cases(
                    p,
                    fn_names[i],
                    filter_syntax=True,
                    add_test_call_solution=(self.result_mode == "ours"),
                )
                temp.append(
                    "\n".join(checked_assertions)[7:].replace(
                        "test_call_solution(", "call_solution("
                    )
                )
            filtered_predictions.append(temp)
        return filtered_predictions

    def _functional_evaluation(
        self, prompt_codes, output_solutions, first_predictions, fn_names
    ):
        """Perform functional evaluation of predictions."""
        logging.info("-----Functional Rate-----")
        functional, coverage, mutation = functional_evaluation(
            prompts=prompt_codes,
            solutions=output_solutions,
            testcases=first_predictions,
            fn_names=fn_names,
            eval_coverage=True,
            eval_mutate=True,
            debug=False,
            on_guard=True,
            on_codet_result=(self.result_mode == "codex-codet"),
            add_test_call_solution=(self.result_mode == "ours"),
            filter_syntax=False,
        )
        if self.on_save:
            with open(
                self.output_dir + f"/evaluation_perfect{self.output_suffix}.pkl", "wb"
            ) as f:
                pickle.dump([functional, coverage, mutation], f)
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__
        logging.info(f"*None-filtered assertions syntax results :{self.pred_mode}*")
        for k in functional:
            logging.info(f"{k}: {functional[k]}%")
            print(f"{k}: {functional[k]}%")
        logging.info(f"coverage: {sum(coverage) / len(coverage)}%")
        print(f"coverage: {sum(coverage) / len(coverage)}%")
        logging.info(f"Mutation score: {sum(mutation) / len(mutation)}%")
        print(f"Mutation score: {sum(mutation) / len(mutation)}%")


if __name__ == "__main__":
    evaluator = Evaluation()
    evaluator.evaluate_predictions()
    logging.info("Finish the evaluation program.")
