import ast
import os
import torch
import logging
import pickle
import datasets
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    PLBartForCausalLM,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


class ModelInference:
    """
    Class to handle model inference.
    """

    def __init__(self):
        # Load configuration
        self.config = Config.inference
        self.device = Config.device
        self.model_name = self.config["model_name"]
        self.model_dir = self.config["model_dir"]
        self.test_dataset_path = self.config["test_dataset_path"]
        self.input_column = self.config["input_column"]
        self.output_column = self.config["output_column"]
        self.pred_mode = self.config["pred_mode"]
        self.output_dir = self.config["output_dir"]
        self.output_suffix = self.config["output_suffix"]
        self.batch_size = self.config["batch_size"]
        self.beam_size = self.config["beam_size"]
        self.max_new_tokens = self.config["max_new_tokens"]
        self.max_length = self.config["max_length"]

        # Initialize logging
        self._setup_logging()

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        # Load test dataset
        self.test_loader = self._load_test_dataset()

    def _setup_logging(self):
        """Set up logging configuration."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log_file = self.output_dir + "/prediction.log"
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.addHandler(logging.StreamHandler())
        logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
        logger.info(
            f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
        logger.info(
            f"""evaluated_model_dir: {self.model_dir},
                    tokenizer_dir: {self.model_dir},
                    test_data_dir: {self.test_dataset_path},
                    input_column: {self.input_column},
                    output_column: {self.output_column},
                    output_dir: {self.output_dir},
                    output_suffix: {self.output_suffix},
                    -- Generator params --
                    batch_size: {self.batch_size},
                    beam_size: {self.beam_size},
                    max_new_tokens: {self.max_new_tokens},
                    max_length: {self.max_length},
                    pred_mode: {self.pred_mode}
                    """
        )

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer based on the configuration."""
        base_model = {
            "pycoder": AutoModelForCausalLM,
            "codegpt": AutoModelForCausalLM,
            "transformers": AutoModelForCausalLM,
            "gpt2": AutoModelForCausalLM,
            "codet5": AutoModelForSeq2SeqLM,
            "plbart": PLBartForCausalLM,
        }
        model = (
            base_model[self.model_name].from_pretrained(self.model_dir).to(self.device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            truncation_side="left",
            padding_side=(
                "left"
                if self.model_name
                in ["codegpt", "pycoder", "gpt2", "transformers", "plbart"]
                else "right"
            ),
            do_lower_case=False,
        )
        if self.model_name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("Loaded model and tokenizer successfully.")
        return model, tokenizer

    def _load_test_dataset(self):
        """Load and prepare the test dataset."""
        test_df = pd.read_csv(self.test_dataset_path)
        # Find all data whose 'language' column is not 'java'
        # todo: for testing!
        # test_df = test_df[test_df["language"] != "java"].head(5)
        logging.info(f"# test data: {len(test_df)}")
        test_inputs = self.tokenizer(
            list(test_df[self.input_column]),
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        test_dataset = datasets.Dataset.from_dict(
            {
                "input_ids": test_inputs["input_ids"],
                "attention_mask": test_inputs["attention_mask"],
            }
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=(
                DataCollatorForSeq2Seq(self.tokenizer)
                if "codet5" in self.model_name
                else DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            ),
        )
        logging.info("Loaded dataset successfully.")
        return test_loader

    def generate_predictions(self):
        """Generate predictions on the test dataset."""
        generation_kwargs = {
            "min_new_tokens": 5,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": self.beam_size,
            "num_beams": self.beam_size,
        }
        logging.info(f"generation_kwargs: {generation_kwargs}")
        predictions = []
        for batch in tqdm(self.test_loader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                predictions.extend(decoded_outputs)
        return predictions

    def save_predictions(self, predictions):
        """Save predictions to file."""
        # first_predictions = []
        # for pred in predictions:
        #     if len(pred) > 0:
        #         first_predictions.append(pred[0])
        #     else:
        #         continue
        with open(
            self.output_dir + f"/prediction-{self.beam_size}{self.output_suffix}.pkl",
            "wb",
        ) as f:
            pickle.dump(predictions, f)
        with open(
            self.output_dir + f"/predictions-{self.beam_size}{self.output_suffix}.txt",
            "w",
        ) as f:
            for data in predictions:
                f.write(data + "\n")


if __name__ == "__main__":
    inference = ModelInference()
    predictions = inference.generate_predictions()
    inference.save_predictions(predictions)
    logging.info("Finish the inference program.")
