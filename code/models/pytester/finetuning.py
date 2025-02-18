import pandas as pd
import numpy as np
import os
import logging
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2Model,
    RobertaModel,
    PLBartForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)
import sys
import re
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config


class FineTuning:
    """
    Class to handle model fine-tuning.
    """

    def __init__(self):
        # Load configuration
        self.config = Config.finetuning
        self.model_name = self.config["model_name"]
        self.is_shuffle = self.config["is_shuffle"]
        self.train_dataset_path = self.config["train_dataset_path"]
        self.input_column = self.config["input_column"]
        self.output_column = self.config["output_column"]
        self.output_dir = self.config["output_dir"]

        # Initialize logging
        self._setup_logging()

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        # Load datasets
        self.train_dataset, self.eval_dataset = self._load_datasets()

        # Initialize Trainer
        self.trainer = self._initialize_trainer()

    def _setup_logging(self):
        """Set up logging configuration."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log_file = self.output_dir + "/train.log"
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

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer based on the configuration."""
        base_model = {
            "pycoder": AutoModelForCausalLM,
            "codegpt": AutoModelForCausalLM,
            "transformers": GPT2Model,
            "gpt2": AutoModelForCausalLM,
            "codet5-small": AutoModelForSeq2SeqLM,
            "codet5-base": AutoModelForSeq2SeqLM,
            "codet5-large": AutoModelForSeq2SeqLM,
            "plbart": PLBartForCausalLM,
        }
        base_checkpoint = {
            "pycoder": "Wannita/PyCoder",
            "codegpt": "microsoft/CodeGPT-small-py",
            "transformers": "gpt2_no_pretrain_weight",
            "gpt2": "gpt2",
            "codet5-small": "Salesforce/codet5-small",
            "codet5-base": "Salesforce/codet5-base",
            "codet5-large": "Salesforce/codet5-large",
            "unixcoder": "microsoft/unixcoder-base",
            "plbart": "uclanlp/plbart-base",
        }
        if self.model_name == "transformers":
            config = GPT2Config()
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", truncation_side="left", do_lower_case=False
            )
            tokenizer.pad_token = tokenizer.eos_token
            model = base_model[self.model_name](config)
        elif self.model_name == "unixcoder":
            config = AutoConfig.from_pretrained(base_checkpoint[self.model_name])
            config.is_decoder = True
            tokenizer = AutoTokenizer.from_pretrained(
                base_checkpoint[self.model_name],
                truncation_side="left",
                do_lower_case=False,
            )
            encoder = RobertaModel.from_pretrained(
                base_checkpoint[self.model_name], config=config
            )
            model = Seq2Seq(
                encoder=encoder,
                decoder=encoder,
                config=config,
                beam_size=5,
                max_length=512,
                sos_id=tokenizer.cls_token_id,
                eos_id=[tokenizer.sep_token_id],
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_checkpoint[self.model_name],
                truncation_side="left",
                do_lower_case=False,
            )
            if self.model_name == "gpt2":
                tokenizer.pad_token = tokenizer.eos_token
            if self.model_name == "plbart":
                model = base_model[self.model_name].from_pretrained(
                    base_checkpoint[self.model_name], add_cross_attention=False
                )
                assert (
                    model.config.is_decoder
                ), f"{model.__class__} has to be configured as a decoder."
            else:
                model = base_model[self.model_name].from_pretrained(
                    base_checkpoint[self.model_name]
                )
        logging.info("Loaded model and tokenizer successfully.")
        return model, tokenizer

    def _load_datasets(self):
        """Load and prepare the training and evaluation datasets."""
        train_df = pd.read_csv(self.train_dataset_path)
        train_df, eval_df = train_test_split(
            train_df, test_size=0.1, random_state=42, shuffle=self.is_shuffle
        )
        logging.info(f"# train data: {len(train_df)}")
        logging.info(f"# eval data: {len(eval_df)}")

        train_inputs = self.tokenizer(
            list(train_df[self.input_column]),
            padding=True,
            truncation=True,
            max_length=512,
        )
        train_outputs = self.tokenizer(
            list(train_df[self.output_column]),
            padding=True,
            truncation=True,
            max_length=512,
        )
        train_dataset = MyDataset(train_inputs, train_outputs)

        eval_inputs = self.tokenizer(
            list(eval_df[self.input_column]),
            padding=True,
            truncation=True,
            max_length=512,
        )
        eval_outputs = self.tokenizer(
            list(eval_df[self.output_column]),
            padding=True,
            truncation=True,
            max_length=512,
        )
        eval_dataset = MyDataset(eval_inputs, eval_outputs)
        logging.info("Loaded datasets successfully.")
        return train_dataset, eval_dataset

    def _initialize_trainer(self):
        """Initialize the Trainer for fine-tuning."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=20,
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="epoch",
            eval_accumulation_steps=1,
            evaluation_strategy="epoch",
            warmup_steps=1000,
            fp16=False,
            save_total_limit=10,
            optim="adamw_torch",
            lr_scheduler_type="inverse_sqrt",
        )
        logging.info(
            f"""training_args: lr={training_args.learning_rate}, 
                batch_size={training_args.per_device_train_batch_size}, 
                epoch={training_args.num_train_epochs}, 
                gradient_accumulation_steps={training_args.gradient_accumulation_steps}, 
                warmup_steps={training_args.warmup_steps}, 
                weight_decay={training_args.weight_decay},
                optim={training_args.optim},
                lr_scheduler_type={training_args.lr_scheduler_type},
                fp16={training_args.fp16}
                """
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=(
                DataCollatorForSeq2Seq(self.tokenizer)
                if "codet5" in self.model_name
                else DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            ),
            tokenizer=self.tokenizer,
        )
        return trainer

    def train(self):
        """Train the model using the Trainer."""
        self.trainer.train()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])


if __name__ == "__main__":
    finetuning = FineTuning()
    finetuning.train()
    logging.info("Finish the fine-tuning program.")
