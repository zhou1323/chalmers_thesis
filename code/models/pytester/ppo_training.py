# 0. imports
import os
import torch
import logging
import datasets
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    set_seed,
)
from handlers.code_processing import *
from handlers.testing_util_v2 import *
from handlers.python_terminal_command import PythonTerminalCommand
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inputs,
        outputs,
        prompt_text,
        fn_names,
        prompt_code,
        output_solution,
    ):
        self.inputs = inputs
        self.fn_names = fn_names
        self.outputs = outputs
        self.prompt_text = prompt_text
        self.prompt_code = prompt_code
        self.output_solution = output_solution

    def __getitem__(self, idx):
        return {
            "query": self.prompt_text[idx],
            "fn_name": self.fn_names[idx],
            "prompt_code": self.prompt_code[idx],
            "output_solution": self.output_solution[idx],
            "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])


class PPOTraining:
    """
    Class to handle PPO training.
    """

    def __init__(self):
        # Load configuration
        self.config = Config.ppo_training
        self.device = Config.device
        self.model_name = self.config["model_name"]
        self.commander = PythonTerminalCommand(
            self.config["unique_name"], random_unique_name=True
        )
        self.model_dir = self.config["model_dir"]
        self.train_dataset_path = self.config["train_dataset_path"]
        self.eval_dataset_path = self.config["eval_dataset_path"]
        self.input_column = self.config["input_column"]
        self.output_column = self.config["output_column"]
        self.output_dir = self.config["output_dir"]
        self.beam_size = self.config["beam_size"]
        self.do_eval = self.config["do_eval"]

        # Initialize logging
        self._setup_logging()

        # Load model and tokenizer
        self.model, self.tokenizer, self.model_ref = self._load_model_and_tokenizer()

        # Load datasets
        self.train_dataset, self.eval_loader, self.train_df, self.eval_df = (
            self._load_datasets()
        )

        # Initialize PPO trainer
        self.ppo_trainer = self._initialize_trainer()

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
            "transformers": AutoModelForCausalLMWithValueHead,
            "gpt2": AutoModelForCausalLMWithValueHead,
            "pycoder": AutoModelForCausalLMWithValueHead,
            "codegpt": AutoModelForCausalLMWithValueHead,
            "codet5": AutoModelForSeq2SeqLMWithValueHead,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, truncation_side="left", do_lower_case=False
        )
        if self.model_name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            base_model[self.model_name].from_pretrained(self.model_dir).to(self.device)
        )
        model_ref = (
            base_model[self.model_name].from_pretrained(self.model_dir).to(self.device)
        )
        logging.info("Loaded model and tokenizer successfully.")
        return model, tokenizer, model_ref

    def _load_datasets(self):
        """Load and prepare the training and evaluation datasets."""
        train_df = pd.read_csv(self.train_dataset_path)
        train_df = train_df[train_df["language"] != "java"].head(10)
        eval_df = pd.read_csv(self.eval_dataset_path)
        eval_df = eval_df[eval_df["language"] != "java"].head(10)  # todo: for testing
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
        train_dataset = MyDataset(
            inputs=train_inputs,
            outputs=train_outputs,
            prompt_text=list(train_df[self.input_column]),
            fn_names=list(train_df["fn_name"]),
            prompt_code=list(train_df["prompt_code"]),
            output_solution=list(train_df["output_solution"]),
        )

        eval_inputs = self.tokenizer(
            list(eval_df[self.input_column]),
            padding=True,
            truncation=True,
            max_length=512,
        )
        eval_dataset = datasets.Dataset.from_dict(
            {
                "input_ids": eval_inputs["input_ids"],
                "attention_mask": eval_inputs["attention_mask"],
            }
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=DataCollatorForSeq2Seq(self.tokenizer),
        )
        logging.info("Loaded datasets successfully.")
        return train_dataset, eval_loader, train_df, eval_df

    def _initialize_trainer(self):
        """Initialize the PPO trainer."""
        ppo_config = PPOConfig(
            remove_unused_columns=False,
            batch_size=8,
            mini_batch_size=4,
            gradient_accumulation_steps=2,
            max_grad_norm=0.5,
            early_stopping=True,
            ppo_epochs=1,
            learning_rate=1e-5,
            adap_kl_ctrl=False,
            init_kl_coef=0.2,
            vf_coef=0.01,
            target=6,
            cliprange=0.05,
            cliprange_value=0.2,
            log_with="wandb",
            optimize_cuda_cache=True,
        )
        logging.info(f"ppo_config: {ppo_config}")
        ppo_trainer = PPOTrainer(
            ppo_config,
            self.model,
            self.model_ref,
            self.tokenizer,
            dataset=self.train_dataset,
        )
        return ppo_trainer

    def train(self):
        """Train the model using PPO."""
        generation_kwargs = {
            "min_new_tokens": 5,
            "temperature": 1.0,
            "top_k": 10,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens": 150,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        rewards_function = {
            "pb_error": 0,
            "syntax_error": -2,
            "runtime_error": -1,
            "assert_error": -0.3,
            "executable": 2,
        }
        logging.info("Training start.")
        logging.info(f"generation_kwargs: {generation_kwargs}")
        logging.info(f"rewards_function: {rewards_function}")

        step = 0
        highest_reward = None
        for epoch in range(3):
            for batch in tqdm(self.ppo_trainer.dataloader):
                query_tensors = [r for r in batch["input_ids"]]
                response_tensors = self.ppo_trainer.generate(
                    query_tensors, return_prompt=False, **generation_kwargs
                )
                decoded_outputs = self.tokenizer.batch_decode(
                    response_tensors,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                batch["response"] = decoded_outputs

                rewards = self._calculate_rewards(
                    batch, decoded_outputs, rewards_function
                )

                train_stats = self.ppo_trainer.step(
                    query_tensors, response_tensors, rewards
                )
                self.ppo_trainer.log_stats(train_stats, batch, rewards)

                curr_mean_reward = torch.tensor(rewards).mean()
                if step == 0:
                    highest_reward = curr_mean_reward
                step += 1
                if curr_mean_reward >= highest_reward:
                    highest_reward = curr_mean_reward
                    self._save_checkpoint(step, curr_mean_reward)
                elif step % 50 == 0 and self.do_eval:
                    self._evaluate(step)

                if step > 300:
                    break

    def _calculate_rewards(self, batch, decoded_outputs, rewards_function):
        """Calculate rewards for the generated responses."""
        rewards = []
        for (
            input_text,
            pred_text,
            fn_name,
            prompt_code,
            output_solution,
        ) in zip(
            batch["query"],
            batch["response"],
            batch["fn_name"],
            batch["prompt_code"],
            batch["output_solution"],
        ):

            reward = 0.0
            tmp_rewards, errors = test_function(
                prompts=[prompt_code],
                solutions=[output_solution],
                testcases=[pred_text],
                fn_names=[fn_name],
                debug=False,
                filter_syntax=False,
                rewards=rewards_function,
            )
            reward += tmp_rewards[0]

            if not errors[0]:
                try:
                    script = transform_to_input(
                        prompt=prompt_code,
                        solution=output_solution,
                        testcase=pred_text,
                        fn_name=fn_name,
                        filter_syntax=False,
                    )
                    _, coverage_sc, executable = self.commander.process_coverage_test(
                        script
                    )
                    coverage_score = coverage_sc / 50 if executable else 0
                except:
                    coverage_score = 0
                reward += coverage_score

            rewards.append(torch.tensor(reward))
        return rewards

    def _save_checkpoint(self, step, curr_mean_reward):
        """Save model checkpoint based on the current mean reward."""
        save_point_dir = (
            self.output_dir + f"/step-{step}-{round(float(curr_mean_reward),2)}"
        )
        if not os.path.exists(save_point_dir):
            os.makedirs(save_point_dir)
        self.model.save_pretrained(save_point_dir)
        self.tokenizer.save_pretrained(save_point_dir)

    def _evaluate(self, step):
        """Evaluate the model during training."""
        eval_generation_kwargs = {
            "min_new_tokens": 5,
            "max_new_tokens": 300,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_beams": self.beam_size,
        }
        coverage_mean = 0
        eval_predictions = []
        for eval_batch in tqdm(self.eval_loader):
            with torch.no_grad():
                query_tensors = [r for r in eval_batch["input_ids"]]
                outputs = self.ppo_trainer.generate(
                    query_tensors, **eval_generation_kwargs
                )
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                eval_predictions.extend(decoded_outputs)
        eval_first_predictions = [pred[0] for pred in eval_predictions]

        eval_fn_names = list(self.eval_df["fn_name"])
        eval_prompt_codes = list(self.eval_df["prompt_code"])
        eval_output_solutions = list(self.eval_df["output_solution"])

        for (
            eval_prompt_code,
            eval_output_solution,
            eval_first_prediction,
            eval_fn_name,
        ) in zip(
            eval_prompt_codes,
            eval_output_solutions,
            eval_first_predictions,
            eval_fn_names,
        ):
            script = transform_to_input(
                prompt=eval_prompt_code,
                solution=eval_output_solution,
                testcase=eval_first_prediction,
                fn_name=eval_fn_name,
                filter_syntax=False,
            )
            try:
                _, coverage_sc, executable = self.commander.process_coverage_test(
                    script
                )
                coverage_score = coverage_sc if executable else 0
            except:
                coverage_score = 0
            coverage_mean += coverage_score
        coverage_mean /= len(self.eval_df)
        logging.info(f"-- Eval step: {step} --")
        logging.info(f"coverage: {coverage_mean}%")


if __name__ == "__main__":
    training = PPOTraining()
    training.train()
    logging.info("Finish the PPO training program.")
