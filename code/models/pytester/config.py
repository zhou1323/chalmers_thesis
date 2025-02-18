# Configuration settings for model inference, training, evaluation, and fine-tuning
import torch
import os


class Config:
    # General settings
    gpu_num = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Inference settings
    inference = {
        "model_name": "codet5",
        "model_dir": "save/APPS/codet5-testcase-prompt_text/checkpoint-418",
        "test_dataset_path": "dataset/APPS_new/test_data.csv",
        "input_column": "prompt_testcase",
        "output_column": "output_testcase",
        "pred_mode": "multi",
        "output_dir": "save/APPS/codet5-testcase-prompt_text/checkpoint-418",
        "output_suffix": "_beam300",
        "batch_size": 5,
        "beam_size": 3,
        "max_new_tokens": 300,
        "max_length": 512,
    }

    # PPO Training settings
    ppo_training = {
        "model_name": "codet5",
        "unique_name": "codet5-testcase-prompt_text",
        "model_dir": "save/APPS/codet5-testcase-prompt_text/checkpoint-418",
        "train_dataset_path": "dataset/APPS_new/train_data.csv",
        "eval_dataset_path": "dataset/APPS_new/test_data.csv",
        "input_column": "prompt_testcase",
        "output_column": "output_testcase",
        "output_dir": "save/APPS/codet5-testcase-prompt_text/PPO_multi_cec-reward24_hyper6-13",
        "beam_size": 3,
        "do_eval": False,
    }

    # Evaluation settings
    evaluation = {
        "test_dataset_path": "datasets/APPS/test_data.csv",
        "input_column": "prompt_testcase",
        "output_column": "output_testcase",
        "result_mode": "codet5",
        "pred_mode": "multi",
        "n_pred_rank": 0,
        "n_test": 5,
        "on_save": False,
        "output_dir": "codet5_outputs",
        "gen_testcase_path": "codet5_outputs/prediction-3_beam300.pkl",
    }

    # Fine-tuning settings
    finetuning = {
        "model_name": "codet5-large",
        "is_shuffle": True,
        "train_dataset_path": "dataset/APPS_new/train_data.csv",
        "input_column": "prompt_testcase",
        "output_column": "output_testcase",
        "output_dir": "save/APPS/codet5-testcase-prompt_text",
    }
