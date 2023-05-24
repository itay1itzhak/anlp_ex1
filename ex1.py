import argparse
import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer #BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
#from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
import datasets
import wandb
import numpy as np

def load_dateset(tokenizer, training_samples, validation_samples, test_samples):
    # Define the dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("sst")
    
    # get training, validation and test sets
    train_dataset = dataset['train'][training_samples:]
    val_dataset = dataset['validation'][validation_samples:]
    test_dataset = dataset['test'][test_samples:]
    
    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    # Preprocess the dataset

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    #return dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='', max_length=512), batched=True)
    

def train(model ,tokenizer, train_data, eval_data, seed):
    # Define the training arguments

    training_args = TrainingArguments(
        output_dir='./results',          # Directory where checkpoints and logs will be saved
        evaluation_strategy="epoch",     # Evaluate model after each epoch
        logging_strategy="epoch",        # Log metrics after each epoch
        save_strategy="no",           # Save checkpoint after each epoch
        #per_device_train_batch_size=16,  # Batch size for training
        #per_device_eval_batch_size=64,   # Batch size for evaluation
        #evaluation_strategy="epoch",     # Evaluate model after each epoch
        disable_tqdm=True,               # Disable tqdm progress bar
        load_best_model_at_end=True,     # Load the best model at the end of training
        metric_for_best_model="accuracy",# Metric to determine the best model
        seed = seed,                     # Seed for experiment reproducibility
        data_seed = seed,                # Seed for data shuffling reproducibility
        report_to="wandb",               # Enable Weights&Biases logging
        run_name = f"{model}_{seed}",    # Name of the W&B run
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=datasets.load_metric("accuracy"),
        preprocess_function=preprocess_function,
    )
    
    # Initialize the Weights&Biases integration
    run = wandb.init(project='anlp1_sentiment-analysis', config=training_args, name=model)
    wandb.watch(model)
    
    # Fine-tune the model
    trainer.train()
    
    # Finish the Weights&Biases run
    run.finish()

    return model, trainer

def load_model(model):
    # Use the AutoModelForSequenceClassification class to load your models.
    if 'bert' in model:
        #tokenizer_name = 'bert-base-uncased'
        model_name = 'bert-base-uncased'
    elif 'roberta' in model:
        #tokenizer_name = 'roberta-base'
        model_name = 'roberta-base'
    elif 'electra' in model:
        #tokenizer_name = 'google/electra-base-generator'
        model_name = 'google/electra-base-generator'
    else:
        print("Invalid model name!")
        raise ValueError
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    return model, tokenizer

def run_main(args):
    logging.info(f"Training models: {args.model}\nOn {args.num_of_seeds} seeds\nWith {args.training_samples} training samples\n{args.validation_samples} validation samples\n{args.test_samples} test samples.")
    logging.info("\n" + "=" * 80 + "\n")

    # The mean and std of the accuracies should be documented in the res.txt file
    models_res = {}
    #best_model = {"model": None, "accuracy": 0, "seed": None}
    res = open("res.txt", "w")
    # 
    if args.models == 'all':
        all_models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
    else:
        all_models = args.models

    # Load the dataset
    dataset = load_dateset(tokenizer, args.training_samples, args.validation_samples, args.test_samples)

    for model_name in all_models:
        models_res[model_name] = {"accuracies": [], "run_times": [], "best_model": None, "best_accuracy": 0, "best_seed": None}
        model, tokenizer = load_model(model_name)
        logging.info(f"Model: {model}")
        for seed in range(args.num_of_seeds):
            logging.info(f"Seed: {seed}/{args.num_of_seeds}")
            # Set up Weights&Biases
            wandb.login()
            model, trainer = train(model, tokenizer,dataset,seed)
            # Evaluate the model
            eval = trainer.evaluate()
            # Run the training and evaluation
            models_res[model_name]["accuracies"].append(eval['eval_accuracy'])
            # Save the accuracy and the run time
            models_res[model_name]["run_times"].append(eval['training_duration'])
            # Save the best model
            if eval['eval_accuracy'] > models_res[model_name]["best_accuracy"]['accuracy']:
                models_res[model_name]["best_accuracy"]['best_model'] = model
                models_res[model_name]["best_accuracy"]['best_accuracy'] = eval['eval_accuracy']
                models_res[model_name]["best_accuracy"]['best_seed'] = seed
        
        logging.info(f"Model accuracies: "+ models_res[model_name]["accuracies"])
        logging.info(f"Model run times: " + models_res[model_name]["accuracies"])
        mean_model_accuracy = np.mean(models_res[model_name]["accuracies"])
        mean_model_std = np.std(models_res[model_name]["accuracies"])
        models_res[model_name]["mean_accuracy"] = mean_model_accuracy
        # save to res.txt in the format <model name>,<mean accuracy> +- <accuracy std>
        res.write(f"{model_name},{mean_model_accuracy} +- {mean_model_std}\n")
    
    # Get the index of the model with the highest mean accuracy on the validation set
    model_accuracies = [(model_name,models_res[model_name]["mean_accuracy"]) for model_name in models_res.keys()]
    best_model_name = max(model_accuracies, key=lambda x: x[1])
    best_model = models_res[best_model_name[0]]["best_model"]

    # run prediction on the best model and generate predictions.txt    
    # load the test set
    test_set = load_test_set(tokenizer, args.test_samples)
    # run prediction
    # save the predictions to a file
    
    # Select the model with the highest mean accuracy on the validation set, and use its best seed to run prediction on the test set
    max_accuracy = max(model_accuracies)
    max_accuracy_index = model_accuracies.index(max_accuracy)
    max_accuracy_seed = args.seeds[max_accuracy_index]
    logging.info(f"Max accuracy: {max_accuracy} with seed: {max_accuracy_seed}")

    # run prediction on the test set
    # load the best model
    # load the test set
    # run prediction
    # save the predictions to a file


    res.close()
    logging.info("\n" + "=" * 80 + "\n")
    logging.info("All Done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_of_seeds",
        type=int,
        default=1,
        help="How many seeds to use.",
    )
    parser.add_argument(
        "--training_samples",
        type=int,
        default=-1,
        help="How many training samples to use.",
    )

    parser.add_argument(
        "--validation_samples",
        type=int,
        default=-1,
        help="How many validation samples to use.",
    )

    parser.add_argument(
        "--test_samples",
        type=int,
        default=-1,
        help="How many test samples to use.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Which model to use, all is default for all of them.",
    )


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_main(args)