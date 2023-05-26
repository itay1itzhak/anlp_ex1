import argparse
import logging
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer #BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
#from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import datasets
import evaluate
import wandb
import numpy as np
import time


def load_dateset(training_samples, validation_samples, test_samples):
    # Define the dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("sst2")
    
    # get training, validation and test sets
    train_dataset = dataset['train'].select(range(training_samples))
    val_dataset = dataset['validation'].select(range(validation_samples))
    test_dataset = dataset['test'].select(range(test_samples))
    
    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    # Preprocess the dataset

def preprocess_function(examples, tokenizer):
    # use dynamic padding
    #max_length = max([len(examples['sentence'][i].split()) for i in range(len(examples['sentence']))])
    # Tokenize the texts
    result = tokenizer(examples['sentence'], truncation=True, max_length=512)
    # Map labels from 0-4 to 0-1
    print(f"preprocess_function:Before{examples['label']=}")
    #result["label"] = [0 if label == -1 else 1 for label in examples["label"]]#[label for label in examples["label"]]
    #print(f"preprocess_function:After{result['label']=}")
    print(f"preprocess_function:{result['input_ids']=}")
    # print(f"{len(result['input_ids'][0])=}")
    
    return result
    
def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")#, "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    print(f"compute_metrics:{labels=}")
    return metric.compute(predictions=predictions, references=labels)

def train(model ,tokenizer, train_data, eval_data, seed,model_name):
    # Define the training arguments

    # map the preocess function to the dataset
    train_dataset = train_data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    val_dataset = eval_data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir='./results',          # Directory where checkpoints and logs will be saved
        #evaluation_strategy="epoch",     # Evaluate model after each epoch
        logging_strategy="epoch",        # Log metrics after each epoch
        save_strategy="no",           # Save checkpoint after each epoch
        #per_device_train_batch_size=16,  # Batch size for training
        #per_device_eval_batch_size=64,   # Batch size for evaluation
        #evaluation_strategy="epoch",     # Evaluate model after each epoch
        disable_tqdm=True,               # Disable tqdm progress bar
        #load_best_model_at_end=True,     # Load the best model at the end of training
        #metric_for_best_model="accuracy",# Metric to determine the best model
        seed = seed,                     # Seed for experiment reproducibility
        data_seed = seed,                # Seed for data shuffling reproducibility
        report_to="wandb",               # Enable Weights&Biases logging
        run_name = f"{model_name}_{seed}",    # Name of the W&B run
    )

    print(f"{seed=}")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,#datasets.load_metric("accuracy"),
        #group_by_length=True,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=512)
    )

    # Overfiting a batch
    # for batch in trainer.get_train_dataloader():
    #   break
    # #batch = {k: v.to(device) for k, v in batch.items()}
    # trainer.create_optimizer()
    # for _ in range(20):
    #     outputs = trainer.model(**batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     trainer.optimizer.step()
    #     trainer.optimizer.zero_grad()
    # with torch.no_grad():
    #   outputs = trainer.model(**batch)
    # preds = outputs.logits
    # labels = batch["labels"]
    #print(f"{compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))=}")
    
    #run = wandb.init(project='anlp1_sentiment-analysis', config=training_args, name=model)
    #wandb.watch(model)
    
    # Fine-tune the model
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Finish the Weights&Biases run
    #run.finish()

    return model, trainer, end_time-start_time

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)#.cuda()

    return model, tokenizer

def write_best_model_prediction(models_res, test_set,res_file):
    # Get the index of the model with the highest mean accuracy on the validation set
    model_accuracies = [(model_name,models_res[model_name]["mean_accuracy"]) for model_name in models_res.keys()]
    best_model_name = max(model_accuracies, key=lambda x: x[1])
    #best_model = models_res[best_model_name[0]]["best_model"]
    best_trainer = models_res[best_model_name[0]]["best_trainer"]
    best_tokenizer = models_res[best_model_name[0]]["best_tokenizer"]
    best_train_time = models_res[best_model_name[0]]["best_train_time"]
    # print best model name
    logging.info(f"Best model: {best_model_name[0]}")


    
    test_set_tokenized = test_set.map(lambda examples: preprocess_function(examples, best_tokenizer), batched=True)
    # run prediction on the best model one by one with no padding and generate predictions.txt    
    #predictions = best_trainer.predict(test_set_tokenized)
    # get the predictions
    #predictions = predictions.predictions.argmax(-1)
    test_set_tokenized = test_set_tokenized.remove_columns("label")
    print(f"{test_set_tokenized=}")
    predictions = best_trainer.predict(test_set_tokenized, metric_key_prefix="predict")#.predictions
    predictions_labels = np.argmax(predictions.predictions, axis=-1)
    # get the prediction run time
    #print(f"{predictions=}")
    prediction_run_time = predictions.metrics['predict_runtime']
    # save the predictions to a file in the following format: <input sentence>###<predicted label 0 or 1>
    with open("predictions.txt", "w") as f:
        for prediction,label in zip(test_set,predictions_labels):
            f.write(f"{prediction['sentence']}###{label}\n")


    # Select the model with the highest mean accuracy on the validation set, and use its best seed to run prediction on the test set
    max_accuracy = max(model_accuracies)
    max_accuracy_index = model_accuracies.index(max_accuracy)
    max_accuracy_seed = max_accuracy_index
    logging.info(f"Max accuracy: {max_accuracy} with seed: {max_accuracy_seed}")

    # save to res.txt train time of the best model in the format: train time,<train time in seconds>
    res_file.write(f"train time,{best_train_time}\n")
    # save to res.txt prediction time of the best model in the format: predict time,<predict time in seconds>
    res_file.write(f"predict time,{prediction_run_time}\n")

def run_main(args):
    logging.info(f"Training models: {args.models}\nOn {args.seeds} seeds\nWith {args.training_samples} training samples\n{args.validation_samples} validation samples\n{args.test_samples} test samples.")
    logging.info("\n" + "=" * 80 + "\n")

    if args.models == 'all':
        all_models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
    else:
        all_models = args.models.split(',')
    all_seeds = range(args.seeds)

    # Load the dataset
    dataset = load_dateset(args.training_samples, args.validation_samples, args.test_samples)

    # Initialize the Weights&Biases integration
    #run = wandb.init(project='anlp1_sentiment-analysis', config=training_args, name=model)

    # The mean and std of the accuracies should be documented in the res.txt file
    models_res = {}
    #best_model = {"model": None, "accuracy": 0, "seed": None}
    res_file = open("res.txt", "w")

    for model_name in all_models:
        models_res[model_name] = {"accuracies": [], "run_times": [], "best_model": None, "best_accuracy": 0, "best_seed": None}
        model, tokenizer = load_model(model_name)
        logging.info(f"Model: {model}")
        for seed in all_seeds:
            logging.info(f"Seed: {seed}/{args.seeds}")
            # Set up Weights&Biases
            #wandb.login(key='4488053a16542eb8f67da602ac86aa39094a2924')
            # Run the training and evaluation
            model, trainer, train_time = train(model, tokenizer,dataset['train'],dataset['validation'],seed,model_name)
            # Evaluate the model
            #eval = trainer.evaluate(eval_dataset=dataset['validation'].map(lambda examples: preprocess_function(examples, tokenizer), batched=True))#dataset['validation'])
            eval = trainer.evaluate(eval_dataset=dataset['train'].map(lambda examples: preprocess_function(examples, tokenizer), batched=True))
            wandb.finish()
            # Save the accuracy and the run time
            models_res[model_name]["accuracies"].append(eval['eval_accuracy'])
            models_res[model_name]["run_times"].append(train_time)
            # Save the best model
            if eval['eval_accuracy'] > models_res[model_name]["best_accuracy"]:
                models_res[model_name]['best_model'] = model
                models_res[model_name]['best_trainer'] = trainer
                models_res[model_name]['best_tokenizer'] = tokenizer
                models_res[model_name]['best_accuracy'] = eval['eval_accuracy']
                models_res[model_name]['best_seed'] = seed
                models_res[model_name]['best_train_time'] = train_time
            # print the accuracies and run times of the model
            logging.info(f"Model accuracy: {eval['eval_accuracy']}")
            logging.info(f"Model run time: {train_time}")
            logging.info("\n" + "-" * 50 + "\n")
        
        logging.info(f"Model accuracies: "+ str(models_res[model_name]["accuracies"]))
        logging.info(f"Model run times: " + str(models_res[model_name]["run_times"]))
        mean_model_accuracy = np.mean(models_res[model_name]["accuracies"])
        mean_model_std = np.std(models_res[model_name]["accuracies"])
        models_res[model_name]["mean_accuracy"] = mean_model_accuracy
        logging.info(f"Model mean accuracy: {mean_model_accuracy} +- {mean_model_std}")
        # save to res.txt in the format <model name>,<mean accuracy> +- <accuracy std>
        res_file.write(f"{model_name},{mean_model_accuracy} +- {mean_model_std}\n")
        logging.info("\n" + "=" * 80 + "\n")
    
    write_best_model_prediction(models_res,dataset['test'],res_file)

    res_file.close()
    logging.info("\n" + "=" * 80 + "\n")
    logging.info("All Done!")

    return models_res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_of_seeds",
        type=int,
        default=3,
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