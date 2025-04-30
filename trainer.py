import os
import logging
import traceback
import math
import torch
from datasets import load_dataset, load_from_disk
from torch.optim.lr_scheduler import CosineAnnealingLR

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from lion_pytorch import Lion

torch.set_float32_matmul_precision('high')  # Enable TF32 for faster training (only for modern BERT)

def main():

    # Use the mounted volume path for all persistent storage
    MODEL_SAVE_PATH = "/root/cross-encoders/models"

    # Create paths for dataset storage within the mounted volume
    DATASET_DIR = "/root/cross-encoders/datasets"
    TRAIN_DATASET_PATH = os.path.join(DATASET_DIR, "ms-marco-train")
    EVAL_DATASET_PATH = os.path.join(DATASET_DIR, "ms-marco-eval")

    # Create the dataset directory if it doesn't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    model_name = "answerdotai/ModernBERT-base"

    pre_trained_path = None  # Set to None if you want to use the base model or full path to a pre-trained model, e.g. f"{MODEL_SAVE_PATH}/reranker-MiniLM-L12-H384-uncased-msmarco-bce-ep-2/final"

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    train_batch_size = 64
    num_epochs = 3
    dataset_size = 2_000_000

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    if pre_trained_path:
        model = CrossEncoder(pre_trained_path, trust_remote_code=True)
        print("Model max length:", model.max_length)
        print("Model num labels:", model.num_labels)
    else:
        model = CrossEncoder(model_name, trust_remote_code=True)
        print("Model max length:", model.max_length)
        print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    logging.info("Read train dataset")
    try:
        train_dataset = load_from_disk(TRAIN_DATASET_PATH)  # Updated path
        eval_dataset = load_from_disk(EVAL_DATASET_PATH)    # Updated path
    except FileNotFoundError:
        logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
        corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
        corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
        queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
        queries = dict(zip(queries["query_id"], queries["query"]))
        dataset = load_dataset("sentence-transformers/msmarco", "triplets", split="train")
        dataset = dataset.select(range(dataset_size // 2))

        def id_to_text_map(batch):
            return {
                "query": [queries[qid] for qid in batch["query_id"]] * 2,
                "passage": [corpus[pid] for pid in batch["positive_id"]]
                + [corpus[pid] for pid in batch["negative_id"]],
                "score": [1.0] * len(batch["positive_id"]) + [0.0] * len(batch["negative_id"]),
            }

        dataset = dataset.map(id_to_text_map, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=10_000)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Save to persistent storage in the mounted volume
        train_dataset.save_to_disk(TRAIN_DATASET_PATH)  # Updated path
        eval_dataset.save_to_disk(EVAL_DATASET_PATH)    # Updated path
        
        logging.info(
            f"The dataset has now been stored in the mounted volume at {DATASET_DIR}. "
            "The script will now stop to ensure that memory is freed. "
            "Please restart the script to start training."
        )
        quit()
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = BinaryCrossEntropyLoss(model)

    # 4. Define the evaluator. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    evaluator(model)
    

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-msmarco-bce-AdamW.Cosine-ep-1-3"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"{MODEL_SAVE_PATH}/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=4000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
        dataloader_num_workers=4,
        resume_from_checkpoint=True
    )

    # 6. Setup the optimizer and scheduler

    # Setup Lion optimizer
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.99)  # Default Lion betas
    )

    optimizer_adamW = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = math.ceil(dataset_size / train_batch_size) * num_epochs

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 7. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        optimizers = (optimizer_adamW, scheduler),  # Pass the optimizer and scheduler to the trainer
    )
    trainer.train()

    # 8. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 9. Save the final model
    final_output_dir = f"{MODEL_SAVE_PATH}/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 10. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )
    
if __name__ == "__main__":
    main()