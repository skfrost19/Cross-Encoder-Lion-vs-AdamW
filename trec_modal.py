import modal

modal_volume = modal.Volume.from_name("cross-encoders", create_if_missing=True)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "curl", "unzip", "openjdk-21-jdk")
    .pip_install(  # required to build flash-attn
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "lion-pytorch",
        "numpy",
        "scikit-learn",
        "hf_xet",
        "tiktoken",
        "sentence-transformers",
        "wandb",
        "accelerate>=0.26.0",
        "ir_datasets",
        "pyserini",
        "faiss-cpu",
    )
    .run_commands("git clone https://github.com/usnistgov/trec_eval.git && cd trec_eval && make")
).add_local_dir("D:\\Thesis\\4th Sem\\cross_encoder", remote_path="/root/cross_encoder")



MOUNT_DIR = "/root/cross-encoders"
ALL_MODEL_PATH = "/root/cross-encoders/models"

app = modal.App(
    name="trec-eval",
    image=image,
    volumes={MOUNT_DIR: modal_volume},
)


def evaluate_model(model_path):
    import subprocess

    """Run evaluation for a single model in a separate process"""
    print(f"Starting evaluation for model: {model_path}")
    MODEL_PATH, ext = model_path.split("/")
    print(MODEL_PATH, ext)
    cmd = f"python /root/cross_encoder/trec_dl_19_eval_2.py --model_name {ALL_MODEL_PATH}/{MODEL_PATH}/{ext} --runs_dir {ALL_MODEL_PATH}/runs --results_dir {ALL_MODEL_PATH}/{MODEL_PATH}/results --cache_dir {MODEL_PATH}/cache --batch_size 1000"
    
    try:
        # Run in subprocess instead of using os.system()
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=False,  # Don't raise error if process returns non-zero exit code
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log results
        print(f"Evaluation completed for {model_path}")
        print(f"Return code: {result.returncode}")
        print(f"Output sample: {result.stdout[:500]}...")
        
        if result.returncode != 0:
            print(f"Error in model {model_path}: {result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Exception when evaluating {model_path}: {str(e)}")
        return False


@app.function(
    gpu="L40S:1",
    timeout=72000,
    secrets=[modal.Secret.from_name("my-huggingface-secret"), modal.Secret.from_name("my-wandb-secret")],
)
def evaluate_trec_dl():

    import os
    # Use a user-accessible directory
    cache_dir = os.path.expanduser("~/.cache/pyserini")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if the source directory exists
    pyserini_src = "/root/longformer_crossencoder/pyserini"
    if os.path.exists(pyserini_src):
        print(f"Source directory {pyserini_src} exists, creating symlink")
        # If cache_dir exists but is not a symlink, remove it
        if os.path.exists(cache_dir) and not os.path.islink(cache_dir):
            os.system(f"rm -rf {cache_dir}")
        # Create the symlink
        if not os.path.exists(cache_dir):
            os.symlink(pyserini_src, cache_dir)
    else:
        print(f"Source directory {pyserini_src} does not exist, creating directory")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Check if ir_datasets source directory exists
    ir_datasets_src = "/root/longformer_crossencoder/.ir_datasets"
    ir_datasets_dest = os.path.expanduser("~/.ir_datasets")
    if os.path.exists(ir_datasets_src):
        print(f"IR datasets directory {ir_datasets_src} exists, creating symlink")
        if os.path.exists(ir_datasets_dest) and not os.path.islink(ir_datasets_dest):
            os.system(f"rm -rf {ir_datasets_dest}")
        if not os.path.exists(ir_datasets_dest):
            os.symlink(ir_datasets_src, ir_datasets_dest)
    else:
        print(f"IR datasets directory {ir_datasets_src} does not exist, creating directory")
        os.makedirs(ir_datasets_dest, exist_ok=True)
    

    # Define the models path and run the evaluation script
    model_paths = ["reranker-ModernBERT-base-msmarco-bce-AdamW.Cosine-ep-1-3/checkpoint-20728"]
    print(f"Models to evaluate: \n{model_paths}")    
    
    # Run sequentially but in separate processes
    for model_path in model_paths:
        evaluate_model(model_path)


@app.function(
    gpu="A100-80GB:1",
    timeout=7200,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def ms_marco_mrr_eval():
    import os
    os.system(f"python cross_encoder/ms_marco_val_mrr.py --parent_dir {ALL_MODEL_PATH}/")


@app.local_entrypoint()
def main():
    """
    Main entry point for the evaluation script.

    This function defines a list of pretrained models to be evaluated and calls the evaluate function
    to perform the evaluation.

    Args:
        None
    """
    evaluate_trec_dl.remote()
    # ms_marco_mrr_eval.remote()