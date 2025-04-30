import modal

modal_volume = modal.Volume.from_name("cross-encoders", create_if_missing=True)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
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
    )
).add_local_dir("D:\\Thesis\\4th Sem\\cross_encoder", remote_path="/root/cross_encoder")


MOUNT_DIR = "/root/cross-encoders"
GPU = "L40S"
GPU_QUANTITY = 3

app = modal.App(
    name="cross-encoder-trainer",
    image=image,
    volumes={MOUNT_DIR: modal_volume},
)



@app.function(
    gpu=f"{GPU}:{GPU_QUANTITY}",
    timeout=72000,
    secrets=[modal.Secret.from_name("my-huggingface-secret"), modal.Secret.from_name("my-wandb-secret")],
)
def trainer():
    import os

    os.system(f"torchrun --nproc_per_node={GPU_QUANTITY} cross_encoder/trainer.py")


@app.local_entrypoint()
def main():
    """
    Main entry point for the evaluation script.

    This function defines a list of pretrained models to be evaluated and calls the evaluate function
    to perform the evaluation.

    Args:
        None
    """
    trainer.remote()

# To run this:
# - Make sure you have modal api-key configured along with huggingface & wandb api keys configured in modal secrets.
# - Run `modal run modal_trainer_offload.py`
# - You can check the results in the `cross-encoders` volume