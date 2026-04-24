from huggingface_hub import hf_hub_download
# Ye model real/fake face classification karta hai
path = hf_hub_download(
    repo_id="prithivMLmods/Deep-Fake-Detector-Model",
    filename="model.safetensors"
)
