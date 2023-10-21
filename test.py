from open_flamingo import create_model_and_transforms
import os

llama_path = './models/llama-7b-hf'
if not os.path.exists(llama_path):
    raise ValueError('Llama model not yet set up, please check README for instructions!')

model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4,
        use_local_files=True
    )