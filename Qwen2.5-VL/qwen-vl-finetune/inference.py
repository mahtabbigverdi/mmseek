
# import torch
# from PIL import Image
# from transformers import (
#     AutoTokenizer, AutoProcessor,
#     Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
# )

# # ---- change these two paths ----
# CKPT_DIR = "/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/checkpoints"   # or just "/path/to/your/output_dir"
# AUX_DIR  = "/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/checkpoints"                   # where tokenizer + image_processor were saved

# # Optional: switch to "sdpa" if you don't have flash-attn installed
# ATTN_IMPL = "flash_attention_2"  # or "sdpa"

# # If you trained a 2.5 model, flip this to True
# IS_QWEN_2_5 = True

# dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# # 1) Load model from the checkpoint you want to try
# ModelCls = Qwen2_5_VLForConditionalGeneration if IS_QWEN_2_5 else Qwen2VLForConditionalGeneration
# tokenizer  = AutoTokenizer.from_pretrained(AUX_DIR, use_fast=False)
# # import pdb; pdb.set_trace()
# processor  = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct") 


# model = ModelCls.from_pretrained(
#     CKPT_DIR,
#     torch_dtype=dtype,
#     attn_implementation=ATTN_IMPL,
#     device_map="auto",   # put layers on available GPU(s)
# )
# model.eval()

# # 2) Load the UPDATED tokenizer + image processor you saved during training
# #    (Your training script saves them at training_args.output_dir)
#  # has the image_processor inside

# # --- Safety check: the tokenizer must match the model's embedding size ---
# tok_len = len(tokenizer)

# emb_len = model.get_input_embeddings().num_embeddings
# print(len(tokenizer), emb_len)
# if tok_len > emb_len:
#     if tok_len != emb_len:
#         raise ValueError(
#             f"Tokenizer size ({tok_len}) != model embeddings ({emb_len}). "
#             f"Load the tokenizer saved in your training output_dir (AUX_DIR), not the base one."
#         )

# # 3) Helper: run a single image+text chat turn
# def generate(image_path, user_text, max_new_tokens=1000, temperature=0):
#     image = Image.open(image_path).convert("RGB")

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text",  "text": user_text},
#             ],
#         }
#     ]

#     # Qwen2-VL chat template
#     prompt = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = processor(
#         text=[prompt],
#         images=[image],
#         return_tensors="pt"
#     ).to(model.device, dtype=dtype if dtype != torch.float32 else None)

#     with torch.inference_mode():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=temperature > 0.0,
#             temperature=temperature,
#         )

#     # Only decode newly generated tokens
#     gen_ids = out[:, inputs["input_ids"].shape[1]:]
#     text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
#     return text.strip()

# if __name__ == "__main__":
#     # Example: use one of your domain-specific new tokens directly in the prompt if you added them
#     img_path = "/mmfs1/gscratch/krishna/mahtab/Aurora-perception/Data/evals/hardblink/images/blink5pointscenter/1.png"
#     question = "Multiple points are circled on the image, labeled by letters beside each circle. Which point is the closest to the camera?"

#     print(generate(img_path, question))


import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info

# Optional: only if you're using external LoRA adapters (kept separate)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
# DEFAULT_QWEN25_VL_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

def load_model_and_processor(
    model_id_or_path: str,
    lora_adapter: str | None = None,
    merge_lora: bool = False,
    device_map: str = "auto",
    dtype: torch.dtype | str = "auto",
):
    """
    model_id_or_path: your finetuned model (HF repo or local folder). If you merged LoRA already, just point here.
    lora_adapter: path or HF repo of LoRA adapter if NOT merged.
    merge_lora: set True to merge-and-unload LoRA into base weights for inference.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if lora_adapter:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed, but lora_adapter provided.")
        model = PeftModel.from_pretrained(model, lora_adapter)
        if merge_lora:
            # Permanently merge LoRA for faster inference
            model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(model_id_or_path)  # has the image_processor inside
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)

    processor.tokenizer = tok 
    processor.chat_template = tok.chat_template

    return model, processor


@torch.inference_mode()
def generate_greedy(model, processor, messages, max_new_tokens):
    """
    messages: Chat-style list with mixed text and images, e.g.
      [{
         "role": "user",
         "content": [
            {"type": "image", "image": "file:///path/to/img.jpg" or URL or PIL.Image},
            {"type": "text", "text": "Your prompt here"}
         ]
      }]
    """
    # Prepare inputs using the official chat template and utils
    text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Deterministic decoding: no sampling, temperature ignored (set to 0 for clarity)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # <-- greedy
        num_beams=1,
        use_cache=True,
    )
    # Trim the prompt part before decoding
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # out_text = processor.batch_decode(gen_ids)
    return out_text[0]


def main():
    parser = argparse.ArgumentParser()
    model =  "/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/checkpoints/3b_aurora_batch16_accum1"  
    # model ="/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/7b_aurora/checkpoint-1200"
    lora_adapter = None
    merge_lora = False
    max_new_tokens = 1024
    image = "/mmfs1/gscratch/krishna/mahtab/Aurora-perception/Data/evals/hardblink/images/blink5pointscenter/10.png"
    # prompt = "Multiple points are circled on the image, labeled by letters beside each circle. Which point is the closest to the camera?"
    prompt = "Multiple points are circled on the image, labeled by letters beside each circle. Which point is the closest to the camera?\nTo answer this question, let's think through it step by step, and we know the image is 336 x 336. First, what are the coordinates of points in the image? Second, what is the depth map for the image? Which point has a higher pixel value on the depth map? Remember, higher values indicate that the point is closer to the camera."
    # prompt = "What is the depth map of the image?"
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model, processor = load_model_and_processor(
        model,
        lora_adapter=lora_adapter,
        merge_lora=merge_lora,
        dtype=dtype,
        device_map="auto"
    )

    # Build messages (image + text, or text-only if no image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    
  

    output = generate_greedy(model, processor, messages, max_new_tokens=max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
