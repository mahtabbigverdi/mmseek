

import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os
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
        do_sample=False, 
        temperature=0,        
        num_beams=1,
        use_cache=True,
    )
    # Trim the prompt part before decoding
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # out_text = processor.batch_decode(gen_ids)
    return out_text[0]


def eval(model, processor, prompt, images, max_new_tokens=2048):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image} for image in images]+[{"type": "text", "text": prompt}],
    }]
    output = generate_greedy(model, processor, messages, max_new_tokens=max_new_tokens)
    return output


def save_evals(model, processor ,model_id):
    val_data_path  =  "/mmfs1/gscratch/krishna/mahtab/mmseek/Data/large_val_path_tracing_2point_16k_withhint.json"
    # val_data_path  =  "/mmfs1/gscratch/krishna/mahtab/mmseek/Data/temp_validation.json"
    import json
    answers = {}
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    for item in val_data:
        index = item['id']
        image_paths = item['image']
        assert len(image_paths) == 2, f"Expected 2 images, got {len(image_paths)}"
        prompt = item['conversations'][0]['value'].replace('<image>\n', '')
        print(prompt)
        output = eval(model, processor, prompt, image_paths, max_new_tokens=2048)
        answers[index] = output
        print("Output:", output)
        print(f"Index: {index}")
    save_path = f"outputs/{model_id}_largeval_path_tracing_2pointwithhint_answers.json"
    with open(save_path, 'w') as f:
        json.dump(answers, f, indent=4)

    
        

def main(model_name):
    parser = argparse.ArgumentParser()
    lora_adapter = None
    merge_lora = False

    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model, processor = load_model_and_processor(
        model_name,
        lora_adapter=lora_adapter,
        merge_lora=merge_lora,
        dtype=dtype,
        device_map="auto"
    )
    save_evals(model, processor, model_name.split("/")[-1])



if __name__ == "__main__":
    #get model name from command line
    import sys
    model_name = sys.argv[1]
    main(model_name)