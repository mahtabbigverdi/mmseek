from transformers import AutoModelForImageTextToText, AutoProcessor


import argparse
import torch
from tqdm import tqdm
import os

def load_model_and_processor(
    model_id_or_path: str,
    device_map: str = "auto",
    dtype: torch.dtype | str = "auto",
):


    model = AutoModelForImageTextToText.from_pretrained(
        model_id_or_path, dtype=dtype, device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(model_id_or_path)
    return model, processor



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


    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False, num_beams=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]






def save_evals(model, processor ,model_id):
    val_data_path  =  "/mmfs1/gscratch/krishna/mahtab/mmseek/Data/validation_path_tracing_2point.json"
    import json
    answers = {}
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    for item in tqdm(val_data):
        index = item['id']
        image_path = item['image']
        prompt = item['conversations'][0]['value'].replace('<image>\n', '')
        assert prompt.split("?")[0].count("right") + prompt.split("?")[0].count("left") >= 1
        side = "right" if "right" in prompt.split("?")[0] else "left"
        oppside = "left" if side == "right" else "right"
        side_view_image = image_path.replace("topdown", "pathtracing_mask")
        messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"Hint: The {oppside} side has been masked out. Only the {side} side view from M1 is shown in the image below to assist:"},
            {"type": "image", "image": side_view_image},
            {"type": "text", "text": "Provide your reasoning, then at the end output your final answer in this exact format: \nAnswer: [A/B/C/D]"},
        ],
        }]
        output = generate_greedy(model, processor, messages, max_new_tokens=2048)
        answers[index] = output
        print(f"Index: {index}", prompt, side, output)
    save_path = f"outputs/{model_id}_path_tracing_2point_answers_mask.json"
    with open(save_path, 'w') as f:
        json.dump(answers, f, indent=4)

    
 
def main(model_name):
    parser = argparse.ArgumentParser()
    

    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model, processor = load_model_and_processor(
        model_name,
        dtype=dtype,
        device_map="auto"
    )
    save_evals(model, processor, model_name.split("/")[-1])



if __name__ == "__main__":
    #get model name from command line
    import sys
    model_name = sys.argv[1]
    main(model_name)


    # Qwen/Qwen3-VL-4B-Instruct