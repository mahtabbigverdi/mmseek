

import argparse
import torch
from tqdm import tqdm
import os
from openai import OpenAI
import base64

client = OpenAI()



@torch.inference_mode()
def generate_greedy(model_id, messages, max_new_tokens):
   
# Load image and convert to base64


    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=0,
        messages=messages
        )

    return response.choices[0].message.content






def save_evals(model_id):
    val_data_path  =  "/mmfs1/gscratch/krishna/mahtab/mmseek/Data/validation_path_tracing_2point.json"
    import json
    answers = {}
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    for item in tqdm(val_data):
        index = item['id']
        image_path = item['image']
        with open(image_path, "rb") as img:
            img_b64 = base64.b64encode(img.read()).decode("utf-8")
        prompt = item['conversations'][0]['value'].replace('<image>\n', '')
        assert prompt.split("?")[0].count("right") + prompt.split("?")[0].count("left") >= 1
        side = "right" if "right" in prompt.split("?")[0] else "left"
        oppside = "left" if side == "right" else "right"
        side_view_image = image_path.replace("topdown", "pathtracing_grayscale")
        with open(side_view_image, "rb") as img:
            side_b64 = base64.b64encode(img.read()).decode("utf-8")
        messages = [{
        "role": "user",
        "content": [
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"Hint: A grayscale first-person view from position M1, looking {side}, is provided below:"},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{side_b64}"}},
            {"type": "text", "text": "Provide your reasoning, then at the end output your final answer in this exact format: \nAnswer: [A/B/C/D]"},
        ],
        }]
        output = generate_greedy(model_id, messages, max_new_tokens=2048)
        answers[index] = output
        print(f"Index: {index}", prompt, side, output)
    # save_path = f"outputs/{model_id}_path_tracing_2point_answers_raw.json"
    save_path = f"outputs/{model_id}_path_tracing_2point_answers_gray.json"
    with open(save_path, 'w') as f:
        json.dump(answers, f, indent=4)

    
        

def main(model_name):
    
    save_evals(model_name)



if __name__ == "__main__":
    #get model name from command line
    import sys
    model_name = sys.argv[1]
    main(model_name)

# gpt-4.1