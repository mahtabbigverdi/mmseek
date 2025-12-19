import re



val_data_path  = "/mmfs1/gscratch/krishna/mahtab/mmseek/Data/validation_path_tracing_2point.json"
import json
with open(val_data_path, 'r') as f:
    val_data = json.load(f)


pred_path = "/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/baseline_checks/outputs/gpt-4.1_path_tracing_2point_answers_canny.json"
with open(pred_path, 'r') as f:
    predictions = json.load(f)


acc, total = 0,0
for item in val_data:
    index = str(item['id'])
    answer = item['conversations'][1]['value'][0].lower()
    match = re.search(r'Answer: ([ABCD])\s*', predictions[index])
    if match:
        pred = match.group(1)  # Gets the captured group (A, B, C, or D)
    else:
        match = re.search(r'answer: ([ABCD])\s*', predictions[index])
        if match:
            pred = match.group(1)
        else:
            print(index,  item['conversations'][0]['value'].replace('<image>\n', ''), predictions[index])
            pred = input()

    if answer in pred.lower():
        acc+=1
        print(index, pred, answer)
    else:
        pass
        # print(index,pred, answer)
    total +=1


print(acc, total, acc/total*100.0)


