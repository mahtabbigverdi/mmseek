import numpy as np
import os
import json
import re


with open("depth_codes_ade_vqgan8192.json", 'r') as f:
    data = json.load(f)

print(len(data))
s = set()
ma = 0
mi = 10000
for k, v in data.items():
    # print(k,v)
    ma = max(ma, max(v))
    mi = min(mi, min(v))
    s.update(v)
print(len(s), mi, ma)
# out_l = ['<DEPTH_START>', "<DEPTH_END>"]
# for el in sorted(s):
#     out_l.append("<DEPTH_" + str(el) + ">")
# print(len(out_l), out_l[:10], out_l[-10:])
# with open("/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/New_tokens_vqgan8192.txt", 'w') as f:
#     for el in out_l:
#         f.write(el + '\n')

with open("/mmfs1/gscratch/krishna/mahtab/LLaVA/train_annealing_data.json", 'r') as f:
    main_f = json.load(f)


new_f = []
for item in main_f:
    answer = item['conversations'][1]['value']
    img_id = item['image'].split('/')[-1].split('.')[0] + "_depth.png"
    depth_codes = data[img_id]
    new_depth = '<DEPTH_START>' + ''.join([f"<DEPTH_{el}>" for el in depth_codes]) + '<DEPTH_END>'

    answer = re.sub(r'<DEPTH_START>.+<DEPTH_END>', new_depth, answer)
    item['conversations'][1]['value'] = answer

    new_f.append(item)

with open("/mmfs1/gscratch/krishna/mahtab/LLaVA/train_annealing_data_vqgan8192.json", 'w') as f:
    json.dump(new_f, f)