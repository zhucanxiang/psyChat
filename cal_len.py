
import os
import json
import numpy as np

with open('/root/zhucanxiang/data/APA_ORIGIN_DATA/apa_origin_data_translate_v2_train/train.json', 'r') as fin:
    lens = []
    gt_max_len = 0
    for line in fin.readlines():
        item = json.loads(line.strip())
        lens.append(len(item['response']))
        if (len(item['prompt']) > 2048):
            gt_max_len += 1
    np_lens = np.array(lens)
    print('总数: ', len(lens))
    print('平均长度: ', np_lens.mean())
    print('大于2048个数: ', gt_max_len)
    for i in range(1, 100, 1):
        print('{}%分位长度: '.format(str(i)), np.percentile(np_lens, i))





