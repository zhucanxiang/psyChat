import os
import json

class APADataPreprocess():
    def __init__(self):
        pass

    def generate_dataset(self, data_dir):
        out_filepath = data_dir + "_train/train.json"
        out_f = open(out_filepath, 'w', encoding='utf-8')
        prompt_template = "你是一名心理咨询师，现在在给患者做心理咨询。你跟患者的前10轮对话信息如下: {}, 其中`|||`为一轮中医生和患者说话内容的分割符, `###`为两轮对话间的分割符。 当前轮用户说的内容如下: {}, 根据你跟患者的当前轮对话的前10轮的对话信息，以及当前轮用户说的内容对患者进行回答。"
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                fin = open(filepath, 'r', encoding='utf-8')
                histories = []
                patient_content = ''
                for line in fin.readlines():
                    if line.startswith('cn_chat_病人'):
                        patient_content = "病人说:" + line.strip().replace('cn_chat_病人:', '')
                    elif line.startswith('cn_chat_医生'):
                        history_content = '###'.join(histories)
                        prompt = prompt_template.format(history_content, patient_content)
                        response = line.strip().replace('cn_chat_医生:', '')
                        data_item = {'prompt': prompt, 'reponse': response}
                        if patient_content:
                            out_f.write(json.dumps(data_item, ensure_ascii=False) + '\n')
                        histories.append('|||'.join([patient_content, '医生说:'+response]))
                        if len(histories):
                            histories = histories[-10:]
                fin.close()
        out_f.flush()
        out_f.close()

data_dir = "/root/zhucanxiang/data/APA_ORIGIN_DATA/apa_origin_data_translate_v2"
apa_data_preprocess = APADataPreprocess()
apa_data_preprocess.generate_dataset(data_dir)
