#coding=utf-8
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

    def generate_dataset_v2(self, data_dir):
        out_filepath = data_dir + "_train/train.json"
        out_f = open(out_filepath, 'w', encoding='utf-8')
        prompt_template = "假设你是一名心理咨询师，你来帮助来访者解决心理问题。来访者说: ###{}###。 来访者说的内容被左右分别一个###符号包围。请你对来访者做出回应，回应要体现对来访者的关切和对来访者有启发。以下是对话的上文信息供参考。对话的上文信息格式如下：```{}```。上下文信息被```被左右分别一个```符号包围，上下文信息中的心理咨询师和来访者说的内容被左右分别一个###包围。"
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                fin = open(filepath, 'r', encoding='utf-8')
                rounds = []
                pre_role = "unknown"
                pre_content = ""
                for line in fin.readlines():
                    line = line.strip()
                    if line.startswith('cn_chat_病人'):
                        cur_content = line.strip().replace('cn_chat_病人:', '')
                        if pre_role == 'unknown':
                            pre_content = cur_content
                            pre_role = 'patient'
                        elif pre_role == "patient":
                            pre_role = 'patient'
                            pre_content += cur_content
                        else:
                            rounds.append('心理咨询师说:###' + pre_content + '###')
                            patient_content = cur_content
                            pre_role = "patient"
                            pre_content = patient_content
                    elif line.startswith('cn_chat_医生'):
                        cur_content = line.strip().replace('cn_chat_医生:', '')
                        if pre_role == 'unknown':
                            pre_content = cur_content
                            pre_role = 'doctor'
                        elif pre_role == 'doctor':
                            pre_role = 'doctor'
                            pre_content += cur_content
                        else:
                            rounds.append('来访者说:###' + pre_content + "###")
                            pre_role = 'doctor'
                            pre_content = cur_content
                if (len(rounds) > 1):
                    histories = []
                    for ro in rounds:
                        if ro.startswith('心理咨询师说:'):
                            context = ''
                            query = ''
                            if len(histories) == 0:
                                histories.append(ro)
                            elif len(histories) == 1:
                                query = histories[-1]
                                context = ''
                            elif len(histories) > 1 and len(histories) < 21:
                                query = histories[-1]
                                context = '\n'.join(histories[0:-1])
                            else:
                                query = histories[-1]
                                context = '\n'.join(histories[-21:-1])
                            if query != '':
                                query = query.replace('来访者说:', '').replace('###', '')
                                prompt = prompt_template.format(query, context)
                                response = ro.replace('心理咨询师说:', '').replace('###', '')
                                item = {'prompt': prompt, 'response': response}
                                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                out_f.flush()
                                histories.append(ro)
                        else:
                            histories.append(ro)


data_dir = "/root/zhucanxiang/data/APA_ORIGIN_DATA/apa_origin_data_translate_v2"
apa_data_preprocess = APADataPreprocess()
apa_data_preprocess.generate_dataset_v2(data_dir)
