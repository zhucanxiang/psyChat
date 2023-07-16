import os, sys
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import time
import copy


import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments

model_name_or_path = "/root/zhucanxiang/model/chatglm2-6b"
#ptuning_checkpoint = "output/apa-chatglm-6b-pt-128-2e-2/checkpoint-3000"
ptuning_checkpoint = "/root/zhucanxiang/ChatGLM2-6B/ptuning/output/apa-chatglm2-6b-pt-128-2e-2/checkpoint-1000"
pre_seq_len=128


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static") # 挂载静态文件，指定目录
templates = Jinja2Templates(directory="templates") # 模板目录

model = None
tokenizer = None

history_dir = "history/"


def predict(prompt, max_length, top_p, temperature, history):
    response, history = model.chat(tokenizer, prompt, history, max_length=max_length, top_p=top_p, temperature=temperature)
    print("prompt: {}, max_length: {}, top_p: {}, temperature: {}".format(prompt, str(max_length), str(top_p), str(temperature)))
    print("response: " + response)
    return response, history


def load_model(model_name_or_path, ptuning_checkpoint, pre_seq_len):
    print('loading model')
    start = time.time()
    global model, tokenizer

    parser = HfArgumentParser((
        ModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
    model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    print(f"Quantized to {model_args.quantization_bit} bit")
    quantization_bit = 4
    model = model.quantize(quantization_bit)

    # P-tuning v2
    model = model.half().cuda()
    model.transformer.prefix_encoder.float().cuda()
    
    model = model.eval()
    print('finish load model. load model time cost: {}'.format(time.time()-start))


load_model(model_name_or_path, ptuning_checkpoint, pre_seq_len)

def write_chat_history(username, patient_say, doctor_say):
    filename = history_dir + username + '.txt'
    with open(filename, "a+", encoding='utf-8') as f:
        patient_say = '病人:' + patient_say
        doctor_say = '医生:' + doctor_say
        f.write(patient_say + '\n')
        f.write(doctor_say + '\n')

    chat_history_str = ""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            chat_history_str += line
    return chat_history_str

def read_chat_history(username):
    filename = history_dir + username + '.txt'
    chat_histories = []
    with open(filename, "r", encoding='utf-8') as f:
        chat_item = {}
        for line in f.readlines():
            if line.startswith('医生:'):
                chat_item['doctor'] = line.replace('医生:', '')
                new_chat_item = copy.deepcopy(chat_item)
                print(chat_item['doctor'])
                chat_histories.append(new_chat_item)
                chat_item = {}
            else:
                chat_item['patient'] = line.replace('病人:', '')
        if 'doctor' in chat_item or 'patient' in chat_item:
            chat_histories.append(chat_item)
    return chat_histories


def load_chat_history(username):
    filename = history_dir + username + '.txt'
    is_new_user = False
    chat_history_str = ""
    if os.path.exists(filename):
        #老用户，则读取聊天记录
        is_new_user = False
        with open(filename, "r", encoding='utf-8') as fr:
            for line in fr.readlines():
                chat_history_str += line
    else:
        # 新用户，则创建聊天记录文件
        is_new_user = True
        with open(filename, "w", encoding='utf-8') as fw:
            doctor_say = '医生:你好, {}, 我是王医生'.format(username) + '\n'
            fw.write(doctor_say)
            chat_history_str =  doctor_say
            fw.flush()
            fw.close()
    return is_new_user, chat_history_str

@app.get("/home/{username}")
def home(request:Request, username:str):
    new_user, chat_history_str = load_chat_history(username)
    return templates.TemplateResponse("index_h.html", {"request": request, "username": username, "chat_history": chat_history_str})

class PredictData(BaseModel):
    query: str       # 输入
    username: str    # username
    #max_length: Optional[int] = 4096 #最大长度
    #top_p:  Optional[float] = 0.7
    #temperature: Optional[float] = 0.95

class ClearHistoryData(BaseModel):
    username: str    # username

@app.post("/predict")
def chat(request: PredictData):
    max_length = 4096
    top_p = 0.7
    temperature = 0.95
    chat_histories = read_chat_history(request.username)
    histories = []
    history_str = ""
    if chat_histories:
        last_ten_chat_histories = chat_histories[-10:]
        #last_ten_chat_histories.reverse()
        for chat_history in last_ten_chat_histories:
            patient_say = ''
            doctor_say = ''
            if 'patient' in chat_history:
                patient_say = "" + chat_history['patient'].strip()
                histories.append(patient_say)
            if 'doctor' in chat_history:
                doctor_say = chat_history['doctor'].strip()
                histories.append(doctor_say)
    if len(histories) > 0:
        history_str = '\n'.join(histories)
    prompt_template = "假设你是一名心理咨询师，你来帮助来访者解决心理问题。来访者说: ###{}###。 来访者说的内容被左右分别一个###符号包围。请你对来访者做出回应，回应要体现对来访者的关切和对来访者有启发。以下是对话的上文信息供参考>。对话的上文信息格式如下：```{}```。上下文信息被```被左右分别一个```符号包围，上下文信息中的心理咨询师和来访者说的内容被左右分别一个###包围。"
    prompt = prompt_template.format(request.query, history_str)
    response, history = predict(prompt, max_length, top_p, temperature, [])
    chat_history_str = write_chat_history(request.username, request.query, response)
    print('prompt:' + prompt)
    return_data = {'response': response, "chat_history": chat_history_str}
    return return_data

@app.post("/clear_history")
def clear_history(request:ClearHistoryData):
    username = request.username
    response = "成功清空{}的历史聊天记录".format(username)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="*", port=8082)
