#coding=utf-8

import os
import json
import re
import requests
import hashlib
import time
from urllib.parse import urlencode


class BaiduTrans:
    def __init__(self, appid, appkey):
        self.appid = appid
        self.appkey = appkey

    def translate(self, text, to_language):
        base_url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        salt = str(int(time.time()))
        sign_str = self.appid+text+salt+self.appkey
        sign = hashlib.md5(sign_str.encode(encoding="UTF-8")).hexdigest()
        base_url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        params = {"q": text, "from": "auto", "to": to_language, "appid": self.appid, "salt": salt, "sign": sign}
        url = base_url + "?" + urlencode(params)

        try:
            response = requests.request("GET", url, headers={}, data={})
            if response and response.text:
                response_json = json.loads(response.text)
                if response_json and "trans_result" in response_json and len(response_json['trans_result'])>0 and 'dst' in response_json['trans_result'][0]:
                    return response_json['trans_result'][0]['dst']
                else:
                    return text
            else:
                return text
        except:
            return text

