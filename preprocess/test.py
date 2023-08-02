import unittest
import os
from baidu_trans import BaiduTrans
from apa_data_preprocess import APADataPreprocess

class TestBaiduTranslate(unittest.TestCase):
    def test_add(self):
        appid = os.getenv("BAIDU_TRANSLATE_APPID")
        appkey = os.getenv("BAIDU_TRANSLATE_APPKEY")
        baidu_trans = BaiduTrans(appid, appkey)
        result = baidu_trans.translate("hello", "zh")
        print(result)
        self.assertEqual(result, "你好")

class TestAPADataPreprocess(unittest.TestCase):
    def test_generate_dataset(self):
        data_dir = '/root/zhucanxiang/data/APA_ORIGIN_DATA/apa_origin_data_translate_v2_sample'
        apa_data_preprocess = APADataPreprocess()
        apa_data_preprocess.generate_dataset(data_dir)

if __name__ == '__main__':
    unittest.main()
