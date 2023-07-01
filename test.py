import unittest
import os
from baidu_trans import BaiduTrans

class TestBaiduTranslate(unittest.TestCase):
    def test_add(self):
        appid = os.getenv("BAIDU_TRANSLATE_APPID")
        appkey = os.getenv("BAIDU_TRANSLATE_APPKEY")
        baidu_trans = BaiduTrans(appid, appkey)
        result = baidu_trans.translate("hello", "zh")
        self.assertEqual(result, "你好")

if __name__ == '__main__':
    unittest.main()
