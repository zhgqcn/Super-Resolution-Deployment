# 连接GPU服务器

import requests, time, os, logging


class SRApi():
    def __init__(self, url = 'ip address', path = 'image path', save = './result/'):
        super(SRApi, self).__init__()
        self.url = url
        self.path = path
        self.filename = os.path.basename(self.path)
        self.save_path = save
        self.session = requests.session()

    def requests_post(self):
        url = self.url + 'get_result'
        files = {'image': open(self.path,'rb'),
                 'filename': self.filename}
        feed_back = "b''"
        try:
            requests.adapters.DEFAULT_RETRIES = 2
            feed_back = self.session.post(url, files = files, timeout = 20).content
            feed_back = str(feed_back)
        except requests.exceptions.RequestException as e:
            logging.info('\n  连接超时 \n')
            print(e)

        if '400' not in feed_back and feed_back != "b''":
            print('[Info LOG]{}图片上传成功！\n 超分辨计算中,请稍等！'.format(
                        time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
            logging.info('\n 图片上传成功，超分计算中，请等待！ \n')
            return True
        else:
            print('[Info LOG]{}图片上传失败,返回信息:{}！'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),feed_back))
            return False

    def request_download(self):
        try:
            requests.adapters.DEFAULT_RETRIES = 2
            r1 = self.session.get(self.url + 'static/output/X2/X2_' + self.filename + '.png', timeout= 8)
            if str(r1.status_code) == '200':
                save1 = self.save_path + 'SR_X2/' + self.filename + '.png'
                with open(save1, 'wb') as f1:
                    f1.write(r1.content)
            else:
                logging.info('服务器请求超时！')
                return False
        except requests.exceptions.RequestException as e:
            logging.info('\n 图片下载超时 \n')
            print(e)
            return False

        try:
            requests.adapters.DEFAULT_RETRIES = 2
            r2 = self.session.get(self.url + 'static/output/X4/X4_' + self.filename + '.png', timeout = 16)
            if str(r2.status_code) == '200':
                save2 = self.save_path + 'SR_X4/' + self.filename + '.png'
                with open(save2, 'wb') as f2:
                    f2.write(r2.content)
            else:
                logging.info('服务器请求超时！')
                return False
        except requests.exceptions.RequestException as e:
            logging.info('\n 图片下载超时 \n')
            print(e)
            return False

        return True

    def main(self):
        flag = self.requests_post()
        if flag:
            if self.request_download():
                print('[Info LOG]{}图片下载成功！保存至：\n {} \n {}'.format(
                        time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),
                        self.save_path + 'SR_X2/' + self.filename + '.png',
                        self.save_path + 'SR_X4/' + self.filename + '.png'
                ))
                return True
        return False


if __name__ == '__main__':
    flag = SRApi(url = 'http://ba478c6af42b.ngrok.io/', path = '../ui/test_4.tif', save = '../result/').main()




