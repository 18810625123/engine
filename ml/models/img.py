from . import *

class Img(models.Model):
    label = models.ForeignKey(ImgLabel,on_delete=models.CASCADE)
    url = models.CharField(max_length=2048)
    md5 = models.CharField(max_length=255, unique=True, blank=False, null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'img'

    def to_json(self):
        return {
            'id': self.id,
            'label_name': (self.label.name if self.label else ''),
            'label_id': self.label_id,
            'url': self.url,
            'md5': self.md5,
            'created_at': self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': self.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


    def read_url(self):
        # 从网络读图片
        response = req.get(self.url)
        # 获取图片2进制数据
        self.bytes = BytesIO(response.content).getvalue()
        # 2进制数据转numpy
        self.image_arr = np.frombuffer(self.bytes, dtype=np.uint8)
        # numpy转图片
        self.image = cv2.imdecode(self.image_arr, cv2.COLOR_RGBA2BGR)
        self.md5 = hashlib.md5(self.image).hexdigest()

    def resize(self, size):
        self.resize_img = cv2.resize(self.image, size, interpolation=cv2.INTER_CUBIC)

    # 转灰度图
    def gray(self):
        self.gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    # filter2D
    def filter2d(self, kernel=None):
        self.filter2d_img = cv2.filter2D(self.gray_img, -1, np.array(kernel))

    # filter2D
    def threshold(self, min=100, max=255):
        ret, image = cv2.threshold(self.filter2d_img, min, max, cv2.THRESH_BINARY)
        self.threshold_img = image

    # 高斯模糊
    def gauss(self, x=9, y=9):
        self.gauss_img = cv2.GaussianBlur(self.filter2d_img, (x, y), 0)

    # 边缘检测
    def canny(self, min=120, max=240):
        self.canny_img = cv2.Canny(self.gray_img, min, max)

    # 数组转base64
    def np_to_base64(self):
        image_base64_bytes = base64.b64encode(self.image_arr)
        return "data:image/jpeg;base64," + str(image_base64_bytes, encoding='utf-8')

    # 图片转base64
    def to_base64(self, img):
        new_img = cv2.imencode('.jpg', img)[1]
        return "data:image/jpeg;base64," + str(base64.b64encode(new_img))[2:-1]

    def show_base64(base64str):
        img = base64.b64decode(base64str)
        nparr = np.fromstring(img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("test", img_np)
        cv2.waitKey(0)

    def save_img(self):
        cv2.imwrite('static/images/%s.jpg'%(self.md5), self.image)
