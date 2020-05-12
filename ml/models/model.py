from . import *

class Model(models.Model):
    name = models.CharField(max_length=255, unique=True, blank=False)
    remake = models.CharField(max_length=1000, null=False)
    size = models.CharField(max_length=50, default='')
    train_x = models.IntegerField(max_length=20, default=0)
    test_x = models.IntegerField(max_length=20, default=0)
    category = models.CharField(max_length=50, default='')
    input_shape = models.CharField(max_length=30, default='')
    y = models.CharField(max_length=1000, default=0)
    epoch = models.IntegerField(max_length=5, default=0)
    batch_size = models.IntegerField(max_length=3, default=0)
    acc = models.FloatField(max_length=10, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'model'

    def to_json(self):
        return {
            'id': self.id,
            'name': self.name,
            'train_x': self.train_x,
            'test_x': self.test_x,
            'y': self.y,
            'epoch': self.epoch,
            'category': self.category,
            'input_shape': self.input_shape,
            'batch_size': self.batch_size,
            'size': self.size,
            'remake': self.remake,
            'created_at': self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': self.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def load_model(self):
        self.model = tf.keras.models.load_model(
            self.get_path(), custom_objects=None, compile=True
        )

    def predict(self, img):
        self.load_model()
        img.resize((224, 224))
        X = np.array([img.resize_img])
        r = self.model.predict(X)[0]
        max_index = np.argmax(r)
        return {
            'similarity': str(round(r[max_index], 3)),
            'label': str(max_index),
            'old_img': {
                'shape': img.image.shape
            },
            'new_img': {
                'shape': img.resize_img.shape,
                'base64': img.to_base64(img.resize_img),
            }
        }
        arrs = []
        for i in range(r.__len__()):
            arr = []
            r_ = np.array([a for a in r[i]])
            for j in range(3):
                index = np.argmax(r_)
                value = r_[index]
                if j == 0 and value < 0.6:
                    arr.append('<0.6')
                    break
                if j > 0 and value < 0.2:
                    continue
                arr.append({
                    'label': str(index),
                    'similarity': str(round(value, 3))
                })
                r_ = np.delete(r_, index)
            arrs.append(arr)
        return arrs

    def train(self, X):
        self.model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                  validation_split=0.1, verbose=1)

    def read_X(self):
        models = os.listdir('./%s/images/%s' % (self.category, self.test_data))
        imgs = []
        self.y = []
        for model_file_name in models:
            self.y.append(model_file_name.split('.')[0])
            print('../%s/images/%s/%s' % (self.category, self.test_data, model_file_name))

            img = Img('../%s/images/%s/%s' % (self.category, self.test_data, model_file_name))
            imgs.append(img.data())
        self.X = np.array(imgs, dtype=np.float32)

