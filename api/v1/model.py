from .helpers import *

class ModelApi():
    def create(request):
        try:
            params = get_params(request)
            model = Model(name=params['name'], remake=params['remake'])
            model.save()
            return custom_success(model.to_json())
        except Exception as e:
            traceback.print_exc()
            return custom_error(1, str(e))

    def search(request):
        params = get_params(request)
        models = Model.objects.all()
        total = models.count()
        if params.get('category'): models = Model.objects.filter(category=params['category'])
        if params.get('name'): models = Model.objects.filter(name=params['name'])
        return custom_success({'total': total, 'list': pg(models, params).to_json()})

    def remove(request):
        params = get_params(request)
        model = Model.objects.filter(id=params['id'])
        model.delete()
        return custom_success()

    def detail(request):
        params = get_params(request)
        model = Model.objects.get(id=params['id'])
        if not model: return custom_error(1, '没有找到这个模型：%s' % params['id'])
        return custom_success(model.to_json())

    def train(request):
        params = get_params(request)
        model = Model.objects.get(id=params['id'])
        if not model: return custom_error(1, '没有找到这个模型：%s' % params['id'])
        model.train()
        return custom_success()

    def predict(request):
        params = get_params(request)
        model = Model.objects.get(id=params['id'])
        if not model: return custom_error(1, '没有找到这个模型：%s' % params['id'])
        img = Img(url=params['img_url'])
        img.read_url()
        return custom_success(model.predict(img))

    def gen_vgg16(request):
        seed = 7
        np.random.seed(seed)
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(500, 500, 3), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                       validation_split=0.1, verbose=1)
        path = 'models/%s.h5' % (m.id)
        model.save(path)
        size = os.path.getsize(path)
