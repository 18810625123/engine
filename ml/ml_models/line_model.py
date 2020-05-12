from .helpers import *

# 创建随机数据
def create_csv(path, max_int):
    arrs = []
    f = open(path, 'rw')
    f.truncate()
    f.write('x1,x1,y')
    for i in range(max_int):
        f.write([random.random()*max_int, random.random()*max_int, 0 if random.random()*max_int>50 else 1])
    return arrs


class LineModel():
    def __init__(self, csv_path, x_columns, y_column, show = True, shuffle = False, success_value = 1, failed_value = 0, learn_rate = 0.01, epochs = 10):
        self.sleep_time = 0.02
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.x_columns = x_columns
        self.y_column = y_column
        self.label_index = 2
        self.train_count = 0
        self.results = []
        self.shuffle = shuffle
        # 从文件读取csv数据

        self.data = pd.read_csv(csv_path, dtype=float)[self.x_columns + [self.y_column]]
        self.X = self.data[self.x_columns].values
        self.y = self.data[self.y_column].values
        self.SUM = self.X.__len__()
        self.Postive = self.data[self.y==success_value]
        self.P_count = self.Postive.__len__()
        self.Negative = self.data[self.y==failed_value]
        self.F_count = self.Postive.__len__()
        # 初始化模型参数
        self.x_min = self.data.min()['x1']
        self.x_max = self.data.max()['x1']
        self.y_min = self.data.min()['x2']
        self.y_max = self.data.max()['x2']
        self.W = [random.random(), random.random()]
        self.b = random.random()
        self.test(0)
        self.show = show
        if self.show: plt.ion()
        if self.show: plt.show()

    # 画散点
    def draw_points(self):
        plt.scatter(x=self.Postive['x1'],y=self.Postive['x2'],c='g')
        plt.scatter(x=self.Negative['x1'],y=self.Negative['x2'],c='r')

    # 画线
    def draw_line(self):
        b = self.b
        w1 = self.W[0]
        w2 = self.W[1]
        # x = [x_min*self.W[0], y_min]
        # y = [x_max*self.W[1], y_max]
        # w1 * x + w2 * y + b = 0
        point1_x = self.x_min
        point1_y = (-(w1 * point1_x) - b) / w2
        point1 = [point1_x, point1_y]

        point2_x = self.x_max
        point2_y = (-(w1 * point2_x) - b) / w2
        point2 = [point2_x, point2_y]

        plt.plot([point1[0], point2[0]], [point1[1], point2[1]])
        plt.pause(self.sleep_time)

    def alg(self, i):
        if (self.b + (self.X[i] * self.W).sum()) >= 0:
            return 1
        else:
            return 0

    def train(self, k = 1):
        for l in range(k):
            print('第%s批：' % l)
            if self.shuffle:
                data = self.data.sample(frac=1)
                self.X = data[self.x_columns].values
                self.y = data[self.y_column].values
            for j in range(self.epochs):
                for i in range(self.X.__len__()):
                    y_real = self.y[i]
                    y_predicted = self.alg(i)
                    if y_real - y_predicted == 1:
                        self.W[0] += self.X[i][0] * self.learn_rate
                        self.W[1] += self.X[i][1] * self.learn_rate
                        self.b += self.learn_rate
                    elif y_real - y_predicted == -1:
                        self.W[0] -= self.X[i][0] * self.learn_rate
                        self.W[1] -= self.X[i][1] * self.learn_rate
                        self.b -= self.learn_rate
                self.test(j+1)

    def test(self, j):
        self.draw_points()
        self.draw_line()
        result = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0,
        }
        for i in range(self.X.__len__()):
            y = self.y[i]
            y_ = self.alg(i)
            if y_ == 1:
                if y == 1:
                    result['TP'] += 1
                else:
                    result['FP'] += 1
            else:
                if y == 1:
                    result['FN'] += 1
                else:
                    result['TN'] += 1
        print(result)
        result['准确率'] = (result['TP'] + result['TN']) / self.SUM
        result['精确率'] = result['TP'] / (result['TP'] + result['FP'])
        result['召回率'] = result['TP'] / (result['TP'] + result['FN'])
        result['TPR'] = result['TP'] / (result['TP'] + result['FN'])
        result['FPR'] = result['FP'] / (result['FP'] + result['TN'])
        if (result['精确率'] + result['召回率']) != 0:
            result['F1'] = (result['精确率'] * result['召回率'] * 2) / (result['精确率'] + result['召回率'])
        else:
            result['F1'] = 0
        self.results.append(result)
        # print('第%s次: W=%s b=%s \n准确率:%.4f 精确率:%.4f \nTPR:%s FPR:%s F1:%s \n' % (j, self.W, self.b, result['准确率'], result['精确率'], result['FPR'], result['TPR'], result['F1']))
        if j == 0:
            print('\t开始 :准确率:%.4f 精确率:%.4f TPR:%s FPR:%.4f F1:%.4f' % (result['准确率'], result['精确率'], result['TPR'], result['FPR'], result['F1']))
        else:
            print('\t第%s次:准确率:%.4f 精确率:%.4f TPR:%s FPR:%.4f F1:%.4f' % (j, result['准确率'], result['精确率'], result['TPR'], result['FPR'], result['F1']))

    # 评估模型性能
    def evaluation(self):
        X = self

    def sigmoid(self, n):
        r = 1 / (1 + np.exp(-n))
        print(r)
        return r

class DataHandler():
    def __index__(self, data):
        self.data = data

    def k(self):
        X = self



# random.seed(2)
# arrs = create_arrs(80)
# liner_model = LinerModel(csv_path='../csv/1.csv', shuffle = True, x_columns = ['x1', 'x2'], y_column ='y', learn_rate = 0.01, epochs = 20)
# liner_model.draw_points()
# liner_model.train(k = 1)
# liner_model.evaluation()
# plt.show()
# d = SVC()
# d.fit(liner_model.X, liner_model.y)
# print(d)

# 画线 y = x**2
def y_x2(plt):
    step = 1
    x1 = -10
    for i in range(1, 20):
        x1 = x2
        x2 = x1 + step
        y1 = x1 ** 2
        y2 = x2 ** 2
        plt.plot([x1, x2], [y1, y2])
    plt.show()

# x = np.arange(2,10)
# y = x ** 2
# x = np.linspace(1,10,num=10)
