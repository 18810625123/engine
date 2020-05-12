class Snake:
    def __init__(self, map, name, length, direction):
        self.map = map
        self.name = name
        self.length = length
        self.direction = direction
        self.arr = []
        x = int(self.map.x / 3)
        y = int(self.map.y / 3)
        for i in range(length):
            if i == 0:
                self.arr.append('%d:%d' % (x, y))
            else:
                self.arr.append('%d:%d' % (x, y))

    def controller(self, action):
        self.action = action

    def judge(self):
        self.map
        print(1)


class Map:
    def __init__(self, x, y):
        self.arrs = []
        self.style = {
            'food': '#',
            'snake': '*',
            'map': '-',
        }
        self.x = x
        self.y = y
        self.snakes = {}
        self.foods = []

    def add_food(self):
        self.foods.append()

    def add_snake(self, name, length = 2, direction = 'down'):
        self.snakes[name] = Snake(self, name, length, direction)

    def draw(self):
        for x in range(self.x):
            arr = []
            for y in range(self.y):
                arr.append(self.style['map'])
            self.arrs.append(arr)

    def look(self):
        for arr in self.arrs:
            for a in arr:
                print(a, end='')
            print()


class Draw:
    # def __init__(self):

    @classmethod
    def asdf(cls):
        print(372)


def start():
    m = Map(50, 50)
    m.init_snake('a')
    m.init_snake('b')


Draw.asdf()
# i e
# n s
# t f
# p j

# S 排除 无法沟通
# P 排除 无法督促成长
# I 排除 话太少

# ENFJ ENTJ ENTP ENFP
# 总结：
# 不要太I S T P J E
