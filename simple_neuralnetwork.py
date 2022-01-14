import numpy as np

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true和y_pred是相同长度的numpy数组。
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:


    def __init__(self):
        # 权重，Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()

        # 截距项，Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # X是一个有2个元素的数字数组。
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        o1 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):

        learn_rate = 0.00001
        epochs = 100000  # 遍历整个数据集的次数

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- 做一个前馈(稍后我们将需要这些值)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w7 * h1 + self.w8 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- 计算偏导数。
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_l_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w7 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w8 * deriv_sigmoid(sum_o1)


                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                # --- 更新权重和偏差
                # Neuron h1
                self.w1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.b1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w4 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w6 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.b2 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w7 -= learn_rate * d_l_d_ypred * d_ypred_d_w7
                self.w8 -= learn_rate * d_l_d_ypred * d_ypred_d_w8
                self.b3 -= learn_rate * d_l_d_ypred * d_ypred_d_b3

            # --- 在每次epoch结束时计算总损失
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss,))


# 定义数据集
data = np.array([
    [70, 170, 1],
    [47, 165, 10],
    [45, 158, 10],
    [65, 185, 1],
    [77, 180, 1],
    [52, 172, 10],
    [65, 175, 1],
    [80, 182, 1],
    [44, 155, 10],
    [60, 168, 10]
])
all_y_trues = np.array([
    0.549,
    0.197,
    0.156,
    0.801,
    0.822,
    0.411,
    0.822,
    1,
    0,
    0.199,
])

# 训练我们的神经网络!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# 做一些预测
frank = np.array([70, 170, 1])
emily = np.array([47, 165, 0])
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
