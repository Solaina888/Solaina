import numpy as np
import pickle

# 定义激活函数（使用ReLU函数）
def relu(x):
    return np.maximum(0, x)

# 定义softmax函数
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # 用于数值稳定性
    return exps / np.sum(exps, axis=1, keepdims=True)

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    # 前向传播
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    # 反向传播
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # 计算梯度
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m

        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    # 训练函数
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # 前向传播
            output = self.forward(X)
            # 反向传播
            self.backward(X, y, learning_rate)
            # 打印损失值
            loss = self.calculate_loss(output, y)
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # 损失函数（交叉熵损失）
    def calculate_loss(self, output, y):
        m = y.shape[0]
        log_probs = -np.log(output[np.arange(m), np.argmax(y, axis=1)])
        loss = np.sum(log_probs) / m
        return loss

    # 预测函数
    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

# 加载MNIST数据集（示例数据）
def load_data():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_test = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    return X_train, y_train, X_test, y_test


# 创建神经网络对象
input_size = 2
hidden_size = 4
output_size = 2
learning_rate = 0.1
num_epochs = 500

nn = NeuralNetwork(input_size, hidden_size, output_size)

# 加载数据集
X_train, y_train, X_test, y_test = load_data()

# 训练神经网络
nn.train(X_train, y_train, num_epochs, learning_rate)

# 保存模型
model_filename = "neural_network_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(nn, file)
print(f"模型已保存到文件: {model_filename}")
