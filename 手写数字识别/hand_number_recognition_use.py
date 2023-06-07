import numpy as np
from PIL import Image

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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions


# 加载MNIST数据集（示例数据）
def load_data():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    return X_train, y_train


# 创建神经网络对象
input_size = 2
hidden_size = 4
output_size = 2

nn = NeuralNetwork(input_size, hidden_size, output_size)

# 加载数据集
X_train, y_train = load_data()


# 前向传播
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = self.sigmoid(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self.sigmoid(self.z2)
    return self.a2


# 后向传播
def backward(self, X, y, learning_rate):
    m = X.shape[0]  # 样本数量

    delta2 = self.a2 - y
    delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1)

    dW2 = np.dot(self.a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0) / m
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0) / m

    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1


# 预测用户自己的图片
def predict_user_image(image_path):
    # 加载用户图片并进行预处理
    user_image = preprocess_image(image_path)
    user_image = np.expand_dims(user_image, axis=0)  # 扩展维度以匹配输入

    # 执行预测
    prediction = nn.predict(user_image)
    return prediction


# 用户自己的图片路径
user_image_path = "微信图片_20230604132352.jpg"


# 预处理用户图片
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((input_size, input_size))  # 调整为与输入尺寸相同
    image = image.convert("L")  # 转换为灰度图像
    image = np.array(image) / 255.0  # 归一化像素值
    image = image.flatten()  # 转换为一维数组
    return image


# 预测用户图片
prediction = predict_user_image(user_image_path)
print("预测结果：", prediction)
