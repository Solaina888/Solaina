import os
import re
import numpy as np
import matplotlib as plt
# 文件夹路径
ham_emails = '20021010_easy_ham/easy_ham'
spam_emails = '20021010_spam/spam'
ham_emails=[]
spam_emails=[]
# 假设你已经有了垃圾邮件和非垃圾邮件的列表，分别命名为spam_emails和ham_emails
# 读取正常邮件
for ham in range(100):
	for filename in os.listdir(ham_emails):
		with open(os.path.join(ham_emails, filename), 'r', encoding='utf-8', errors='ignore') as f:
			ham_emails.append(f.read())
# 读取垃圾邮件
for spam in range(100):
	for filename in os.listdir(spam_emails):
		with open(os.path.join(spam_emails, filename), 'r', encoding='utf-8', errors='ignore') as f:
			spam_emails.append(f.read())
# 搭建
# 数据清理：去除特殊字符、标点符号和HTML标签
def clean_text(text):
    # 去除HTML标签
    cleaned_text = re.sub('<.*?>', '', text)
    # 去除特殊字符和标点符号
    cleaned_text = re.sub('[^\w\s]', '', cleaned_text)
    return cleaned_text
# 对垃圾邮件进行清理和预处理
spam_emails = []
spam_dir = "spam"
spam_files = os.listdir(spam_dir)
for filename in spam_files:
    with open(os.path.join(spam_dir, filename), "r", encoding="latin-1") as f:
        email = f.read()
        cleaned_email = clean_text(email)
        spam_emails.append(cleaned_email)

# 对非垃圾邮件进行清理和预处理
ham_emails = []
ham_dir = "easy_ham"
ham_files = os.listdir(ham_dir)
for filename in ham_files:
    with open(os.path.join(ham_dir, filename), "r", encoding="latin-1") as f:
        email = f.read()
        cleaned_email = clean_text(email)
        ham_emails.append(cleaned_email)

# 合并垃圾邮件和非垃圾邮件的数据和标签
X = np.array(spam_emails + ham_emails)
y = np.concatenate((np.ones(len(spam_emails)), np.zeros(len(ham_emails))), axis=0)

# 数据集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征提取
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 构建神经网络模型
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=100)

# 模型训练
model.fit(X_train_counts, y_train)
usersmail = input("请输入待检测邮件：")
# 模型预测
y_pred = model.predict(usersmail)

# 模型性能评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)