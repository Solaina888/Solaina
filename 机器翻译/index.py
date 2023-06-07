import torch
import torch.nn as nn
import torch.optim as optim

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        encoder_output, encoder_hidden = self.encoder(input)
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        loss = 0

        for i in range(target.size(1)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target[:, i])
            decoder_input = target[:, i].unsqueeze(1)

        return loss

# 定义训练函数
def train(input_tensor, target_tensor, encoder, decoder, seq2seq, criterion, optimizer):
    optimizer.zero_grad()
    loss = 0

    input_tensor = input_tensor.unsqueeze(1)
    target_tensor = target_tensor.unsqueeze(1)

    loss = seq2seq(input_tensor, target_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()

# 参数
input_size = len(train_data[0][0])  # 根据train_data确定input_size
output_size = len(train_data[0][1])  # 根据train_data确定output_size
hidden_size = 256
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建编码器、解码器和Seq2Seq模型
encoder = Encoder(input_size, hidden_size).to(device)
decoder = Decoder(hidden_size, output_size).to(device)
seq2seq = Seq2Seq(encoder, decoder).to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.SGD(seq2seq.parameters(), lr=learning_rate)

# 定义训练数据集
SOS_token = 1  # 句子起始标记的索引
train_set = [
    ("hello", "你好"),
    ("world", "世界"),
    ("good morning", "早上好"),
    # ...
]

for input_sentence, target_sentence in train_set:
    input_indexes = [word2index[word] for word in input_sentence.split()]
    target_indexes = [word2index[word] for word in target_sentence.split()]
    train_data.append((torch.tensor(input_indexes), torch.tensor(target_indexes)))

# 训练模型
num_epochs = 10
train_data = []

for epoch in range(1, num_epochs + 1):
    total_loss = 0

    for input_tensor, target_tensor in train_data:
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        loss = train(input_tensor, target_tensor, encoder, decoder, seq2seq, criterion, optimizer)
        total_loss += loss

    avg_loss = total_loss / len(train_data)
    print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
