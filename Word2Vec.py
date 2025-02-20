# Word2Vec的Skip-gram模型
# 超参数:embedding_dim、window_size、batch_size
# SkipGramModel: 这是Skip-gram模型的定义，包含一个嵌入层和一个线性层。
# Word2VecDataset: 这是一个自定义的数据集类，用于生成训练数据。
# 数据准备: 我们使用一个简单的句子作为训练数据，并将其转换为词汇表和索引。
# 训练: 使用交叉熵损失函数和Adam优化器来训练模型。
# 词向量: 训练完成后，我们可以从嵌入层中获取词向量。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义Skip-gram模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

# 自定义数据集类
class Word2VecDataset(Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, window_size):
        self.text = text
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for i, target in enumerate(self.text):
            for j in range(max(0, i - self.window_size), min(len(self.text), i + self.window_size + 1)):
                if j != i:
                    context = self.text[j]
                    data.append((self.word_to_idx[target], self.word_to_idx[context]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 准备数据
text = "I like natural language processing and deep learning".split()
vocab = set(text)
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# 超参数
embedding_dim = 10
window_size = 2
batch_size = 2
learning_rate = 0.001
num_epochs = 10

# 创建数据集和数据加载器
dataset = Word2VecDataset(text, word_to_idx, idx_to_word, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = SkipGramModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for target, context in dataloader:
        optimizer.zero_grad()
        output = model(target)
        loss = criterion(output, context)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# 获取词向量
word_vectors = model.embeddings.weight.data.numpy()
for i, word in enumerate(vocab):
    print(f"{word}: {word_vectors[i]}")