import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(42)

STATE_DIM = 16
SEQ_DIM = 8
SEQ_LEN = 5
POS_EMB_DIM = 8
BATCH_SIZE = 64
# 反例：不同位置的任务底层特征完全不相关
# 例如：位置0处理图像、位置1处理文本、位置2处理音频
# 此时共享LSTM会严重损害性能

class IndependentModel(nn.Module):
    """
    独立模型：为每个位置（地主/农民）单独训练的模型
    结构：LSTM编码序列 + 全连接层预测
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(SEQ_DIM, 16, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(16+STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, seq, state):
        out, _ = self.lstm(seq)
        return self.fc(torch.cat([out[:, -1], state], dim=-1))

class UnifiedModel(nn.Module):
    """
    统一Emb模型：通过位置Embedding区分不同位置的共享模型
    结构：位置编码 + LSTM编码序列 + 全连接层预测
    """
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(3, POS_EMB_DIM)
        self.lstm = nn.LSTM(SEQ_DIM, 16, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(16+STATE_DIM+POS_EMB_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, pos_ids, seqs, states):
        # pos_ids: [3] 三个位置的ID [0,1,2]
        # seqs: [192, 5, 8] = (3位置 × 64batch, 序列长度, 特征维度)
        # states: [192, 16] = (3位置 × 64batch, 状态维度)
        
        # 扩展位置ID: [0,1,2] -> [0...0, 1...1, 2...2] (每个重复64次) shape: [192]
        pos_indices = torch.repeat_interleave(pos_ids, seqs.shape[0]//3)
        
        # LSTM编码: [192,5,8] -> [192,5,16] (所有时间步的隐状态)
        out, _ = self.lstm(seqs)
        
        # 提取最后一步: [192,5,16] -> [192,16] (只取每个序列的最终状态)
        last_step = out[:, -1]
        
        # 生成位置编码: [192] -> [192,8]
        pos_emb = self.pos_emb(pos_indices)
        
        # 特征融合: [192,16] + [192,16] + [192,8] = [192,40]
        features = torch.cat([last_step, states, pos_emb], dim=-1)
        
        # 预测: [192,40] -> [192,1]
        return self.fc(features)

def generate_data(batch=1, pos_id=0):
    """
    生成测试数据
    pos_id=0: 使用seq均值 + state和
    pos_id=1: 使用seq最大值 + state均值
    pos_id=2: 使用seq方差 + state方差
    """
    seq = torch.randn(batch, SEQ_LEN, SEQ_DIM)
    state = torch.randn(batch, STATE_DIM)
    if pos_id == 0:
        label = (seq.mean(dim=1).sum(dim=1) + state.sum(dim=1)).unsqueeze(-1) * 0.1
    elif pos_id == 1:
        label = (seq.max(dim=1).values.sum(dim=1) + state.mean(dim=1)).unsqueeze(-1) * 0.1
    else:
        label = (seq.var(dim=1).sum(dim=1) + state.var(dim=1)).unsqueeze(-1) * 0.1
    return seq, state, label + torch.randn(batch, 1) * 0.01

print("="*70)
print("[训练收敛过程对比] - 学习率: 5e-4, 训练步数: 20000")
print("="*70)
print(f"{'Step':>6} | {'独立模型':>10} | {'统一Emb':>10} | {'独立耗时':>8} | {'统一耗时':>8}")
print("-"*70)

# 初始化独立模型（3个独立的模型）
m1,m2,m3 = IndependentModel(),IndependentModel(),IndependentModel()
opt1,opt2,opt3 = optim.Adam(m1.parameters(),lr=5e-4),optim.Adam(m2.parameters(),lr=5e-4),optim.Adam(m3.parameters(),lr=5e-4)

# 初始化统一Emb模型（1个共享模型）
m_emb = UnifiedModel()
opt_emb = optim.Adam(m_emb.parameters(), lr=5e-4)

t_ind_total = 0
t_emb_total = 0

for step in range(0, 20001, 1000):
    # ========== 独立模型训练 ==========
    t0 = time.time()
    # 为3个位置分别生成数据
    s1,st1,y1 = generate_data(BATCH_SIZE,0)
    s2,st2,y2 = generate_data(BATCH_SIZE,1)
    s3,st3,y3 = generate_data(BATCH_SIZE,2)
    # 分别计算3个模型的loss
    l1 = (m1(s1,st1)-y1).pow(2).mean()
    l2 = (m2(s2,st2)-y2).pow(2).mean()
    l3 = (m3(s3,st3)-y3).pow(2).mean()
    loss_ind = (l1 + l2 + l3) / 3
    # 分别反向传播更新参数
    l1.backward(); opt1.step(); opt1.zero_grad()
    l2.backward(); opt2.step(); opt2.zero_grad()
    l3.backward(); opt3.step(); opt3.zero_grad()
    t_ind_total += time.time() - t0

    # ========== 统一Emb模型训练 ==========
    t1 = time.time()
    # 合并3个位置的数据，传入位置ID
    p1,p2,p3 = m_emb(torch.tensor([0,1,2]), torch.cat([s1,s2,s3]), torch.cat([st1,st2,st3])).chunk(3)
    l_emb1 = (p1-y1).pow(2).mean()
    l_emb2 = (p2-y2).pow(2).mean()
    l_emb3 = (p3-y3).pow(2).mean()
    loss_emb = (l_emb1 + l_emb2 + l_emb3) / 3
    loss_emb.backward()
    opt_emb.step(); opt_emb.zero_grad()
    t_emb_total += time.time() - t1

    print(f"{step:>6} | {loss_ind.item():>10.4f} | {loss_emb.item():>10.4f} | {t_ind_total:>7.3f}s | {t_emb_total:>7.3f}s")

print("-"*70)
print(f"[总结] 独立模型总耗时: {t_ind_total:.3f}s | 统一Emb总耗时: {t_emb_total:.3f}s | 加速比: {t_ind_total/t_emb_total:.2f}x")