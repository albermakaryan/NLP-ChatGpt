import torch
import torch.nn as nn
from torch.nn import functional as F    
from icecream import ic


# hyperparameters
batch_size = 32
block_size = 256
max_iters = 10000
eval_intervals = 500
learning_rate = 3e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 256
n_head = 8
n_layer = 8
droupout = 0.3

model_path = "../models/gpt.pt"
model_path = "../models/airbnb_chat_bot.pt"
model_path = "../models/go_travel_articles_bot.pt"

data_path = '../DATA/go_world_travel_article.txt'
# ------------------- #

# read data
with open(data_path,'r',encoding='utf-8') as file:
    
    text = file.read()
    
    
print(len(text))
    
    
# tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_index = {c:i for i,c in enumerate(chars)}
index_to_char = {i:c for i,c in enumerate(chars)}

# create an encoder and decoder functions

encode = lambda x: torch.tensor([char_to_index[c] for c in x])
decode = lambda x: [index_to_char[i] for i in x]


# split data into train and validation sets
data = encode(text)
train_size = int(0.9*len(data))
train_set = data[:train_size]
val_set = data[train_size:]

# print(chars)
# quit()

# create a dataloader

def get_batch(split):
    
    
    data = train_set if split == "train" else val_set
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    
    return x,y


# model estimation
@torch.no_grad()
def esimate_loss(model):
    
    model.eval()
    out = {}
    
    for split in ['train','val']:
        
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            _,loss = model(x,y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    
    return out


# class head

class Head(nn.Module):
    
    def __init__(self,head_size):
        
        super(Head, self).__init__()  
              
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.droupout = nn.Dropout(droupout)
        
    def forward(self,x):
        
        # input size  (batch_size,time-step,channles)
        # output size (batch_size,time-step,head_size)
        
        B,T,C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores ("affinities")
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.droupout(wei)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out
    
    
# multi-head attention
class MultiHeadAttention(nn.Module):
    
    def __init__(self,n_head,head_size):
        
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head*head_size,n_embd)
        self.dropout = nn.Dropout(droupout)
        
    
        
        
        
    def forward(self,x):
        
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    
# feed forward layer
class FeedForward(nn.Module):
    
    def __init__(self,n_embd):
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(droupout)
            )
        
    def forward(self,x):
        
        return self.net(x)
    
    
    
# transformer block
class Block(nn.Module):
    
    def __init__(self,n_embd,n_head):
        
        super().__init__()
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x
    
    
# transformer model
class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,std=0.02,mean=0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,std=0.02,mean=0)
            
    def forward(self,idx,targets=None):
        
        B,T = idx.shape
        
        # idx and targets have shape (batch_size,time_step)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
            
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss
    
    
    def generate(self,idx,max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:,-block_size:]
            
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx,idx_next],dim=-1)
            
        return idx
    
    


context = torch.zeros((1, 1), dtype=torch.long, device=device)



    
model = GPTLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6,'M parameters')


# create a torch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)


for iter in range(max_iters):
    

    
    if iter % eval_intervals == 0:
        
        losses = esimate_loss(m)
        print(f"iter {iter} | train loss {losses['train']} | val loss {losses['val']}")
        
    
    x,y = get_batch("train")
    logits,loss = m(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
    
torch.save(m,model_path)



model = torch.load(model_path)


while True:
    
    context = input("Enter a prompt: ")
    
    if context == "exit":
        break
    
    context = encode(context).reshape(-1,1).to(device)
    
    max_tokens = int(input("Enter max tokens: "))
    
    
    print("".join(decode(model.generate(context, max_new_tokens=max_tokens)[0].tolist())))
    print()

    
    
    
quit()
    
    