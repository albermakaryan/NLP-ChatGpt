import torch
import torch.nn as nn
from torch.nn import functional as F
from icecream import ic

# read data

with open("../DATA/input.txt",'r') as file:
    text = file.read()
    
    
    
# hyperparameters
batch_size = 2
block_size = 10
max_iters = 3000 # how many iterations to train for
evan_interval = 300 # interval to evaluate the model performance
learning_rate = 0.0001 
device = "cuda" if torch.cuda.is_available() else "cpu" # use gpu if available
eval_iters = 200


torch.manual_seed(123)

# tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from char to index and vice versa
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


# create a dataloader
def get_batch(split,block_size,batch_size):
    
    data = train_set if split == "train" else val_set
    
    # randomly sample a starting index by batch size
    start_index = torch.randint(len(data)-block_size,(batch_size,))
    # genearte x and y batch_size times, where a single batch
    # contains blosk_size list of characters' indices
    
    x = torch.stack([data[i:i+block_size] for i in start_index])
    y = torch.stack([data[i+1:i+block_size+1] for i in start_index])
    
    x = x.to(device)
    y = y.to(device)
    
    return x,y


# estimate loss

@torch.no_grad()
def estimate_loss(model):
    
    out = {}
    model.eval()
    
    for split in ['train','validation']:
        
        # initialize loss tesnor to store loss for each batch
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters): # use eval_iters batches to estimate loss for each split
            
            X,Y = get_batch(split,block_size,batch_size)
            _,loss = model(X,Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
        
    
    model.train()
    return out


# bigram model

class BigramLanguageModel(nn.Module):
    
    
    def __init__(self,vocab_size):
        
        super().__init__()
        
        
        # initialize embedding lookup table
        self.token_embedding = nn.Embedding(vocab_size,vocab_size)
        
        
    def forward(self,idx,targets=None):
        
        
        logits = self.token_embedding(idx)
        
        # print("Shapes before")
        # ic(logits.shape)
        # ic(idx.shape)
        # ic(targets.shape)        
    
        
        
        # compute loss
        if targets is None:
            
            loss = None
        else:
            
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            
            # print("Shape after")
            # ic(logits.shape)
            # ic(targets.shape)
            # ic(targets)
            # ic(logits[0])
            loss = F.cross_entropy(logits,targets)
            
        return logits,loss
        
    
    
    def generate(self,idx,max_new_tokens):
        
        # idx is (B,T) array of indecies in the current context
        
        for _ in range(max_new_tokens): # generate max_new_tokens number of tokens
            
            # get the prediction
            logits,_ = self.forward(idx)
            # focus only on the last time step
            logits = logits[:,-1,:]
            # apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
            
        return idx
            
            
        
model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and validation sets
    if iter%eval_iters == 0:
        
        losses = estimate_loss(model)
        print(f"At iteration {iter}/{max_iters}, Train loss: {losses['train']}, Validation loss: {losses['validation']}")

    # get a batch of data
    x,y = get_batch("train",block_size,batch_size)
    
    # evaluate loss and update parameters
    lotigs, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



while True:
    context = input("\nEnter a context: ")
    
    if context == 'quit':
        break
    context = encode(context).reshape(-1,1)
    
    context = context.to(device)
    
    response = m.generate(context,10)[0].tolist()
    
    
    # print([index_to_char[i] for i in response])
    response = "".join(decode(response))
    print(response)

