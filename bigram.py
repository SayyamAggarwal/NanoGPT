import torch
import torch.nn as nn
from torch.nn import functional as f

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embd = 32
# encoding part 
enc = {ch:i for i,ch in enumerate(chars)}
dec = {i:ch for i,ch in enumerate(chars)}

encode = lambda ch: [enc[char] for char in ch]
decode = lambda nums: ''.join([dec[num] for num in nums])


data = torch.tensor(encode(text),dtype = torch.long)

# traininng and validation
n = int(0.9*len(data))#train-size
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  data = train_data if split =='train' else val_data
  xi = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in xi])#block size of numbers
  y = torch.stack([data[i+1:i+block_size+1] for i in xi])#output lables
  x,y = x.to(device),y.to(device)
  return x,y


@torch.no_grad()
def est_loss():
   out = {}
   model.eval()
   for split in ['train','val']:
      losses = torch.zeros(eval_iters)

      for x in range(eval_iters):
         X,Y = get_batch(split)
         logits,loss = model(X,Y)
         losses[x] = loss.item()

      out[split] = losses.mean().item()
       
   model.train()
   return out



# Implemening single head attention mechanism

class Head(nn.Module):
  '''one head of attention'''
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias = False)
    self.query = nn.Linear(n_embd,head_size,bias = False)
    self.value = nn.Linear(n_embd,head_size,bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)# B,T,C
    q = self.query(x)# B,T,C
    v = self.value(x)# B,T,C
    wei = q @ k.transpose(-2,-1)# B,T,C * B,C,T => B,T,T
    wei = wei.masked_fill(self.tril[:T,:T] == 0,float("-inf"))
    wei = wei / (k.shape[-1]**0.5)
    wei = f.softmax(wei,dim= -1)
    out = wei @ v # B,T,T* B,T,C = B,T,C

    return out


class MultiHeadAttention(nn.Module):

  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for num in range(num_heads)])
  
  def forward(self,x):
    return  torch.cat([h(x) for h in self.heads],dim=-1)

class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(n_embd,n_embd),
      nn.ReLU()
    )

  def forward(self,x):
    out = self.layers(x)
    return out



# bigram model 
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)
    self.multi_heads = MultiHeadAttention(4,n_embd//4) #4 heads for 8_dim self_attention
    self.ffnn = FeedForward(n_embd)
    self.position_embedding_table = nn.Embedding(block_size,n_embd)

  def forward(self,idx,targets = None):
    B,T = idx.shape

    tok_emd = self.token_embedding_table(idx)#(B,T,C)
    pos_emd = self.position_embedding_table(torch.arange(T,device = device))#T,C
    x = pos_emd + tok_emd
    x = self.multi_heads(x)
    x = self.ffnn(x)
    logits  = self.lm_head(x) # (B,T,vocab_size)
    if targets is None: # Check if targets is None
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)#making resizing array
      targets = targets.view(B*T)
      #neg loss
      loss = f.cross_entropy(logits,targets)

    return logits,loss

  def generate(self,idx,max_word_tokens):
    for _ in range(max_word_tokens):
      #cropping idx to be equal to stay in postion encodeing table
      idx_cropped = idx[:,-block_size:]
      logits,loss = self(idx_cropped)
      logits = logits[:,-1,:]#(gives (B,C))
      #applying softmax
      probs = f.softmax(logits ,dim = 1)
      next_idx = torch.multinomial(probs,num_samples = 1)#(B,1)
      idx = torch.cat((idx,next_idx),dim = 1)
    return idx

model = BigramLanguageModel()
m = model.to(device)


# declaring optimizers and training loop 
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)
for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = est_loss()
    print(f"step{iter}: train_loss {losses['train']:.4f},val_loss = {losses['val']:.4f}")
  xb,yb = get_batch('train')
  logits,loss = m(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1),dtype = torch.long,device = device)
print(decode(m.generate(context,max_word_tokens=100)[0].tolist()))










         















