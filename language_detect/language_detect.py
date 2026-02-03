import torch
import torch.nn.functional as F
import random

words = open("words_big.txt",'r').read().splitlines()

tr_words = words[0:1000]
en_words = words[1000:2000]
de_words = words[2000:]

languages = ["TR","EN","DE"]
combined_data = []

for w in tr_words: combined_data.append((w.lower(),0))
for w in en_words: combined_data.append((w.lower(),1))
for w in de_words: combined_data.append((w.lower(),2))

chars = sorted(list(set(''.join(words).lower())))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 5

def build_dataset(data): # ------------------_-_--_-_--__--_-------->>>>>>>>> BURADA KALDIM
    X, Y = [], []
    
    for w , label in data:
        if len(w) < block_size:
            w_padded = w + '.' * (block_size - len(w))
            context = [stoi[ch] for ch in w_padded]
            X.append(context)
            Y.append(label)
        else:
            for i in range(len(w) - block_size + 1):
                window_string = w[i: i+block_size]
                context = [stoi[ch] for ch in window_string]
                X.append(context)
                Y.append(label)

    X = torch.tensor(X) 
    Y = torch.tensor(Y)
    
    print(X.shape, Y.shape)
    return X,Y

random.shuffle(combined_data)

n1 = int(0.8*len(combined_data))
n2 = int(0.9*len(combined_data))

Xtr , Ytr = build_dataset(combined_data[:n1])
Xdev , Ydev = build_dataset(combined_data[n1:n2])
Xte , Yte = build_dataset(combined_data[n2:])

alphabet_size = len(stoi)
embed_size = 8
hidden_size = 32
output_size = 3

C = torch.randn(alphabet_size, embed_size)
W1 = torch.randn(block_size * embed_size,hidden_size)
b1 = torch.randn(hidden_size)
W2 = torch.randn(hidden_size,output_size) * 0.1
b2 = torch.randn(output_size) * 0
parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True

batch_size = 64

for i in range(12000):
    
    #minibatch constructing
    ix = torch.randint(0, Xtr.shape[0],(batch_size,))   

    #forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Ytr[ix])

    #backward pass 
    for p in parameters:
        p.grad = None
    loss.backward()

    #update parameters
    lr = 0.1 if i<6000 else 0.05
    for p in parameters:
        p.data += -lr * p.grad
    
    if i%1000 == 0:
       print(f"{i}th iteration's loss is: {loss.item()}")

print("--------"*10)

emb = C[Xtr]
h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ytr)
print(loss)

print("--------"*10)

emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ydev)
print(loss)

print("--------"*10)

def predict_soft(word):
    word = str(word).lower()
    
    total_probs = torch.zeros(1, output_size) 
    
    if len(word) < block_size:
        padded_word = word + '.' * (block_size - len(word))
        context = [stoi[ch] for ch in padded_word]

        total_probs += get_probs(context)

    else:
        num_windows = len(word) - block_size + 1
        for i in range(num_windows):
            target_word = word[i : i+block_size]
            context = [stoi[ch] for ch in target_word]

            total_probs += get_probs(context)
            
    ix = torch.argmax(total_probs).item()
    
    final_percentages = (total_probs / total_probs.sum()) * 100
    print(f"Detay: {word} -> TR: %{final_percentages[0][0]:.1f}, EN: %{final_percentages[0][1]:.1f}, DE: %{final_percentages[0][2]:.1f}")

    return languages[ix]

def get_probs(context):
    emb = C[torch.tensor(context)]
    h = torch.tanh(emb.view(1,-1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1) # Örn: [0.1, 0.8, 0.1]
    return probs

# TEST
for _ in range(5):
    target_word = str(input("Enter the word: "))
    print(f"Result: {predict_soft(target_word)}\n")