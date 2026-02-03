import pandas as pd
import torch
import torch.nn.functional as F

df = pd.read_csv('bmw.csv')

# Clean data
df['model'] = df['model'].str.strip()
df['transmission'] = df['transmission'].str.strip()
df['fuelType'] = df['fuelType'].str.strip()

#shuffling
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# %90 train , %10 test slicing
df_tr = df.iloc[:9000]
df_te = df.iloc[9000:]

# Embeddings
emb_mdl_sz = 20
model = df_tr['model'].unique()
model_stoi = {s:i for i,s in enumerate(sorted(model))}
model_size = model.shape[0] 
C_model = torch.randn(model_size,emb_mdl_sz,requires_grad=True) 

emb_trns_sz = 8
transmiss = df_tr["transmission"].unique()
trans_stoi = {s:i for i,s in enumerate(sorted(transmiss))}
trans_size = transmiss.shape[0]
C_transmiss = torch.randn(trans_size,emb_trns_sz,requires_grad=True)

emb_fl_sz = 4
f_type = df_tr["fuelType"].unique()
fType_stoi = {s:i for i,s in enumerate(sorted(f_type))}
fType_size = f_type.shape[0]
C_fType = torch.randn(fType_size,emb_fl_sz,requires_grad=True)

# Tensors
mpg_t = torch.tensor(df_tr["mpg"].values / 100, dtype=torch.float32).view(-1,1)
year_t = torch.tensor((df_tr["year"].values - 2000) / 20, dtype=torch.float32).view(-1,1)
mlg_t = torch.tensor(df_tr["mileage"].values / 100000, dtype=torch.float32).view(-1,1)
tx_t = torch.tensor(df_tr["tax"].values / 1000, dtype=torch.float32).view(-1,1)
ngsz_t = torch.tensor(df_tr["engineSize"].values / 10, dtype=torch.float32).view(-1,1)

model_ix = torch.tensor(df_tr['model'].map(model_stoi).values, dtype=torch.long).view(-1,1)
trans_ix = torch.tensor(df_tr['transmission'].map(trans_stoi).values, dtype=torch.long).view(-1,1)
fType_ix = torch.tensor(df_tr['fuelType'].map(fType_stoi).values, dtype=torch.long).view(-1,1)

Xtr_ix = torch.cat([model_ix,trans_ix,fType_ix],dim=1) 
Xtr_t = torch.cat([mpg_t,year_t,mlg_t,tx_t,ngsz_t],dim=1)
Ytr = torch.tensor(df_tr["price"].values / 100000, dtype=torch.float32).view(-1,1)

# Test Data
mpg_te = torch.tensor(df_te["mpg"].values / 100, dtype=torch.float32).view(-1,1)
year_te = torch.tensor((df_te["year"].values - 2000) / 20, dtype=torch.float32).view(-1,1)
mlg_te = torch.tensor(df_te["mileage"].values / 100000, dtype=torch.float32).view(-1,1)
tx_te = torch.tensor(df_te["tax"].values / 1000, dtype=torch.float32).view(-1,1)
ngsz_te = torch.tensor(df_te["engineSize"].values / 10, dtype=torch.float32).view(-1,1)

model_ix_te = torch.tensor(df_te['model'].map(model_stoi).fillna(0).values, dtype=torch.long).view(-1,1)
trans_ix_te = torch.tensor(df_te['transmission'].map(trans_stoi).fillna(0).values, dtype=torch.long).view(-1,1)
fType_ix_te = torch.tensor(df_te['fuelType'].map(fType_stoi).fillna(0).values, dtype=torch.long).view(-1,1)

Xte_ix = torch.cat([model_ix_te, trans_ix_te, fType_ix_te], dim=1) 
Xte_t = torch.cat([mpg_te, year_te, mlg_te, tx_te, ngsz_te], dim=1)
Yte = torch.tensor(df_te["price"].values / 100000, dtype=torch.float32).view(-1,1)

# --- MODAL ---
input_size = 37
hidden_size = 256
hidden_size2 = 128
hidden_size3 = 64
output_size = 1

# Kaiming He Initialization
W1 = torch.randn(input_size, hidden_size) * (2/input_size)**0.5
b1 = torch.zeros(hidden_size)
W2 = torch.randn(hidden_size, hidden_size2) * (2/hidden_size)**0.5
b2 = torch.zeros(hidden_size2)
W3 = torch.randn(hidden_size2, hidden_size3) * (2/hidden_size2)**0.5
b3 = torch.zeros(hidden_size3)
W4 = torch.randn(hidden_size3, output_size) * (2/hidden_size3)**0.5
b4 = torch.zeros(output_size)

parameters = [C_model,C_transmiss,C_fType,W1,b1,W2,b2,W3,b3,W4,b4]
for p in parameters:
    p.requires_grad = True

# --- OPTIMIZER ---
optimizer = torch.optim.AdamW(parameters, lr=0.01)

# Scheduler: Reduce LR just only in platos
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)

print("Training is starting...")

for i in range(5001):

    # 1. Forward pass
    emb_m = C_model[Xtr_ix[:, 0]]
    emb_t = C_transmiss[Xtr_ix[:, 1]]
    emb_f = C_fType[Xtr_ix[:, 2]]
    h_input = torch.cat([emb_m, emb_t, emb_f, Xtr_t], dim=1)

    h1 = F.relu(h_input @ W1 + b1)
    h1 = F.dropout(h1, p=0.1, training=True) # Dropout 
    
    h2 = F.relu(h1 @ W2 + b2)
    h3 = F.relu(h2 @ W3 + b3)
    pred = h3 @ W4 + b4

    loss = ((pred - Ytr)**2).mean()

    # 2. Backward & Update passes
    optimizer.zero_grad() # Reset gradients
    loss.backward()
    optimizer.step()      # Update parameters
    
    # Update LR Scheduler
    scheduler.step(loss)

    # 3. Test and logging
    if i % 500 == 0:
        with torch.no_grad():
            emb_m_te = C_model[Xte_ix[:, 0]]
            emb_t_te = C_transmiss[Xte_ix[:, 1]]
            emb_f_te = C_fType[Xte_ix[:, 2]]
            
            h_input_te = torch.cat([emb_m_te, emb_t_te, emb_f_te, Xte_t], dim=1)
            
            h1_te = F.relu(h_input_te @ W1 + b1)
            h2_te = F.relu(h1_te @ W2 + b2)
            h3_te = F.relu(h2_te @ W3 + b3)
            pred_te = h3_te @ W4 + b4
            
            loss_te = ((pred_te - Yte)**2).mean()
            
            print(f"Iter {i} | Train Loss: {loss.item():.6f} | Test Loss: {loss_te.item():.6f} | LR: {optimizer.param_groups[0]['lr']}")

# --- KAYDETME VE SONUÇ ---
torch.save({
    'model_state': parameters,
    'dicts': {'model_stoi': model_stoi, 'trans_stoi': trans_stoi, 'fType_stoi': fType_stoi},
    'sizes': {'input_size': input_size, 'hidden_sizes': [hidden_size, hidden_size2, hidden_size3]}
}, 'bmw_fiyat_modeli_adam.pt')

print("\n--- RESULTS ---")
with torch.no_grad():
    
    emb_m_te = C_model[Xte_ix[:, 0]]
    emb_t_te = C_transmiss[Xte_ix[:, 1]]
    emb_f_te = C_fType[Xte_ix[:, 2]]
    h_input_te = torch.cat([emb_m_te, emb_t_te, emb_f_te, Xte_t], dim=1)
    
    h1_te = F.relu(h_input_te @ W1 + b1)
    h2_te = F.relu(h1_te @ W2 + b2)
    h3_te = F.relu(h2_te @ W3 + b3)
    pred_te = h3_te @ W4 + b4

    real_prices = Yte * 100000
    guessed_prices = pred_te * 100000
    
    for i in range(10):
        real = real_prices[i].item()
        guess = guessed_prices[i].item()
        diff = guess - real
        accuracy = (abs(diff) / real) * 100
        print(f"Real: {real:.0f} | Guess: {guess:.0f} | Error: %{accuracy:.1f}")