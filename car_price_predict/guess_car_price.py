import torch
import torch.nn.functional as F

print("Modal loading...")
checkpoint = torch.load('bmw_fiyat_modeli_adam.pt')

dicts = checkpoint['dicts']
model_stoi = dicts['model_stoi']
trans_stoi = dicts['trans_stoi']
fType_stoi = dicts['fType_stoi']

sizes = checkpoint['sizes']
input_size = sizes['input_size']

params = checkpoint['model_state']
C_model, C_transmiss, C_fType = params[0], params[1], params[2]
W1, b1 = params[3], params[4]
W2, b2 = params[5], params[6]
W3, b3 = params[7], params[8]
W4, b4 = params[9], params[10]

print("Modal has initialized succesfully")

def predict(model_adi, yil, vites, yakit, km, vergi, motor_hacmi, mpg=50.0):

    model_adi = str(model_adi).strip()
    vites = str(vites).strip() 
    yakit = str(yakit).strip()
    
    try:
        mdl_ix = torch.tensor([model_stoi[model_adi]])
        trns_ix = torch.tensor([trans_stoi[vites]])
        fuel_ix = torch.tensor([fType_stoi[yakit]])
    except KeyError:
        return "ERROR: You have initialized wrong model_name , transmission or fuel_type!"

    # Normalizing data
    year_val = (yil - 2000) / 20.0
    km_val = km / 100000.0
    tax_val = vergi / 1000.0
    eng_val = motor_hacmi / 10.0
    mpg_val = mpg / 100.0

    # Create tensors
    x_num = torch.tensor([[mpg_val, year_val, km_val, tax_val, eng_val]], dtype=torch.float32)

    # Forward Pass
    with torch.no_grad():
        emb_m = C_model[mdl_ix]
        emb_t = C_transmiss[trns_ix]
        emb_f = C_fType[fuel_ix]

        h_input = torch.cat([emb_m, emb_t, emb_f, x_num], dim=1)

        # ReLU -> ReLU -> ReLU -> Linear
        h1 = F.relu(h_input @ W1 + b1) # Nx37.37x256 -> Nx256
        h2 = F.relu(h1 @ W2 + b2) # Nx256.256x128 -> Nx128
        h3 = F.relu(h2 @ W3 + b3) # Nx128.128x64 -> Nx64 
        pred = h3 @ W4 + b4 # Nx64.64x1 -> Nx1
        
        # Denormalize
        fiyat = pred.item() * 100000
        
    return fiyat

# Test
print("-" * 30)
car1 = predict("3 Series", 2016, "Automatic", "Diesel", 120000, 125, 2.0)
print(f"2016 BMW 3.20d (120k KM): {round(car1,4)} £")

car2 = predict("X5", 2019, "Automatic", "Diesel", 50000, 145, 3.0)
print(f"2019 BMW X5 (50k KM): {round(car2,4)} £")

car3 = predict("1 Series", 2012, "Manual", "Petrol", 80000, 100, 1.6)
print(f"2012 BMW 1.16i (80k KM): {round(car3,4)} £")
print("-" * 30)