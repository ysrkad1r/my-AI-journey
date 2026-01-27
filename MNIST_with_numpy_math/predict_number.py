import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def load_model():
    data = np.load('model_weights.npz')
    return data['W1'], data['b1'], data['W2'], data['b2']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def shift_to_center_of_mass(img_arr):
    total_mass = np.sum(img_arr)
    
    if total_mass == 0:
        return img_arr

    Y, X = np.indices(img_arr.shape)
    
    center_y = np.sum(Y * img_arr) / total_mass
    center_x = np.sum(X * img_arr) / total_mass
    
    shift_y = 14.0 - center_y
    shift_x = 14.0 - center_x
    
    rows, cols = img_arr.shape
    shifted_img = np.zeros_like(img_arr)
    
    shift_y_int = int(round(shift_y))
    shift_x_int = int(round(shift_x))
    
    for y in range(rows):
        for x in range(cols):
            new_y = y + shift_y_int
            new_x = x + shift_x_int
            
            if 0 <= new_y < rows and 0 <= new_x < cols:
                shifted_img[new_y, new_x] = img_arr[y, x]
                
    return shifted_img

def smart_predict_image(image_path, W1, b1, W2, b2):

    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img_arr = np.array(img)
    
    coords = np.argwhere(img_arr > 50)
    if coords.size == 0: return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_arr[y0:y1, x0:x1]
    cropped_img = Image.fromarray(cropped)
    
    w, h = cropped_img.size
    scale = 18.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_digit = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    temp_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    temp_img.paste(resized_digit, (paste_x, paste_y))
    temp_arr = np.array(temp_img)
    
    final_arr = shift_to_center_of_mass(temp_arr)
    
    X_input = final_arr.reshape(784, 1) / 255.0
    
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_input)
    prediction = get_predictions(A2)
    
    print(f"Tahmin: {prediction[0]}")
    probabilities = {i: round(A2[i,0]*100, 2) for i in range(10)}
    print(f"Olasılıklar: {probabilities}")
    
    plt.imshow(final_arr, cmap='gray')
    plt.title(f"-> Tahmin: {prediction[0]}")
    plt.show()

print("Model ağırlıkları yükleniyor...")
try:
    W1, b1, W2, b2 = load_model()
    print("Model başarıyla yüklendi!")
except FileNotFoundError:
    print("HATA: 'model_weights.npz' dosyası bulunamadı!")
    print("Lütfen önce eğitim kodunu çalıştırıp ağırlıkları kaydettiğinden emin ol.")
    exit()

print("Tahmin yapılıyor...")
try:
    smart_predict_image('sayi.png', W1, b1, W2, b2)
except Exception as e:
    print(f"Bir hata oluştu: {e}")