import os
import random
import pandas as pd
from PIL import Image
import numpy as np

def create_dummy_image(path, size=(224, 224), color=(0,0,0)):
    # Create a solid color image for testing
    img = Image.new("RGB", size, color)
    # Add some random noise
    noise = np.random.randint(0, 50, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(np.array(img) + noise)
    img.save(path)

# --- 1. Skin (ISIC 2020) ---
skin_dir = "data/isic2020/train"
os.makedirs(skin_dir, exist_ok=True)
skin_data = []
for i in range(20):
    img_name = f"ISIC_dummy_{i:04d}"
    create_dummy_image(f"{skin_dir}/{img_name}.jpg", size=(256, 256), color=(200, 150, 150))
    skin_data.append({
        "image_name": img_name,
        "diagnosis": random.choice(["melanoma", "melanocytic nevus", "basal cell carcinoma"]),
        "age_approx": random.randint(20, 80),
        "sex": random.choice(["male", "female"]),
        "anatom_site_general_challenge": random.choice(["torso", "lower extremity", "head/neck"])
    })
pd.DataFrame(skin_data).to_csv("data/isic2020/train.csv", index=False)
print("Created dummy ISIC 2020 data.")

# --- 2. APTOS (DR) ---
aptos_dir = "data/aptos/train_images"
os.makedirs(aptos_dir, exist_ok=True)
aptos_data = []
for i in range(20):
    img_name = f"aptos_dummy_{i:04d}"
    create_dummy_image(f"{aptos_dir}/{img_name}.png", size=(380, 380), color=(150, 50, 50))
    aptos_data.append({
        "id_code": img_name,
        "diagnosis": random.randint(0, 4)
    })
pd.DataFrame(aptos_data).to_csv("data/aptos/train.csv", index=False)
print("Created dummy APTOS 2019 data.")

# --- 3. RIM-ONE DL (Glaucoma) ---
rim_glaucoma = "data/rim_one/Images/Glaucoma"
rim_normal = "data/rim_one/Images/Normal"
os.makedirs(rim_glaucoma, exist_ok=True)
os.makedirs(rim_normal, exist_ok=True)
for i in range(10):
    create_dummy_image(f"{rim_glaucoma}/glaucoma_{i}.png", size=(224, 224), color=(200, 200, 200))
    create_dummy_image(f"{rim_normal}/normal_{i}.png", size=(224, 224), color=(200, 200, 200))
print("Created dummy RIM-ONE GL data.")

# --- 4. Chest X-Ray ---
chest_dir = "data/chest_xray"
os.makedirs(chest_dir, exist_ok=True)
# Module1 expects a specific structure for chest xrays.
# Let's create a dummy train.csv if there is one expected or just dummy images.
# Looking at typical structures, let's create a minimal setup or we can update module1_chest.py if needed.
print("Basic dummy data generation complete! To test the skin and eye modules, you can now run train_all.py")
