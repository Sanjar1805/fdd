import streamlit as st

from fastai.vision.all import *
import pathlib

# Fayl yo'llarini tuzatish (Windows platformasi uchun)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Rasimlarni klassifikatsiya qilish")

# Rasim yuklash interfeysi
files = st.file_uploader("Rasm yuklash", type=["jpg", "svg", "png"])

if files:
    st.image(files, caption="Yuklangan rasm", use_column_width=True)
    
    # Yuklangan rasimni o'qish
    img = PILImage.create(files)
    
    # Modellni yuklash
    model = load_learner('classificator.pkl')
    
    # Bashorat qilish
    pred, pred_idx, probs = model.predict(img)
    
    # Bashoratlarni o'zbekchaga tarjima qilish uchun lug'at
    translations = {
        "Weapon": "Qurol",
        "Bird": "Qush",
        "Car": "Mashina",
        "Airplane": "Samolyot",
        "Boat": "Qayiq",
        "Telephone": "Telefon",
        "Toy": "O'yinchoq",
        "Helmet": "Dubulg'a",
        "Ball": "To'p"
    }
    
    # O'zbekcha bashorat
    uzbek_pred = translations.get(str(pred), "Noma'lum")
    
    # Natijalarni ko'rsatish
    st.write(f"Bashorat: {uzbek_pred}")
    st.write(f"Ishonch darajasi: {probs[pred_idx]:.4f}")
