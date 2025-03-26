import os
import google.generativeai as genai
import streamlit as st
from PIL import Image
import io
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Ensure this is set in your .env file

if not GOOGLE_API_KEY:
    st.error("🚨 Google Gemini API key is missing. Set it as an environment variable.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Set the updated model
MODEL_NAME = "gemini-1.5-flash"

# -------------- 🎨 Streamlit UI Styling ----------------
st.set_page_config(page_title="CHAT WITH YOUR IMAGES", page_icon="🤖", layout="wide")

# ✅ Apply Dark Robo-Themed Background
def set_background(image_url):
    page_bg = f"""
    <style>
    .stApp {{
        background: url("{image_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    .content-box {{
        background: rgba(0, 0, 0, 0.8);
        padding: 20px;
        border-radius: 15px;
        max-width: 700px;
        margin: auto;
        color: white;
    }}
    h1, h4, h2 {{
        color: #00FFFF; /* Futuristic cyan color */
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# 🔹 Set Dark Robo-Themed Background Image
background_image_url = "https://wallpaperaccess.com/full/2118293.jpg"  
set_background(background_image_url)

# -------------- 📤 Main Content ----------------
st.markdown(
    """
    
    <div class="content-box">
     <h1 style="text-align: center;color:rgb(94, 169, 255);">🤖 CHAT WITH YOUR IMAGES 👀</h1>
     <h4 style="text-align: center;color:rgb(227, 227, 54);">Upload an image and let AI analyze it!</h4>
     <h5 style="text-align: center;color:rgb(230, 99, 191);">AI doesn’t just mimic nature—it dreams in forests and oceans.</h5>
        <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Instructions
st.sidebar.header("🔹 How to Use:")
st.sidebar.info(
    """
    1. Upload an image (JPEG, PNG, JPG).  
    2. Type your question about the image.  
    3. Click 'Analyze Image' and get AI insights!  
    """
)

# 📤 File Upload Section
st.subheader("📤 Upload an Image")

file = st.file_uploader("Drag and drop or browse files", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)  # Display uploaded image
    st.success("✅ Image uploaded successfully!")

    # Text Input for Question
    user_question = st.text_input("💬 Ask a question about your image:", placeholder="e.g., What objects are in this image?")

    # Convert file to PIL Image
    image = Image.open(file).convert("RGB")  

    # Convert image to byte stream
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="JPEG")
    img_byte_array = img_byte_array.getvalue()  # Convert to bytes

    # -------------- 🚀 Image Processing ----------------
    if user_question:
        if st.button("🔍 Analyze Image"):
            with st.spinner(text="🤖 AI is thinking..."):
                try:
                    # Use Google Gemini API with image data
                    model = genai.GenerativeModel(MODEL_NAME)
                    response = model.generate_content(
                        [user_question, {"mime_type": "image/jpeg", "data": img_byte_array}]  # Correct way to pass an image
                    )

                    # Display response
                    st.markdown(
                        f"""
                        <div class="content-box">
                            <h2>🧠 AI Response:</h2>
                            <p>{response.text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
