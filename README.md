

## installation

1. Clone the repository:

2. Change to the project directory:

        
3. Install the required dependencies:

        pip install -r requirements.txt

4. Obtain an **GOOGLE GEMINI**. You can sign up for an API key at. https://aistudio.google.com

5. Replace the placeholder API key in the main.py file with your actual GOOGLE GEMINI API key:


6. Run the Streamlit application:

        streamlit run main.py

7. Open your web browser and go to http://localhost:8501 to access the application.

## usage

1. Upload an image by clicking the file upload button.

2. The uploaded image will be displayed.

3. Enter a question about the image in the text input field.

4. The conversational AI agent will generate a response based on the provided question and image.

5. The response will be displayed below the question input.

## tools

The application utilizes the following custom tools:

- **ImageCaptionTool**: Generates a textual caption for the uploaded image.
- **ObjectDetectionTool**: Performs object detection on the uploaded image and identifies the objects present.
