#Umut Ozdemir - FastBook - 02_production

# Import the necessary libraries
import gradio as gr
from fastai.vision.all import *

# Load your pre-trained model (replace 'export.pkl' with your model's path)
learn = load_learner('export.pkl')

# Define a function to classify an input image
def classify_image(input_img):
    # Load and classify the input image
    img = PILImage.create(input_img)
    pred, pred_idx, probs = learn.predict(img)
    
    # Format the results as labels and probabilities
    labels = learn.dls.vocab
    results = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    return results

# Create a Gradio interface
demo = gr.Interface(
    fn=classify_image,           # Function to process input
    inputs="image",              # Input type is an image
    outputs="text"               # Output type is text (labels and probabilities)
)

# Launch the Gradio interface for image classification
demo.launch()
