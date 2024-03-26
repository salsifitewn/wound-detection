from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from deep_translator import GoogleTranslator
import streamlit as st

processor = ViTImageProcessor.from_pretrained('Hemg/Wound-classification')
model = ViTForImageClassification.from_pretrained('Hemg/Wound-classification')
def main():
   uploaded_file = st.file_uploader("Upload image",type=["png", "jpg", "jpeg"] )
   if uploaded_file is not None:
      bytes_data = uploaded_file.getvalue()
      st.write("filename:", uploaded_file.name)
      image = Image.open(uploaded_file).convert("RGB")


      inputs = processor(images=image, return_tensors="pt")
      outputs = model(**inputs)
      logits = outputs.logits
      predicted_class_idx = logits.argmax(-1).item()
      print("Predicted class:", GoogleTranslator(source='auto', target='fr').translate(model.config.id2label[predicted_class_idx]))
      print("Predicted class:",model.config.id2label[predicted_class_idx])
      st.image(bytes_data)
      st.text('Predicted class:' + GoogleTranslator(source='auto', target='fr').translate(model.config.id2label[predicted_class_idx]))
      st.text('Predicted class:' + model.config.id2label[predicted_class_idx])

if __name__ == "__main__":
    main()