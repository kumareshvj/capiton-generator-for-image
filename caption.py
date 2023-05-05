# importing libraries
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch


# pre trained object creation model, tokenizer and processor from HuggingFace
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
 
# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prediction(img_list):
    
    max_length = 15
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    img = []
    
    for image in tqdm(img_list):
        
        i_image = Image.open(image) 

        # checking whether the image is RGB or not, if it is not RGB we are going to RGB
        if i_image.mode != "RGB": 
            i_image = i_image.convert(mode="RGB")
        # Image is add to the img list
        img.append(i_image)

    # converting the imagedata to pixel value
    pixel_val = feature_extractor(images=img, return_tensors="pt").pixel_values
    # conveting th pixel value to pytroch tensor value and store in the device variable
    pixel_val = pixel_val.to(device)

    # generating the output from the pixel value
    output = model.generate(pixel_val, **gen_kwargs)

    # converting the output to text remove any speical token to strip whitespaces
    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict

# we are creating the sample image for the intereface to display
def examples():
    
    sp_images = {'Sample 1':'image/baby.jpeg','Sample 2':'image/circket.png','Sample 3':'image/photo.jpeg','Sample 4':'image/river.jpeg'} 
    
    # adding four columns 
    colms = cycle(st.columns(4)) 
    
    # to display the sample images
    for sp in sp_images.values():
        next(colms).image(sp, width=150)
        
    # looping the images to generate the caption
    for i, sp in enumerate(sp_images.values()):
        
        # creating the button for all the images to generate the caption
        if next(colms).button("Generate",key=i): 
            
            description = prediction([sp])
            st.subheader("Caption for the Image:")
            st.write(description[0])            

# creating the uploader on the interface
def upload():
    
    with st.form("uploader"):
        # Image input 
        image = st.file_uploader("Upload Images",accept_multiple_files=True,type=["jpg","png","jpeg"])
        # Generate button
        submit = st.form_submit_button("Generate")
        
        # creating the caption after clicking on the generator button
        try:
            if submit:  
                description = prediction(image)
                
                st.subheader("Description for the Image:")
                for i,caption in enumerate(description):
                    st.write(caption)
        # except is created if the image is not upload and click the genereator button 
        except IndexError:
            st.write('Image is not uploaded please try upload the image and generate again')

def main():
    # title on the tab
    st.set_page_config(page_title="Caption generation") 
    # title of the page
    st.title("Get Captions for your Image")
    
    # creating two tab side by side one for uploading and another for sample 
    tab1, tab2= st.tabs(["Upload Image", "Examples"])
    
    # upload images tab
    with tab1:
        upload()
    # example images tab
    with tab2:
        examples()
        

if __name__ == '__main__': 
    main()