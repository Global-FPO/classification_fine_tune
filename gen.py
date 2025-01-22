from cleaning import text_handle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

ven_mod="fine_tuned_T5small_01"
gl_mod="fine_tuned_T5small_02"

model_n_ft_01 = AutoModelForSeq2SeqLM.from_pretrained(ven_mod)
tokenizer_n_ft_01 = AutoTokenizer.from_pretrained(ven_mod)

model_gl_ft_02 = AutoModelForSeq2SeqLM.from_pretrained(gl_mod)
tokenizer_gl_ft_02 = AutoTokenizer.from_pretrained(gl_mod)
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_n_ft_01.to(device)   
model_gl_ft_02.to(device)   
   
# def generate_vendor_name(description):
#     input_text = description
#     inputs = tokenizer(input_text, return_tensors="pt")#.to(device)
#     outputs = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)    

def generate_vendor_name(description):
    
    input_text = description
    inputs_name = tokenizer_n_ft_01(input_text, return_tensors="pt").to(device)
    outputs_name = model_n_ft_01.generate(**inputs_name, max_length=100)
    vname=tokenizer_n_ft_01.decode(outputs_name[0], skip_special_tokens=True)
    
    # inference of GL Account
    inputs_gl = tokenizer_gl_ft_02(input_text, return_tensors="pt").to(device)
    outputs_gl = model_gl_ft_02.generate(**inputs_gl, max_length=100)
    glname=tokenizer_gl_ft_02.decode(outputs_gl[0], skip_special_tokens=True)
    #print(vname,'name',glname)

    # et=time.time()
    # print(et-st)
    return vname,glname



def generate_vendor_conf(description):
    st=time.time()
    input_text = description
    
#    desc=f"""You are special AI that  generate vendor name for given transaction description.Transaction Description : {input_text} and his vendor name is
#"""
#    

    inputs_name = tokenizer_n_ft_01(input_text, return_tensors="pt", max_length=100, truncation=True, padding=True).to(device)
    decoder_input_ids_name = torch.tensor([[tokenizer_n_ft_01.pad_token_id]]).to(device)

    # Inference
    with torch.no_grad():
        outputs_name = model_n_ft_01(**inputs_name, decoder_input_ids=decoder_input_ids_name)
    outputs_text_name = model_n_ft_01.generate(**inputs_name, max_length=100)

    # Get logits and apply softmax to obtain probabilities
    logits_name = outputs_name.logits
    probs_name = torch.nn.functional.softmax(logits_name, dim=-1)

    # Get the predicted class and confidence score
    confidences_name, predicted_classes_name = torch.max(probs_name, dim=-1)

    # Convert to CPU and numpy for further processing
    predicted_classes_name = predicted_classes_name.cpu().numpy()
    confidence_name = confidences_name.cpu().numpy()[0]
    conf_name=confidence_name.item()

    # Decode predicted token IDs into text
    #predicted_text = tokenizer_n_ft_01.decode(predicted_classes[0], skip_special_tokens=True)
    predicted_name=tokenizer_n_ft_01.decode(outputs_text_name[0], skip_special_tokens=True)
    #print('text :',t)
    
    
    
    # FOR GL ACCOUNT
    
    
    
    

    inputs_gl = tokenizer_gl_ft_02(input_text, return_tensors="pt", max_length=100, truncation=True, padding=True).to(device)
    decoder_input_ids_gl = torch.tensor([[tokenizer_gl_ft_02.pad_token_id]]).to(device)

    # Inference
    with torch.no_grad():
        outputs_gl = model_gl_ft_02(**inputs_gl, decoder_input_ids=decoder_input_ids_gl)
    outputs_text_gl = model_gl_ft_02.generate(**inputs_gl, max_length=100)

    # Get logits and apply softmax to obtain probabilities
    logits_gl = outputs_gl.logits
    probs_gl = torch.nn.functional.softmax(logits_gl, dim=-1)

    # Get the predicted class and confidence score
    confidences_gl, predicted_classes_gl = torch.max(probs_gl, dim=-1)

    # Convert to CPU and numpy for further processing
    predicted_classes_gl = predicted_classes_gl.cpu().numpy()
    confidence_gl = confidences_gl.cpu().numpy()[0]
    conf_gl=confidence_gl.item()
 
    # Decode predicted token IDs into text
    #predicted_text = tokenizer_n_ft_01.decode(predicted_classes[0], skip_special_tokens=True)

    predicted_gl=tokenizer_gl_ft_02.decode(outputs_text_gl[0], skip_special_tokens=True)
    et=time.time()
    print('time :',et-st)
    # Return the predicted vendor name and its confidence score
    return predicted_name, conf_name, predicted_gl, conf_gl



def vendor_gen(data):
    data['clean_Descriptions']=data['Descriptions'].apply(text_handle)
    #data[''],data['gl']=data['clean_Descriptions'].apply(generate_vendor_name)
    data['Predicted Vendor Name'], data['Vendor Name Confidence'], data['Predicted GL Account'],data['GL Account Confidence']= zip(*data['clean_Descriptions'].apply(generate_vendor_conf))   
    return data

#v=generate_vendor_conf('foreign exchange rate adjustment fee wagamama')
#print(v)


