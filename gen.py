from cleaning import text_handle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_n_ft_01 = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_T5small_01")
tokenizer_n_ft_01 = AutoTokenizer.from_pretrained("fine_tuned_T5small_01")

model_gl_ft_02 = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_T5small_02")
tokenizer_gl_ft_02 = AutoTokenizer.from_pretrained("fine_tuned_T5small_02")
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_n_ft_01.to(device)   
model_gl_ft_02.to(device)   
   
# def generate_vendor_name(description):
#     input_text = description
#     inputs = tokenizer(input_text, return_tensors="pt")#.to(device)
#     outputs = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)    

def generate_vendor_name(description):
    # st=time.time()
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

def vendor_gen(data):
    data['clean_Descriptions']=data['Descriptions'].apply(text_handle)
    #data[''],data['gl']=data['clean_Descriptions'].apply(generate_vendor_name)
    data['Predicted Vendor Name'], data['Predicted GL Account'] = zip(*data['clean_Descriptions'].apply(generate_vendor_name))   
    return data

# v=generate_vendor_name('decrease zirin tax debits')
# print(v)