import json
import streamlit as st

st.title("左右布局示例")

# create two columns
left_col, right_col = st.columns(2)

with open('unified_physician_label_encoder.json', 'r') as json_file:
    loaded_mapping_physician = json.load(json_file)

# Print the loaded mapping
# print(f"---------Loaded LabelEncoder mapping: {loaded_mapping}")

# To reverse the mapping (from encoded value to label):
reversed_mapping_physician = {v: k for k, v in loaded_mapping_physician.items()}
# print(f"Reversed LabelEncoder mapping: {reversed_mapping}")

# Example: Using the reversed mapping to get labels
# encoded_value_physician = 12
# label_physician = reversed_mapping_physician[encoded_value_physician]
# print(f"Encoded value {encoded_value_physician} corresponds to label: {label_physician}")

physician_label_list = list(reversed_mapping_physician.values())


default_label = "NaN"  # 假设你想默认选这个名字
if default_label in physician_label_list:
    default_index_physician = physician_label_list.index(default_label)
else:
    default_index_physician = 0

with open('Provider_label_encoder_mapping.json', 'r') as json_file:
    loaded_mapping_provider = json.load(json_file)
reversed_mapping_provider = {v: k for k, v in loaded_mapping_provider.items()}
provider_label_list = list(reversed_mapping_provider.values())

default_label = "NaN"  # 假设你想默认选这个名字
if default_label in provider_label_list:
    default_index_provider = provider_label_list.index(default_label)
else:
    default_index_provider = 0

with open('unified_diagnosis_label_encoder.json', 'r') as json_file:
    loaded_mapping_diagnosis = json.load(json_file)
reversed_mapping_diagnosis = {v: k for k, v in loaded_mapping_diagnosis.items()}
diagnosis_label_list = list(reversed_mapping_diagnosis.values())
default_label = "NaN"  # 假设你想默认选这个名字
if default_label in diagnosis_label_list:
    default_index_diagnosis = diagnosis_label_list.index(default_label)
else:
    default_index_diagnosis = 0
# 左边内容
with left_col:
    st.header("Patient info")
    st.write("Please enter patient information here")
    with st.form(key='patient_form'):
      name = st.text_input("Name")
      age = st.slider('Age', 0, 120, 50)
      race = st.slider('Race', 1, 120, 50)
      gender = st.selectbox("Gender", ["Male", "Female"])
      patient_submitted = st.form_submit_button("Submit")

# 右边内容
with right_col:
    st.header("Prediction")
    if patient_submitted:
      st.success(f"已提交：年龄={age}, 收入={race}, 性别={gender}")
    
st.title("Claim info enter")

# 创建表单
with st.form(key='claim_form'):
    amount_reimbursed = st.number_input("Amount Reimbursed", min_value=0, value=0)
    provider = st.selectbox("Provider", provider_label_list)
    attending_physician = st.selectbox("Attending Physician", physician_label_list,index=default_index_physician)
    operating_physician = st.selectbox("Operating Physician", physician_label_list,index=default_index_physician)
    other_physician = st.selectbox("Other Physician", physician_label_list,index=default_index_physician)
    diagnosis_code_1 = st.selectbox("Diagnosis code 1", diagnosis_label_list,index=default_index_diagnosis)
    diagnosis_code_2 = st.selectbox("Diagnosis code 2", diagnosis_label_list,index=default_index_diagnosis)
    claim_submitted = st.form_submit_button("提交")

# 提交后的处理

if claim_submitted:
      st.success(f"Amount Reimbursed={amount_reimbursed}\r\rAttending Physician={attending_physician}\r\rOperating Physician={operating_physician}\r\rOperating Provider={provider}")