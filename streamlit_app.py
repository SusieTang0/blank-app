import json
import streamlit as st
from xgboost import XGBClassifier

st.title("左右布局示例")

# create two columns
left_col, right_col = st.columns(2)

with open('unified_physician_label_encoder.json', 'r') as json_file:
    loaded_mapping = json.load(json_file)

# Print the loaded mapping
# print(f"---------Loaded LabelEncoder mapping: {loaded_mapping}")

# To reverse the mapping (from encoded value to label):
reversed_mapping = {v: k for k, v in loaded_mapping.items()}
# print(f"Reversed LabelEncoder mapping: {reversed_mapping}")

# Example: Using the reversed mapping to get labels
encoded_value = 12
label = reversed_mapping[encoded_value]
print(f"Encoded value {encoded_value} corresponds to label: {label}")

label_list = list(reversed_mapping.values())
print(label_list)

default_label = "NaN"  # 假设你想默认选这个名字
if default_label in label_list:
    default_index = label_list.index(default_label)
else:
    default_index = 0

with open('Provider_label_encoder_mapping.json', 'r') as json_file:
    loaded_mapping_provider = json.load(json_file)
reversed_mapping_provider = {v: k for k, v in loaded_mapping_provider.items()}
provider_label_list = list(reversed_mapping_provider.values())


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
    income = st.number_input("月收入（元）", value=5000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    attending_physician = st.selectbox("Attending Physician", label_list,index=default_index)
    operating_physician = st.selectbox("Operating Physician", label_list,index=default_index)
    other_physician = st.selectbox("Other Physician", label_list,index=default_index)
    provider = st.selectbox("Provider", provider_label_list)
    claim_submitted = st.form_submit_button("提交")

# 提交后的处理

if claim_submitted:
      xgb_model = XGBClassifier()
      xgb_model.load_model("xgb_model_op.json")
      y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

      # 如果你有 threshold
      threshold = 0.5631  # 自定义的
      y_pred = (y_pred_prob > threshold).astype(int)
     