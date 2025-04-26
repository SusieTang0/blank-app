import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import time

st.markdown("""
    <style>
        .center {
            paading:0px;
            margin:0px;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
        }
    </style>
    <div class="center">
       <h1>DetectiSure</h1>
    </div>
     <div class="center">
       <h3>insurance fraud prediction</h3>   
    </div>
    """, unsafe_allow_html=True)
    


if 'page' not in st.session_state:
    st.session_state.page = None

st.write("Choose the feature you want to use")
col1, col2 = st.columns(2)
with col1:
    if st.button("Benificiary Prediction"):
        st.session_state.page = "Benificiary"
with col2:
    if st.button("Insurance Fraud Prediction"):
        st.session_state.page = "Fraud"



# page = st.radio("Choose a feature", ("Benificiary Prediction","Insurance Fraud Predication"),index=-1)
if st.session_state.page == "Benificiary":
    st.header("Benificiary Predication")

   
   

    with open('state_encode_mapping.json', 'r') as json_file:
        loaded_mapping_state= json.load(json_file)
    reversed_mapping_state = {v: k for k, v in loaded_mapping_state.items()}
    state_label_list = list(reversed_mapping_state.values())
    


    with open('age_encoder_mapping.json', 'r') as json_file:
        loaded_mapping_age= json.load(json_file)
    reversed_mapping_age = {v: k for k, v in loaded_mapping_age.items()}
    age_label_list = list(reversed_mapping_age.values())
    default_label_age = "<50" 
    if default_label_age in age_label_list:
        default_index_age = age_label_list.index(default_label_age)
    else:
        default_index_age = 0

    with open('race_label_mapping.json', 'r') as json_file:
        loaded_mapping_race = json.load(json_file)
    reversed_mapping_race = {v: k for k, v in loaded_mapping_race.items()}
    race_label_list = list(reversed_mapping_race.values())
    default_label_race = "Refused to Report / Unknown" 
    if default_label_race in race_label_list:
        default_index_race = race_label_list.index(default_label_race)
    else:
        default_index_race = 0

    with open('county_encode_mapping.json', 'r') as json_file:
        loaded_mapping_county = json.load(json_file)
    reversed_mapping_county = {v: k for k, v in loaded_mapping_county.items()}
    county_label_list = list(reversed_mapping_county.values())

    gender_encode_mapping = {'Male':1,'Female':2}
    
    sample = st.selectbox('Choose the sample you want to use:', ['No sample','Sample 1','Sample 2'], index=0)
    if(sample == 'Sample 1'):
        age = "65-69"
        race = "American Indian or Alaska Native"
        gender = "Male"
        state = "OHIO"
        county = "Clark County"
        renal_disease_indicator = 'No'
        part_a_month = 12
        part_b_month = 12
        selected_diseases = ['Alzheimer', 'KidneyDisease', 'Depression','Diabetes','IschemicHeart','rheumatoidarthritis','stroke']
        charlson_index = 7
        st.write(f"Sample paitent info:\r\rAge: {age} - Race:{race} - Gender:{gender}\r\rState: {state} - County:{county}\r\rHave Renal Disease:{renal_disease_indicator}\r\rMonth number of Part A Coverage:{part_a_month}\r\rMonth number of Part B Coverage:{part_b_month}\r\rDisease:{selected_diseases}\r\rCharlson Index:{charlson_index}") 
    elif(sample == 'Sample 2'):
        age = "90-94"
        race = "Asian"
        gender = "Female"
        state = "NEW JERSEY"
        county = "Crockett County"
        renal_disease_indicator = 'No'
        part_a_month = 12
        part_b_month = 12
        selected_diseases = ['Alzheimer','Heartfailure', 'KidneyDisease', 'ObstrPulmonary','Diabetes','IschemicHeart','Osteoporasis','stroke']
        charlson_index = 8
        st.write(f"Sample paitent info:\r\rAge: {age} - Race:{race} - Gender:{gender}\r\rState: {state} - County:{county}\r\rHave Renal Disease:{renal_disease_indicator}\r\rMonth number of Part A Coverage:{part_a_month}\r\rMonth number of Part B Coverage:{part_b_month}\r\rDisease:{selected_diseases}\r\rCharlson Index:{charlson_index}")
    
    # create two columns
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Patient info")
        st.write("Please enter patient information here")
       
        with st.form(key='patient_form'):
              age = st.selectbox('Age', age_label_list, index=default_index_age)
              race = st.selectbox('Race', race_label_list, index=default_index_race)
              gender = st.selectbox("Gender", ["Male", "Female"])
              state = st.selectbox("State", state_label_list)
              county = st.selectbox("County", county_label_list)
              renal_disease_indicator = st.selectbox("Do you have Renal Disease?", ['Yes','No'])
              part_a_month = st.slider('Month number of Part A Coverage', 0,12,12)
              part_b_month = st.slider('Month number of Part B Coverage', 0,12,12)
              options = ['Alzheimer','Heartfailure', 'KidneyDisease', 'Cancer', 'ObstrPulmonary','Depression','Diabetes','IschemicHeart','Osteoporasis','rheumatoidarthritis','stroke']
              selected_diseases = st.multiselect('Please select the chronic condition you have', options)
              charlson_index = st.slider('Charlson Index', 0,37,1)
        
              patient_submitted = st.form_submit_button("Submit")
        
        

    with right_col:
        st.subheader("Prediction")
        encoded_age = loaded_mapping_age[age]
        encoded_gender = gender_encode_mapping[gender]
        encoded_race = loaded_mapping_race[race]
        encoded_state = loaded_mapping_state[state]
        encoded_county = loaded_mapping_county[county]
        encoded_renal_disease = 1 if renal_disease_indicator == 'Yes' else 0
        encoded_ChronicCond_mapping = {
            'Alzheimer': 0,
            'Heartfailure': 0,
            'KidneyDisease': 0,
            'Cancer': 0,
            'ObstrPulmonary': 0,
            'Depression': 0,
            'Diabetes': 0,
            'IschemicHeart': 0,
            'Osteoporasis': 0,
            'rheumatoidarthritis': 0,
            'stroke': 0
        }


        for disease in selected_diseases:
            encoded_ChronicCond_mapping[disease] = 1

        encoded_alzheimer = encoded_ChronicCond_mapping['Alzheimer']
        encoded_heartfailure = encoded_ChronicCond_mapping['Heartfailure']
        encoded_kidneydisease = encoded_ChronicCond_mapping['KidneyDisease']
        encoded_cancer = encoded_ChronicCond_mapping['Cancer']
        encoded_obstrpulmonary = encoded_ChronicCond_mapping['ObstrPulmonary']
        encoded_depression = encoded_ChronicCond_mapping['Depression']
        encoded_diabetes = encoded_ChronicCond_mapping['Diabetes']
        encoded_ischemicheart = encoded_ChronicCond_mapping['IschemicHeart']
        encoded_osteoporasis = encoded_ChronicCond_mapping['Osteoporasis']
        encoded_rheumatoidarthritis = encoded_ChronicCond_mapping['rheumatoidarthritis']
        encoded_stroke = encoded_ChronicCond_mapping['stroke']

        if(sample == 'No sample'):
          X_test = pd.DataFrame({
                'Gender':[encoded_gender],
                "RenalDiseaseIndicator":[encoded_renal_disease], 
                'State':[encoded_state], 
                'County':[encoded_county], 
                "NoOfMonths_PartACov":[part_a_month],
                "NoOfMonths_PartBCov":[part_b_month], 
                "ChronicCond_Alzheimer":[encoded_alzheimer],
                "ChronicCond_Heartfailure":[encoded_heartfailure], 
                "ChronicCond_KidneyDisease":[encoded_kidneydisease], 
                "ChronicCond_Cancer":[encoded_cancer],
                "ChronicCond_ObstrPulmonary":[encoded_obstrpulmonary], 
                "ChronicCond_Depression":[encoded_depression], 
                "ChronicCond_Diabetes":[encoded_diabetes],
                "ChronicCond_IschemicHeart":[encoded_ischemicheart], 
                "ChronicCond_Osteoporasis":[encoded_osteoporasis], 
                "ChronicCond_rheumatoidarthritis":[encoded_rheumatoidarthritis],
                "ChronicCond_stroke":[encoded_stroke],
                'ConditionsTotal':[len(selected_diseases)],
                'CharlsonIndex':[charlson_index],
                'RaceCatEncode':[encoded_race],   
                'AgeCatEncode':[encoded_age], 
            })
        elif(sample == 'Sample 1'):
             X_test = pd.DataFrame({
                'Gender':[1],
                "RenalDiseaseIndicator":[0], 
                'State':[39], 
                'County':[230], 
                "NoOfMonths_PartACov":[12],
                "NoOfMonths_PartBCov":[12], 
                "ChronicCond_Alzheimer":[1],
                "ChronicCond_Heartfailure":[0], 
                "ChronicCond_KidneyDisease":[1], 
                "ChronicCond_Cancer":[0],
                "ChronicCond_ObstrPulmonary":[0], 
                "ChronicCond_Depression":[1], 
                "ChronicCond_Diabetes":[1],
                "ChronicCond_IschemicHeart":[1], 
                "ChronicCond_Osteoporasis":[0], 
                "ChronicCond_rheumatoidarthritis":[1],
                "ChronicCond_stroke":[1],
                'ConditionsTotal':[7],
                'CharlsonIndex':[7],
                'RaceCatEncode':[0],  
                'AgeCatEncode':[3],   
             })
        elif(sample == 'Sample 2'):
            X_test = pd.DataFrame({
              'Gender':[2],
              "RenalDiseaseIndicator":[0], 
              'State':[34], 
              'County':[400], 
              "NoOfMonths_PartACov":[12],
              "NoOfMonths_PartBCov":[12], 
              "ChronicCond_Alzheimer":[1],
              "ChronicCond_Heartfailure":[1], 
              "ChronicCond_KidneyDisease":[1], 
              "ChronicCond_Cancer":[0],
              "ChronicCond_ObstrPulmonary":[1], 
              "ChronicCond_Depression":[0], 
              "ChronicCond_Diabetes":[1],
              "ChronicCond_IschemicHeart":[1], 
              "ChronicCond_Osteoporasis":[1], 
              "ChronicCond_rheumatoidarthritis":[0],
              "ChronicCond_stroke":[1],
              'ConditionsTotal':[8],
              'CharlsonIndex':[8],
              'RaceCatEncode':[1],  
              'AgeCatEncode':[8], 
               
            })
       
        if patient_submitted:
          xgb_model = XGBClassifier()
          xgb_model.load_model("claim_predictor_final.json")
          y_pred = xgb_model.predict(X_test)
          if y_pred == [0]:
              st.success("There is a 76% chance that this patient will receive the Annual Reimbursement Amount.")
          else:
              st.error("There is a 76% chance that this patient will NOT receive the Annual Reimbursement Amount.")
    

elif st.session_state.page == "Fraud":
    st.header("Insurance Fraud Predication")
    with open('unified_physician_label_encoder.json', 'r') as json_file:
        loaded_mapping_physician = json.load(json_file)
    reversed_mapping_physician = {v: k for k, v in loaded_mapping_physician.items()}
    physician_label_list = list(reversed_mapping_physician.values())


    default_label = "Unknown"  # 假设你想默认选这个名字
    if default_label in physician_label_list:
        default_index_physician = physician_label_list.index(default_label)
    else:
        default_index_physician = 0

    with open('Provider_label_encoder_mapping.json', 'r') as json_file:
        loaded_mapping_provider = json.load(json_file)
    reversed_mapping_provider = {v: k for k, v in loaded_mapping_provider.items()}
    provider_label_list = list(reversed_mapping_provider.values())

    default_label = "Unknown"  
    if default_label in provider_label_list:
        default_index_provider = provider_label_list.index(default_label)
    else:
        default_index_provider = 0

    with open('unified_diagnosis_label_encoder.json', 'r') as json_file:
        loaded_mapping_diagnosis = json.load(json_file)
    reversed_mapping_diagnosis = {v: k for k, v in loaded_mapping_diagnosis.items()}
    diagnosis_label_list = list(reversed_mapping_diagnosis.values())
    default_label = "Unknown"  
    if default_label in diagnosis_label_list:
        default_index_diagnosis = diagnosis_label_list.index(default_label)
    else:
        default_index_diagnosis = 0

  
    if 'input_count' not in st.session_state:
        st.session_state.input_count = 1

    col1, col2,col3 = st.columns(3)

    with col1:
        # add diagnosis selection
        if st.button("Add diagnosis code"):
            if st.session_state.input_count < 10:
                st.session_state.input_count += 1
            else:
                st.warning("At most 10 diagnosis")

    with col2:
        # delete diagnosis selection
        if st.button("Delete last diagnosis code"):
            if st.session_state.input_count > 1:
                st.session_state.input_count -= 1
            else:
                st.warning("At least one diagnosis")

    with col3:
        # reset diagnosis selection
        if st.button("Reset all diagnosis codes"):
            st.session_state.input_count = 1  # 只保留一个输入框

    if 'new_diagnosis_list' not in st.session_state:
        st.session_state.new_diagnosis_list = []
    in_hospital = st.radio(
                label="Form of visit",
                options=["Inpatient", "Outpatient"],
                index=0, 
                horizontal=True 
            )
    sample = st.selectbox('Choose the sample you want to use:', ['No sample','Sample 1','Sample 2'], index=0)
    if(sample == 'Sample 1'):
        claim_days = 1
        amount_reimbursed = 8000
        attending_physician = "PHY340149"
        operating_physician = "PHY321651"
        other_physician = "Unknown"
        diagnosis_codes = ["655",'5921','28521','5939','5989']
        provider = 'PRV54936'
        st.write(f"Sample claim info:\r\rDays for claim duration:{claim_days}\r\rAmount Reimbursed: {amount_reimbursed}\r\rAttending Physician:{attending_physician}\r\rOperating Physician:{operating_physician}\r\rOther Physician: {other_physician}\r\rDiagnosis code:{diagnosis_codes}\r\rProvider:{provider}") 
    elif(sample == 'Sample 2'):
        claim_days = 12
        amount_reimbursed = 500
        attending_physician = "PHY412904"
        operating_physician = "Unknown"
        other_physician = "PHY396473"
        diagnosis_codes = ["7237"]
        provider = 'PRV56011'
        st.write(f"Sample claim info:\r\rDays for claim duration:{claim_days}\r\rAmount Reimbursed: {amount_reimbursed}\r\rAttending Physician:{attending_physician}\r\rOperating Physician:{operating_physician}\r\rOther Physician: {other_physician}\r\rDiagnosis code:{diagnosis_codes}\r\rProvider:{provider}") 
    with st.form(key='claim_form'):
        claim_days = st.number_input("Duration of the Claim (in days)", min_value=0, value=0)
        amount_reimbursed = st.number_input("Amount Reimbursed", min_value=0, value=0)
        provider = st.selectbox("Provider", provider_label_list)
        attending_physician = st.selectbox("Attending Physician", physician_label_list,index=default_index_physician)
        operating_physician = st.selectbox("Operating Physician", physician_label_list,index=default_index_physician)
        other_physician = st.selectbox("Other Physician", physician_label_list,index=default_index_physician)
        diagnosis_codes = [] 
        for i in range(st.session_state.input_count):
            diagnosis_code = st.selectbox(f"Diagnosis code {i + 1}",diagnosis_label_list,index=default_index_diagnosis)
            if(diagnosis_code):
              diagnosis_codes.append(diagnosis_code)

        claim_submitted = st.form_submit_button("Submit")

    encoded_attending_physician = loaded_mapping_physician[attending_physician]
    encoded_operating_physician = loaded_mapping_physician[operating_physician]
    encoded_other_physician = loaded_mapping_physician[other_physician]
    encoded_provider = loaded_mapping_provider[provider]

    encoded_diagnosis_codes = [loaded_mapping_diagnosis.get("NaN")] * 10
    new_encode = len(diagnosis_label_list) + 1 
    for i in range(min(len(diagnosis_codes), 10)):
        code = loaded_mapping_diagnosis.get(diagnosis_codes[i], new_encode)
        if code:  
          encoded_diagnosis_codes[i] = code  
        else:
          encoded_diagnosis_codes[i] = new_encode 
          new_encode += 1

    encoded_diagnosis_code_1 = encoded_diagnosis_codes[0]
    encoded_diagnosis_code_2 = encoded_diagnosis_codes[1]
    encoded_diagnosis_code_3 = encoded_diagnosis_codes[2]
    encoded_diagnosis_code_4 = encoded_diagnosis_codes[3]
    encoded_diagnosis_code_5 = encoded_diagnosis_codes[4]
    encoded_diagnosis_code_6 = encoded_diagnosis_codes[5]
    encoded_diagnosis_code_7 = encoded_diagnosis_codes[6]
    encoded_diagnosis_code_8 = encoded_diagnosis_codes[7]
    encoded_diagnosis_code_9 = encoded_diagnosis_codes[8]
    encoded_diagnosis_code_10 = encoded_diagnosis_codes[9]

    if(in_hospital == "Outpatient"):
        
        X_test = pd.DataFrame({
            'AttendingPhysician_Label_Encoded':[encoded_attending_physician], 
            'OperatingPhysician_Label_Encoded':[encoded_operating_physician],
            'OtherPhysician_Label_Encoded':[encoded_other_physician], 'ClmDiagnosisCode_1_Label_Encoded':[encoded_diagnosis_code_1],
            'ClmDiagnosisCode_2_Label_Encoded':[encoded_diagnosis_code_2], 'ClmDiagnosisCode_3_Label_Encoded':[encoded_diagnosis_code_3],
            'ClmDiagnosisCode_4_Label_Encoded':[encoded_diagnosis_code_4], 'ClmDiagnosisCode_5_Label_Encoded':[encoded_diagnosis_code_5],
            'ClmDiagnosisCode_6_Label_Encoded':[encoded_diagnosis_code_6], 'ClmDiagnosisCode_7_Label_Encoded':[encoded_diagnosis_code_7],
            'ClmDiagnosisCode_8_Label_Encoded':[encoded_diagnosis_code_8], 'ClmDiagnosisCode_9_Label_Encoded':[encoded_diagnosis_code_9],
            'ClmDiagnosisCode_10_Label_Encoded':[encoded_diagnosis_code_10],
            'Physician_group_String_Label_Encoded':[0],
            'DiagnosisCode_group_String_Label_Encoded':[0],
            'ClaimCompletedSameDay':[claim_days],
            'Provider_Label_Encoded':[encoded_provider],
            'InscClaimAmtReimbursed':[0],
        })
        if(sample == 'Sample 2'):
             X_test = pd.DataFrame({
            'AttendingPhysician_Label_Encoded':[61293], 
            'OperatingPhysician_Label_Encoded':[28532],
            'OtherPhysician_Label_Encoded':[30983], 'ClmDiagnosisCode_1_Label_Encoded':[6411],
            'ClmDiagnosisCode_2_Label_Encoded':[4475], 'ClmDiagnosisCode_3_Label_Encoded':[3918],
            'ClmDiagnosisCode_4_Label_Encoded':[3455], 'ClmDiagnosisCode_5_Label_Encoded':[2988],
            'ClmDiagnosisCode_6_Label_Encoded':[2591], 'ClmDiagnosisCode_7_Label_Encoded':[2287],
            'ClmDiagnosisCode_8_Label_Encoded':[1962], 'ClmDiagnosisCode_9_Label_Encoded':[1622],
            'ClmDiagnosisCode_10_Label_Encoded':[431],
            'Physician_group_String_Label_Encoded':[0],
            'DiagnosisCode_group_String_Label_Encoded':[0],
            'ClaimCompletedSameDay':[1],
            'Provider_Label_Encoded':[3726],
            'InscClaimAmtReimbursed':[0],
        })

        if claim_submitted:
              xgb_model = XGBClassifier()
              xgb_model.load_model("xgb_model_op.json")
              y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

              threshold = 0.5631  
              y_pred = (y_pred_prob > threshold).astype(int)
              st.write(f"The predicted probability of fraud is: {y_pred_prob[0] * 100:.2f}%")
              if y_pred[0] == 1:
                  st.error(f"There is a {y_pred_prob[0] * 100:.2f}% chance that the provider in this claim is **fraudulent**.")
              else:
                  st.success(f"There is a {y_pred_prob[0] * 100:.2f}% chance that the provider in this claim is **not fraudulent**.")

    elif(in_hospital == "Inpatient"):
        X_test = pd.DataFrame({
            'AttendingPhysician_Label_Encoded':[encoded_attending_physician], 
            'OperatingPhysician_Label_Encoded':[encoded_operating_physician],
            'OtherPhysician_Label_Encoded':[encoded_other_physician], 'ClmDiagnosisCode_1_Label_Encoded':[encoded_diagnosis_code_1],
            'ClmDiagnosisCode_2_Label_Encoded':[encoded_diagnosis_code_2], 'ClmDiagnosisCode_3_Label_Encoded':[encoded_diagnosis_code_3],
            'ClmDiagnosisCode_4_Label_Encoded':[encoded_diagnosis_code_4], 'ClmDiagnosisCode_5_Label_Encoded':[encoded_diagnosis_code_5],
            'ClmDiagnosisCode_6_Label_Encoded':[encoded_diagnosis_code_6], 'ClmDiagnosisCode_7_Label_Encoded':[encoded_diagnosis_code_7],
            'ClmDiagnosisCode_8_Label_Encoded':[encoded_diagnosis_code_8], 'ClmDiagnosisCode_9_Label_Encoded':[encoded_diagnosis_code_9],
            'ClmDiagnosisCode_10_Label_Encoded':[encoded_diagnosis_code_10],
            'Physician_group_String_Label_Encoded':[0],
            'DiagnosisCode_group_String_Label_Encoded':[0],
            'TimeforCLAIM':[claim_days],
            'Provider_Label_Encoded':[encoded_provider],
            'InscClaimAmtReimbursed':[0],
        })
        if(sample == 'Sample 1'):
             X_test = pd.DataFrame({
                'AttendingPhysician_Label_Encoded':[2228], 
                'OperatingPhysician_Label_Encoded':[559],
                'OtherPhysician_Label_Encoded':[2368], 'ClmDiagnosisCode_1_Label_Encoded':[1230],
                'ClmDiagnosisCode_2_Label_Encoded':[338], 'ClmDiagnosisCode_3_Label_Encoded':[1193],
                'ClmDiagnosisCode_4_Label_Encoded':[1213], 'ClmDiagnosisCode_5_Label_Encoded':[2203],
                'ClmDiagnosisCode_6_Label_Encoded':[2186], 'ClmDiagnosisCode_7_Label_Encoded':[2142],
                'ClmDiagnosisCode_8_Label_Encoded':[2093], 'ClmDiagnosisCode_9_Label_Encoded':[1950],
                'ClmDiagnosisCode_10_Label_Encoded':[852],
                'Physician_group_String_Label_Encoded':[0],
                'DiagnosisCode_group_String_Label_Encoded':[0],
                'TimeforCLAIM':[1],
                'Provider_Label_Encoded':[972],
                'InscClaimAmtReimbursed':[0],
            })
             
        if claim_submitted:
              xgb_model = XGBClassifier()
              xgb_model.load_model("xgb_inpatient_model.json")
              y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

              threshold = 0.4540  
              y_pred = (y_pred_prob > threshold).astype(int)
              st.write(f"The predicted probability of fraud is: {y_pred_prob[0] * 100:.2f}%")
              if y_pred[0] == 1:
                  st.error(f"There is a {y_pred_prob[0] * 100:.2f}% chance that the provider in this claim is **fraudulent**.")
              else:
                  st.success(f"There is a {y_pred_prob[0] * 100:.2f}% chance that the provider in this claim is **not fraudulent**.")

            