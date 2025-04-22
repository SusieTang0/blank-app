import pandas as pd
import seaborn as sns
import numpy as np

import math

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scipy.special import inv_boxcox
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import precision_recall_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve


df_op = pd.read_csv('./train_outpatient_encoded.csv')
df_ip = pd.read_csv('./train_inpatient_encoded.csv')


X_op = df_op[[ 'AttendingPhysician_Label_Encoded', 'OperatingPhysician_Label_Encoded',
       'OtherPhysician_Label_Encoded', 'ClmDiagnosisCode_1_Label_Encoded',
       'ClmDiagnosisCode_2_Label_Encoded', 'ClmDiagnosisCode_3_Label_Encoded',
       'ClmDiagnosisCode_4_Label_Encoded', 'ClmDiagnosisCode_5_Label_Encoded',
       'ClmDiagnosisCode_6_Label_Encoded', 'ClmDiagnosisCode_7_Label_Encoded',
       'ClmDiagnosisCode_8_Label_Encoded', 'ClmDiagnosisCode_9_Label_Encoded',
       'ClmDiagnosisCode_10_Label_Encoded',
       'Physician_group_String_Label_Encoded',
       'DiagnosisCode_group_String_Label_Encoded','ClaimCompletedSameDay',
       'Provider_Label_Encoded','InscClaimAmtReimbursed']]
y_op = df_op['PotentialFraud']  # Target variable (1: Fraud, 0: Legitimate)
X_train_op, X_test_op, y_train_op, y_test_op = train_test_split(X_op, y_op, test_size=0.2, random_state=42)

X_ip = df_ip[[ 'AttendingPhysician_Label_Encoded', 'OperatingPhysician_Label_Encoded',
       'OtherPhysician_Label_Encoded','ClmDiagnosisCode_1_Label_Encoded',
       'ClmDiagnosisCode_2_Label_Encoded', 'ClmDiagnosisCode_3_Label_Encoded',
       'ClmDiagnosisCode_4_Label_Encoded', 'ClmDiagnosisCode_5_Label_Encoded',
       'ClmDiagnosisCode_6_Label_Encoded', 'ClmDiagnosisCode_7_Label_Encoded',
       'ClmDiagnosisCode_8_Label_Encoded', 'ClmDiagnosisCode_9_Label_Encoded',
       'ClmDiagnosisCode_10_Label_Encoded','Physician_group_String_Label_Encoded',
       'DiagnosisCode_group_String_Label_Encoded','TimeforCLAIM',
       'Provider_Label_Encoded','InscClaimAmtReimbursed',]]
y_ip = df_ip['PotentialFraud']  # Target variable (1: Fraud, 0: Legitimate)
X_train_ip, X_test_ip, y_train_ip, y_test_ip = train_test_split(X_ip, y_ip, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
scale_op = len(y_train_op[y_train_op == 0]) / len(y_train_op[y_train_op == 1])

xgb_model_op = XGBClassifier(scale_pos_weight=scale_op, eval_metric='logloss')
xgb_model_op.fit(X_train_op, y_train_op)

xgb_preds_op = xgb_model_op.predict_proba(X_test_op)[:, 1]



scale_ip = len(y_train_ip[y_train_ip == 0]) / len(y_train_ip[y_train_ip == 1])

xgb_model_ip = XGBClassifier(scale_pos_weight=scale_ip, eval_metric='logloss')
xgb_model_ip.fit(X_train_ip, y_train_ip)

xgb_preds_ip = xgb_model_ip.predict_proba(X_test_ip)[:, 1]


xgb_model_ip.save_model("xgb_model_ip.json")
xgb_model_op.save_model("xgb_model_op.json")