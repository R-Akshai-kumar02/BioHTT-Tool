#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit import DataStructs


# In[2]:


import streamlit as st


# In[3]:


from joblib import load

model = load("htt_random_forest_model.pkl")
mod = load("htt_ic50_regressor.pkl")


# In[8]:


st.markdown(
    """
    <h1 style="text-align:center;">
        BioHTT
    </h1>
    """,
    unsafe_allow_html=True
)
st.subheader("About the Tool")
st.text("BioHTT (Bioactivity Prediction for HTT Protein) is a machine learning based tool that predicts whether small molecules are likely to be active or inactive against the Huntingtin (HTT) protein. The model uses Morgan fingerprints (ECFP4, 2048 bits) as molecular features and achieved an accuracy of 91%. Huntington's disease (HD) is a progressive neurodegenerative disorder caused by mutations in the Huntingtin (HTT) gene. The mutant HTT protein is a key therapeutic target for the development of new treatments.")

st.subheader("Input")
smiles = st.text_input("Enter Smiles")
def smiles_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    fp= AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits= 2048)
    return np.array(fp)

if st.button("Predict"):
    fp = smiles_fp(smiles)
    if fp is None:
        st.warning("Invalid Smiles")
    else:
        pred=model.predict(fp.reshape(1, -1))[0]
        pred_prob = model.predict_proba(fp.reshape(1,-1))[0]
        inactive_prob = pred_prob[0] * 100
        active_prob = pred_prob[1] * 100
        predicted_pchembl = mod.predict(fp.reshape(1, -1))[0]
        predicted_ic50 = 10 ** (9 - predicted_pchembl)
        st.subheader("Output") 
        if pred == 1:
            st.success("Active compound")

        else:
            st.warning("Inactive compound")

        st.subheader("Prediction Probability")

        st.write(f"**Active:** {active_prob:.2f}%")
        st.progress(active_prob / 100)

        st.write(f"**Inactive:** {inactive_prob:.2f}%")
        st.progress(inactive_prob / 100)
        st.subheader("Predicted Potency")
        st.write(f"**Predicted pChEMBL:** {predicted_pchembl:.2f}")
        st.write(f"**Estimated IC50:** {predicted_ic50:.2f} nM")




# In[ ]:




