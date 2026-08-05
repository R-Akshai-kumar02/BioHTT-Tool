#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs


# In[5]:


import pandas as pd


# In[2]:


import streamlit as st


# In[3]:


from joblib import load

model = load("htt_random_forest_model.pkl")


# In[4]:


def calculate_qsar(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atoms": Descriptors.HeavyAtomCount(mol),
        "Ring Count": Descriptors.RingCount(mol),
        "Fraction Csp3": round(Descriptors.FractionCSP3(mol), 2),
        "Molar Refractivity": round(Descriptors.MolMR(mol), 2)
    }


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
st.text("BioHTT (Bioactivity Prediction for HTT Protein) is a machine learning based tool that predicts whether small molecules are likely to be active or inactive against the Huntingtin (HTT) protein while also calculating their key QSAR (Quantitative Structure–Activity Relationship) properties. The model uses Morgan fingerprints (ECFP4, 2048 bits) as molecular features and achieved an accuracy of 91%. Huntington's disease (HD) is a progressive neurodegenerative disorder caused by mutations in the Huntingtin (HTT) gene. The mutant HTT protein is a key therapeutic target for the development of new treatments.")

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

        qsar = calculate_qsar(smiles)



        if qsar is not None:

            st.subheader("QSAR Properties")

            qsar_df = pd.DataFrame(
                qsar.items(),
                columns=["Property", "Value"]
            )

            st.table(qsar_df)



# In[ ]:




