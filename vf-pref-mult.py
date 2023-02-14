import streamlit as st
# other libs
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from streamlit_tags import st_tags,st_tags_sidebar
from random import randint, random
import datetime
import subprocess
import sys

from Bio.Blast.Applications import NcbiblastpCommandline

from sklearn.ensemble import GradientBoostingClassifier

#!pip install Bio
#!pip install iFeature
#!sudo apt-get install emboss
#!git clone https://github.com/Superzchen/iFeature

import base64
from email.message import EmailMessage


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os

from Bio.Blast.Applications import NcbiblastpCommandline

import os

import re
import time
import numpy
import math
from Bio.Emboss.Applications import NeedleCommandline


def blast_sim(df):
  os.system('makeblastdb -in train_negative.fasta -dbtype prot')
  os.system('makeblastdb -in train_positive.fasta -dbtype prot')
  from Bio.Blast.Applications import NcbiblastpCommandline
  cline = NcbiblastpCommandline(query="seq.fasta", db="train_positive.fasta",
                              evalue=10, remote=False)
  out_data, err = cline()
  out_split = out_data.split("\n")
  print(out_split[29][70:75])
  print("rownum"  + "positive")
  if (len(out_split[29]) > 0 and out_split[29][0:35].replace(" ",'') == out_split[23].replace("Query= ",'').replace(" ",'') ):
      if len(out_split[30]) > 0: 
        df['seq_pos'] = float(out_split[30][70:75])
      else:
        df_seq_tr_pos['seq_pos'] = 0    
  elif(len(out_split[29]) > 0 and out_split[29][0:35].replace(" ",'') != out_split[23].replace("Query= ",'').replace(" ",'') ):
     df['seq_pos'] = float(out_split[29][70:75])
  else:
      df['seq_pos'] = 0

  cline = NcbiblastpCommandline(query="seq.fasta", db="train_negative.fasta",
                              evalue=10, remote=False)
  out_data, err = cline()
  out_split = out_data.split("\n")
  print(out_split[29][70:75])
  print("rownum"  + "negative")
  if (len(out_split[29]) > 0 and out_split[29][0:35].replace(" ",'') == out_split[23].replace("Query= ",'').replace(" ",'') ):
      if len(out_split[30]) > 0: 
        df['seq_neg'] = float(out_split[30][70:75])
      else:
        df['seq_neg'] = 0    
  elif(len(out_split[29]) > 0 and out_split[29][0:35].replace(" ",'') != out_split[23].replace("Query= ",'').replace(" ",'') ):
    df['seq_neg'] = float(out_split[29][70:75])
  else:
    df['seq_neg'] = 0

  return(df)


####https://www.biostars.org/p/208540/

def needle_align(query_seq, target_seq):
        print(query_seq)
        print(target_seq)
        needle_cline = NeedleCommandline(asequence="asis:" + query_seq,
                                        bsequence="asis:" + target_seq,
                                        aformat="simple",
                                        gapopen=10,
                                        gapextend=0.5,
                                        outfile='stdout'
                                        )
        out_data, err = needle_cline()
        out_split = out_data.split("\n")
        p = re.compile("\((.*)\)")
        return p.search(out_split[25]).group(1).replace("%", "")

def align_per(protein_seq, df):

  max_match = 0
  for i in range(len(df)):
    sim = float(needle_align(protein_seq,df.iloc[i]))
    #print(protein_seq, df.iloc[i], sim)
    if max_match < sim:  
      max_match = sim
  #print(max_match)
  return max_match

def align_similarity(df):
    df_seq_tr_pos = pd.read_csv('train_postive.csv', sep="\t")
    df_seq_tr_neg = pd.read_csv('train_negative.csv', sep="\t")
    df_seq_tr_pos_tmp = df_seq_tr_pos[1:1000]
    df_seq_tr_neg_tmp = df_seq_tr_neg[1:1000]
    remove = df_seq_tr_pos_tmp['seq'].isin([df['seq']])
    df_seq_tr_pos_tmp = df_seq_tr_pos_tmp[~remove]
    remove = df_seq_tr_neg_tmp['seq'].isin([df['seq']])
    df_seq_tr_neg_tmp = df_seq_tr_neg_tmp[~remove]
    print(df)
    start = time.time()
    print(start)
    df['seqsim_pos'] = align_per(df.iloc[0]['seq'],df_seq_tr_pos_tmp['seq'] )
    df['seqsim_neg'] = align_per(df.iloc[0]['seq'],df_seq_tr_neg_tmp['seq'] )

    end = time.time()
    print(end)

    return (df)

def input_fastafile(pro_id, pro_seq):
  ofile = open("seq.fasta", "w")
  ofile.write(">" + pro_id + "\n" + pro_seq + "\n")
  df_input_seq = pd.DataFrame({'identifiers': pro_id,'seq': pro_seq}, index=[0])
  ofile.close()  
  return (df_input_seq)

def pred(df_X):
  import pickle
  global in_df, df_majority, X, colname
  in_df = df_X
  dyn_featurelist = ['AAC','DDE','DPC','PAAC','QSO']
  models_considered =['SVM_clf','XGB_clf','MLP_clf','RF_clf']##,'gb_model']
  df_out = pd.DataFrame( )
  sim_featurelist = ['seqsim_pos','seqsim_neg','seq_pos','seq_neg']

  default = ['Target','seq']
  df_majority = pd.DataFrame( )

  for i in range(len(dyn_featurelist)):
    global colused
    colused =[]
    print(dyn_featurelist[i])
    stra = "colused = [col for col in in_df  if col.startswith('"+ dyn_featurelist[i]+ "_')]"
    exec(stra,globals()  )

    colused = sim_featurelist + colused
    X = in_df[colused]
    X = X.loc[:,~X.columns.isin(list(X.filter(like='#').columns))]
    
    print(X.columns)
    for model in models_considered:
      pickle_folder = "Pickle//"
      pickle_file = pickle_folder + model+ dyn_featurelist[i] +"_pkl"
      print(pickle_file)
      # load saved model
      with open(pickle_file , 'rb') as f:
        lr = pickle.load(f)

      Y_Pred = lr.predict(X)
      
      colname = model+"_" +dyn_featurelist[i]
      print("works "+ colname)
      df_majority[colname] = Y_Pred
  print(df_majority)
  mode = df_majority.mode(axis=1).iloc[0,0]
  print("mode value is"+str(mode))
  return mode



def gen_features(df_seq_tr):
 # os.system('python  iFeature/iFeature.py --file seq.fasta --type AAC --out temp/AAC_tr.txt')
 # os.system('python  iFeature/iFeature.py --file seq.fasta --type DPC --out temp/DPC_tr.txt')
 # os.system('python  iFeature/iFeature.py --file seq.fasta --type DDE --out temp/DDE_tr.txt')

 # os.system('python  iFeature/iFeature.py --file seq.fasta --type PAAC --out temp/PAAC_tr.txt')
 # os.system('python  iFeature/iFeature.py --file seq.fasta --type QSOrder --out temp/QSOrder_tr.txt')
  #AAC_tr = open('temp/AAC_tr.txt', 'w')
  #DPC_tr = open('temp/DPC_tr.txt', 'w')
  #DDE_tr = open('temp/DDE_tr.txt', 'w')
  #PAAC_tr = open('temp/PAAC_tr.txt', 'w')
  #QSOrder_tr = open('temp/QSOrder_tr.txt', 'w')


  #subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta ", "--type", "AAC"], stdout=AAC_tr)
  #subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", " --file", "seq.fasta", " --type", "DPC"], stdout=DPC_tr)
  #subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", " --file", "seq.fasta ", "--type", "DDE"], stdout=DDE_tr)
  #subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", " --file ", "seq.fasta ", "--type", "PAAC"], stdout=PAAC_tr)
  #subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", " --file", "seq.fasta", " --type", "QSOrder"], stdout=QSOrder_tr)

  subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta", "--type", "AAC", "--out", 'temp/AAC_tr.txt'  ])
  subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta", "--type", "DPC", "--out", 'temp/DPC_tr.txt'] )
  subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta", "--type", "DDE", "--out", 'temp/DDE_tr.txt'] )
  subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta", "--type", "PAAC", "--out", 'temp/PAAC_tr.txt'] )
  subprocess.run([f"{sys.executable}", "iFeature/iFeature.py", "--file", "seq.fasta", "--type", "QSOrder", "--out", 'temp/QSOrder_tr.txt'] )

  df_AAC_tr = pd.read_csv('temp/AAC_tr.txt', sep="\t")
  df_AAC_tr.columns = 'AAC_' + df_AAC_tr.columns
  df_DPC_tr = pd.read_csv('temp/DPC_tr.txt', sep="\t")
  df_DPC_tr.columns = 'DPC_' + df_DPC_tr.columns
  df_DDE_tr  = pd.read_csv('temp/DDE_tr.txt', sep="\t")
  df_DDE_tr.columns = 'DDE_' + df_DDE_tr.columns

  df_PAAC_tr = pd.read_csv('temp/PAAC_tr.txt', sep="\t")
  df_PAAC_tr.columns = 'PAAC_' + df_PAAC_tr.columns
  df_QSO_tr = pd.read_csv('temp/QSOrder_tr.txt', sep="\t")
  df_QSO_tr.columns = 'QSO_' + df_QSO_tr.columns

  df_seq_tr = pd.merge(df_seq_tr, df_AAC_tr, how='inner', left_on='identifiers', right_on='AAC_#')
  df_seq_tr = pd.merge(df_seq_tr, df_DDE_tr, how='inner', left_on='identifiers', right_on='DDE_#')
  df_seq_tr = pd.merge(df_seq_tr, df_DPC_tr, how='inner', left_on='identifiers', right_on='DPC_#')
  df_seq_tr = pd.merge(df_seq_tr, df_PAAC_tr, how='inner', left_on='identifiers', right_on='PAAC_#')
  df_seq_tr = pd.merge(df_seq_tr, df_QSO_tr, how='inner', left_on='identifiers', right_on='QSO_#')
  return (df_seq_tr)

# -- Set page config
apptitle = 'VF Predictor'

st.set_page_config(page_title=apptitle, page_icon='random', layout= 'wide', initial_sidebar_state="expanded")
# random icons in the browser tab


def create_email_token():
  SCOPES = ['https://www.googleapis.com/auth/gmail.send']
  creds = None

  

  
  if os.path.exists(tmpPath+'pentagon_token.json'):
    print("file exists")
    creds = Credentials.from_authorized_user_file(tmpPath+'pentagon_token.json', SCOPES)
    
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            tmpPath+'pentagon_keys.json', SCOPES)
        creds = flow.run_local_server(port=60088)
    # Save the credentials for the next run
    with open(tmpPath+'pentagon_token.json', 'w') as token:
        token.write(creds.to_json())    

def sendemail(message_content):
  SCOPES = ['https://www.googleapis.com/auth/gmail.send']
  creds = None
    
    
  if os.path.exists(tmpPath+'pentagon_token.json'):
    print("file exists")
    creds = Credentials.from_authorized_user_file(tmpPath+'pentagon_token.json', SCOPES)
  
    #creds, _ = google.auth.default()

    try:
        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()
        message_content=message_content.replace("\\n","")
        message_content=message_content.replace("\#","")
        message_content=message_content.replace(",","")
        message_content=message_content.replace("\'","")
        message_content=message_content.replace("]","")
        message_content=message_content.replace("[","")
        message.set_content(message_content)
        xtime = datetime.datetime.now()

        message['To'] = ''
        message['From'] = 'capstone.pentagon.iss@gmail.com'
        message['Subject'] = 'Protein Sequence Prediction of Virulence Factor id-'+ str(xtime)

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()) \
            .decode()

        create_message = {
            'raw': encoded_message
        }
        # pylint: disable=E1101
        send_message = (service.users().messages().send
                        (userId="me", body=create_message).execute())
        print(F'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message

def main():

  primaryColor="#F63366"
  backgroundColor="#FFFFFF"
  font="sans serif"

  
  global tmpPath
  email_message_1 =""
  email_message_2 =""
  email_message_3= ""
  tmpPath="C:/Users/shreya/Desktop/Shreya/Masters/Capstone/streamlit/temp/"
  
  st.title('Classification Model to predict Virulence Factor')
  st.balloons() 
  # st.balloons() 
  # st.balloons() 

  st.markdown('<p style="font-family:sans serif; color:Black; font-size: 18px;">Virulence factors (VFs) enable pathogens to infect their hosts. The spread of infectious diseases caused by pathogenic bacteria is a major cause of human and animal mortality, and predictions for how dire this problem will grow due to increases in drug-resistant bacteria are being revised upwards. The pathogenesis associated with a bacterial infection can be complex but, in many cases, is primarily driven by virulence factors (VFs): proteins produced by the bacterium that enable it to persist, grow and do damage to the tissues of its human or animal host', unsafe_allow_html=True)
  st.text("")

  # Let's add a sub-title
  st.markdown('<p style="font-family:sans serif; color:Blue; font-size: 25px;">Check Virulence Factor for your protein(max-10)</p>', unsafe_allow_html=True)
 
  with st.form(key='columns_in_form'):
    c1= st.columns(1)
  #pro_id = st.text_input('Enter the Protein Identifier','0|1|gi|218561911|ref|YP_002343690.1|')
    pro_id_in = st_tags(
    label='# Enter Protein Identifier:',
    text='Press enter to add more',
    value=[],
    maxtags = 10,
    key='1')
    pro_seq_in = st_tags(
    label='# Enter the Protein Sequence:',
    text='Press enter to add more',
    value=[],
    maxtags = 10,
    key='2')
    submitButton = st.form_submit_button(label = 'SUBMIT')
    if submitButton:
      print(pro_id_in)
      print(pro_seq_in)
      temp_msg=[]
      for i in range(len(pro_seq_in)):
        pro_id=pro_id_in[i]
        pro_seq=pro_seq_in[i]
        df_input_seq = input_fastafile(pro_id, pro_seq)
        df_feature= gen_features(df_input_seq)
        df_feature = align_similarity(df_feature)
        df_feature = blast_sim(df_feature)
        print(df_feature.columns)
        pred_output = pred(df_feature)
        print(type(pred_output))
        print("The final output" + str(pred_output))
        outprint = "low" if pred_output == 0 else "high"
        print(str(pred_output)+"-" + outprint)
        st.markdown("""
        #### This protein id {pro_id_print} has {temp} probability of Virulence Factors.
        # #""".format(temp=outprint,pro_id_print=pro_id))
        temp_msg.append("""
        
        {n}) The protein sequence entered was {pro_seq_print} 
        This protein sequence has {temp} probability of VF.   

        """.format(temp=outprint,pro_seq_print=pro_seq,n=i+1))
        print(temp_msg[i])
if __name__ == '__main__':
	main()