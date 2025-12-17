import pandas as pd
import numpy as np
import csv
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
import re
import pickle
import warnings
warnings.filterwarnings("ignore")
import torch

########################################################3-mer embedding########################################################
# define 3-mer
all_3mers = [
    "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT",
    "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT",
    "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT",
    "GAA", "GAC", "GAG", "GAT", "GCA", "GCC", "GCG", "GCT",
    "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT",
    "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT"
]
#sequence to 3mer embedding
def get_3mer_feature_vector(sequence):
    kmer_count = defaultdict(int)
    k = 3
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmer_count[kmer] += 1
    frequency_vector = [kmer_count[kmer] / (len(sequence) - 2) for kmer in all_3mers] #freq
    return frequency_vector


########################################################For LLM embedding########################################################
#prepare input for RNAErnie
def generate_input_for_llm1(infile='CrossHost_preMatrix_ORF.csv'):
    df = pd.read_csv(infile)
    fasta_filelist=[]
    with open('llm1.fasta', 'w') as fasta_file:
        for index, row in df.iterrows():
            fasta_file.write(f">{row['ID']}__Seq\n{row['Seq']}\n")
            fasta_file.write(f">{row['ID']}__Spike\n{row['Spike']}\n")
            fasta_file.write(f">{row['ID']}__Nucleocapsid\n{row['Nucleocapsid']}\n")
            fasta_file.write(f">{row['ID']}__Membrane\n{row['Membrane']}\n")
            fasta_filelist.extend([f"{row['ID']}__Seq.npy", f"{row['ID']}__Spike.npy", f"{row['ID']}__Nucleocapsid.npy", f"{row['ID']}__Membrane.npy"])
    print(len(fasta_filelist))
    with open('CrossHost_LLM_fasta_filelist1.txt', 'w') as list_file:
        for filename in fasta_filelist:
            list_file.write(f"{filename}\n")
    print("FASTA done")

#read output from RNAErnie, save the dict
def analyze_llm1_npy_for_dict(dir_path,filelist_path):
    vecdict = {}
    with open(filelist_path, 'r') as file:
        filenames = file.readlines()
    filenames = [filename.strip() for filename in filenames]
    # read npy, save to dict
    for filename in filenames:
        npy_path = f"{dir_path}/{filename}"
        try:
            data = np.load(npy_path)
            vecdict[filename[:-4]] = data.tolist()
            print(len(data.tolist()))
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
    print('Embedding file num:',len(filenames))
    print('Embedding file num UNIQUE:',len(vecdict),len(set(filenames)))
    with open('CrossHost_llm1_vecdict.pkl', 'wb') as f:
        pickle.dump(vecdict, f)
    print("Done for llm1 vecdict.pkl")

#LucaOne-Not used any more
def analyze_llm2_npy_for_dict(dir_path,filelist_path):
    vecdict = {}
    with open(filelist_path, 'r') as file:
        filenames = file.readlines()
    filenames = [
        'vector_' + filename.strip().replace('.npy', '.pt')
        for filename in filenames
    ]
    for filename in filenames:
        pt_path = f"{dir_path}/{filename}"
        try:
            tensor = torch.load(pt_path)
            tensor_vector = tensor.flatten()
            vecdict[filename[7:-3]] = tensor_vector
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
    print('Embedding file num:',len(filenames))
    print('Embedding file num UNIQUE:',len(vecdict),len(set(filenames)))
    with open('CrossHost_llm2_vecdict.pkl', 'wb') as f:
        pickle.dump(vecdict, f)
    print("Done for llm2 vecdict.pkl")


########################################################Risk Model Training########################################################
#Main Core
def train_and_predict(filepath,model,model_name):
    print(model_name,'Doing...')
    df = pd.read_csv(filepath, na_values=["nan"])

    with open('CrossHost_llm1_vecdict.pkl', 'rb') as f:
        vecdict1 = pickle.load(f)
    def get_feature_llm1(row, seq_col):
        key = f"{row['ID']}__{seq_col}"
        return vecdict1.get(key, [])  # return RNAErnie embedding
    # with open('CrossHost_llm2_vecdict.pkl', 'rb') as f:
    #     vecdict2 = pickle.load(f)
    # def get_feature_llm2(row, seq_col):
    #     key = f"{row['ID']}__{seq_col}"
    #     return vecdict2.get(key, [])  # return LucaOne embedding - No use
    df['feature'] = df.apply(lambda row: np.concatenate([get_3mer_feature_vector(row['Spike']) ,get_feature_llm1(row,'Spike')]),axis=1)

    #training
    train_df = df[df['Risk'].notna()] #only sequences with R=0 or 1 are utilized for training
    train_df = shuffle(train_df, random_state=random_state_seed)
    X_train = list(train_df['feature'])  # feature
    y_train = train_df['Risk']  # R
    model.fit(X_train, y_train)

    #predict the Risk of sequences with R = unknown
    predict_df = df[df['Risk'].isna()]
    X_predict = list(predict_df['feature'])
    predicted_risk = model.predict(X_predict)

    df = df[df['Risk'].isna()]
    df.loc[df['Risk'].isna(), 'Risk'] = predicted_risk

    df_sorted = df.sort_values(by='Risk', ascending=False)
    df_sorted = df_sorted.drop(columns=['feature','Seq','feature1','feature2'], errors='ignore')
    df_sorted=df_sorted.drop_duplicates()
    df_sorted.to_csv('02_Predict_Risk.csv', index=False)
    print(model_name,'Done')



random_state_seed=42
if __name__ == "__main__":
    #llm1 IO - RNAErnie
    # src：https://kkgithub.com/CatIIIIIIII/RNAErnie/tree/v1.0
    # [NMI]: https://www.nature.com/articles/s42256-024-00836-4
    generate_input_for_llm1(infile='CrossHost_preMatrix_ORF.csv') #prepare input for RNAErnie
    analyze_llm1_npy_for_dict(dir_path='./llm1',filelist_path='CrossHost_LLM_fasta_filelist1.txt') #read output from RNAErnie, save the dict  #vec length=768

    #llm2 IO - LucaOne, NOT USED ANY MORE
    #[NMI]:https://www.nature.com/articles/s42256-025-01044-4
    # https://mp.weixin.qq.com/s/nD7yCtmr9A1oVzyB5JZicA
    ## analyze_llm2_npy_for_dict(dir_path='./llm2',filelist_path='CrossHost_LLM_fasta_filelist2.txt')

    #[Main Core] Model training and prediction
    model, model_name = ensemble.RandomForestRegressor(),'RF_conc'
    train_and_predict(filepath='CrossHost_preMatrix_ORF.csv',model=model,model_name=model_name)









