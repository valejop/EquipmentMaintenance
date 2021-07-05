# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:14:56 2020

@author: lumini_its02
"""
import pandas as pd
import matplotlib_venn as venn
import numpy as np 
import datetime as dt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sb
import os
import ipywidgets as widgets
import IPython.display as display
import itertools as it
import boto3 as b3
import anytree

from io import StringIO

def read_and_treat(nome_string,delimit='|',nrows=None,indexcol=None):
    df=pd.read_csv(nome_string,delimiter=delimit,nrows=nrows,index_col=indexcol)
    print('Tabela importada')
    df=limpeza_whitespace_df(df)
    print('Tabela Limpa')
    DT=list(df.columns[df.columns.str.startswith('DT')])
    if 'DT_NASCIMENTO' in DT: 
        DT.remove('DT_NASCIMENTO')
        df['DT_NASCIMENTO']=pd.to_datetime(df['DT_NASCIMENTO'])
    for i in DT:
        df[i]=series_to_date2(df[i],Upr_Limit=pd.Timestamp.today(),Lwr_Limit=dt.date(1990,1,1))
        df[i]=pd.to_datetime(df[i])
        print(f'Coluna {i} transformada')
    return(df)

def woe_iv(dataframe,var,target,limit=100,n_bins=20,inf_deal=True):
#Caso as variáveis sejam contínuas
    if len(dataframe[var].unique())>limit:
        if dataframe.dtypes[var]=='int64' or dataframe.dtypes[var]=='float64':
            df2=dataframe[[var,target]]
            df2[var]=pd.qcut(dataframe[var],n_bins,duplicates='drop')
            return(woe_iv(df2,var,target))
        if dataframe.dtypes[var]=='<M8[ns]':
            df2=dataframe[[var,target]]
            df2[var]=pd.qcut(df2[var],n_bins,duplicates='drop')
            return(woe_iv(df2,var,target))
#Else:
    table=dataframe.groupby([var,target]).size().unstack().fillna(0)
    temp=table
    for j in range(table.shape[1]):
        soma=sum(table.iloc[:,j])
        for i in range(table.shape[0]):
            temp.iloc[i,j]=table.iloc[i,j]/soma
    if len(dataframe[var])>limit:
        notwoe=[]
        for i in temp.index:
            if(temp.loc[i,:]!=0).sum()<2:
                notwoe.append(i)
        temp=temp.drop(notwoe)
        print(table.shape)
    woe=list()
    iv=list()
    for i in range(0,temp.shape[0]):
        woe.append(np.log(temp.iloc[i,0]/temp.iloc[i,1]))
        if inf_deal:
            if np.abs(np.log(temp.iloc[i,0]/temp.iloc[i,1]))==np.Inf:
                iv.append(0)
            else:
                iv.append(np.log(temp.iloc[i,0]/temp.iloc[i,1])*(temp.iloc[i,0]-temp.iloc[i,1]))
        else:
            iv.append(np.log(temp.iloc[i,0]/temp.iloc[i,1])*(temp.iloc[i,0]-temp.iloc[i,1]))            
    iv_total=sum(iv)
    table=pd.DataFrame({'WOE':woe,'IV':iv})
    table.index=[temp.index.name+'_'+str(i) for i in temp.index]
    return([table,iv_total])

def series_to_date2(x,Upr_Limit=dt.date.today(),Lwr_Limit=dt.date(1900,1,2)):
    Y=x
    if Y.dtypes=='O':
        hoje=str(Upr_Limit)
        Data_Limite=str(Lwr_Limit)
        temp=list()
        for i in Y.unique():
            if str(i)>hoje or str(i)<Data_Limite:
                temp.append(i)
        for i in temp:
            Y[Y==i]=np.nan
    return(Y)

#Verifica o quanto as seções são semelhantes.
def intersection(serieA,serieB,labels=('serie 1','serie 2')):
#    serieA=pd.Series(serie1)
#    serieB=pd.Series(serie2)
    A_notB=sum(~serieA.isin(serieB))
    B_notA=sum(~serieB.isin(serieA))
    A_and_B=sum(serieA.isin(serieB))
    venn.venn2(subsets=(A_notB,B_notA,A_and_B),set_labels=tuple(labels))

#Transforma string em timestamp. Caso o dado seja maior do que o limite, converte.
def to_date(x,i=str(0)):
    if type(x[x.notna()].iloc[0])==str:
        x=x.fillna('2500-12-31')
        if(any(x.str.startswith('2500'))):
            x[x.str.startswith('2500')]=pd.Timestamp(1970,1,1)
        return(pd.to_datetime(x))
    else:
        print(f'Erro: coluna {i}')
        return(x)
    
#Função retorna colunas com capacidade de se tornarem chaves primárias da tabela
def pk_candidates(df,max_keys=3,até_a_morte=False,max_rows=-1):
    nrow=df.shape[0]
    candidates=list()
    if max_rows>0:
        df=df.iloc[0:max_rows]
    for i in df.columns:
        if df[i].unique().shape[0]==nrow:
            candidates.append(i)
    if len(candidates)==0 or até_a_morte==True:
        print('Foi necessário mais de uma coluna')
        i=2
        while i<=max_keys:
            #Enquanto estiver vazio e não for para ir até o fim, continuamos
            #Fazemos as combinações de todos as colunas com n colunas
            #Guardamos em list temporaria
            temp=list()
            temp.append(list(it.combinations(df.columns,r=i)))
            temp=list(it.chain.from_iterable(temp))
            #Checamos se existem valores duplicados
            #Se não, adicionamos a candidatos
            for j in temp:
                if ~(df[list(j)].duplicated().any()):
                    candidates.append(j)
            #Se candidatos, ao fim da iteração, estiver preenchido e não irmos até o fim, break
            if len(candidates)>0 and not até_a_morte:
                return(candidates)
                break
            i=i+1
    return(candidates)

#Após novo merge, checa se valores NA existem nas colunas unidas.
def check_new_na(df,cols,suffix):
    lista_novos=list(cols)+[i+suffix for i in list(cols)]
    for i in lista_novos:
        if i in df.columns:
            print(f'Vazios na coluna {i}:\t{any(df[i].isna())}')

#Elimina whitespace de todas as colunas que são strings 
def limpeza_whitespace_df(df,verbose=False):
    for i in range(0,df.shape[1]):
        if df.dtypes[i]=='O':
            if verbose==True:
                print(i)
            df[df.columns[i]]=df[df.columns[i]].astype(str).str.lstrip().str.rstrip()
    return(df)

def table(dataframe,var1,var2,normalize=False):
    temp=dataframe[[var1,var2]].groupby([var1,var2]).size().unstack().fillna(0)
    if(normalize==False):
        return temp
    elif(normalize==0 or normalize=='0' or normalize=='row'):
        return temp.divide(temp.sum(axis=1),axis=0)
    elif(normalize==1 or normalize=='1' or normalize=='col'):
        return temp.divide(temp.sum(axis=0),axis=1)
    else:
        return temp/dataframe.shape[0]
    
# Matriz de confusão persolnalizada
def confusion_matrix_custom(y_pred,y_test,digits=2,main='Matriz de confusão',xlab='Valores previstos',ylab='Valores reais'):
    mc=metrics.confusion_matrix(y_pred=y_pred,y_true=y_test)
    mc_pr=mc/y_pred.shape[0]*100
    mc_pr=np.round(mc_pr,digits)
    #String
    mc=mc.astype(str)
    mc_pr=mc_pr.astype(str)
    mc_out=mc
    #Unindo os 2 casos
    for i in range(0,mc.shape[0]):
        for j in range(0,mc.shape[1]):
            mc_out[i][j]=mc[i][j]+'\n('+mc_pr[i][j]+'%)'
    #Fazendo matriz
    plt.figure(figsize=(10,8))
    sb.heatmap(metrics.confusion_matrix(y_pred=y_pred,y_true=y_test),
           annot=np.array(mc_out),fmt='',cmap='Blues',annot_kws={'size':16})
    plt.title(main)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

#Abre arquivo e lê salva cada linha em lista
def ler_vars(arquivo):
    file=open(arquivo,'r')
    nom=file.readlines()
    file.close()
    nom=[i.replace('\n','') for i in nom]
    return(nom)

#Lê diferença de tempo entre colunas de datas e retorna diferença em anos, meses, semanas ou dias
def diff_time(dates1,dates2,period='y'):
    diff_year=dates1.dt.year-dates2.dt.year
    diff_month=dates1.dt.month-dates2.dt.month
    diff_day=dates1.dt.day-dates2.dt.day
    if period=='y':
        diff=diff_year-np.logical_or(diff_month<0, np.logical_and(diff_month==0,diff_day<0))
    if period=='m':
        diff=diff_year*12+diff_month-(diff_day<0)
    if period=='d' or period=='w':
        diff=(dates1-dates2).dt.days
        if period=='w':
            diff=diff//7
    return(diff)

#Para encontrar uam linha de código dentro de um arquivo ipynb ou py
def localizador(string,fmt='.ipynb'):
    for i in os.listdir():
        if i.endswith(fmt):
            print(i)
            file=open(i,'r',encoding='utf-8')
            linhas=file.readlines()
            for line in linhas:
                if string in line:
                    print(line)
            file.close()

#Mesmo sentido do isin do pandas só que para DataFrame ao invés de Series
def DFisin(df1,df2,cols=None):
    if cols==None:
        if df1.shape[1]==df2.shape[1]:
            if all(df1.columns==df2.columns):
                cols=df1.columns
            else:
                raise(ValueError('Colunas das tabelas com nomes diferentes'))
        else:
            raise(ValueError("Número diferente de colunas entre df1 e df2"))
    s1=df1.loc[:,cols[0]].astype(str)
    s2=df2.loc[:,cols[0]].astype(str)
    for i in cols:
        s1=s1.str.cat('/'+df1.loc[:,i].astype(str))
        s2=s2.str.cat('/'+df2.loc[:,i].astype(str))
    return(s1.isin(s2))

#Salva dataframe do código direto no bucket do S3, sem precisar salvar na máquina
def toS3csv(Var=None,Key=None,Bucket=None):
    csv_buffer=StringIO()
    Var.to_csv(csv_buffer)
    resource=b3.resource('s3')
    resource.Object(Bucket,Key).put(Body=csv_buffer.getvalue())


#Encontra a posição da n-ésima ocorrência da substring na string  
def find_nth(frase,sub,n):
    if n<0:
        return(-1)
    s=0
    while n>0:
        s=frase.find(sub,s+1)
        n=n-1
    return(s)

# Recebe dataframe pandas com dados dos arquivos de bucket e retorna uma árvore anytree com todos os arquivos
def df_to_tree(bucket,tabela):
    fs=anytree.AnyNode(id=bucket,folder=1)
    level=tabela.Key.str.count('/')-tabela.Key.str.endswith('/')
    for i in tabela.Key[level==0].index:
        r=tabela.loc[i]
        anytree.AnyNode(parent=fs,id=r.Key,address=r.Key.replace('/',''),Modified=r.LastModified,Size=r.Size)
    for lvl in range(1,level.max()+1):
        rows=tabela.Key[level==lvl].index
        for row in rows:
            r=tabela.loc[row]
            for i in fs.descendants:
                endereco=r.Key[:find_nth(r.Key,'/',lvl)+1]
                if i.id==endereco:
                     anytree.AnyNode(parent=i,address=r.Key[find_nth(r.Key,'/',lvl)+1:],id=r.Key,Modified=r.LastModified,Size=r.Size)
    return(fs)

#Recebe árvore anytree e devolve wigdet explorer 
def tree_to_explorer(node):
    wg=[]
    file=dict(zip(['.csv','.dat','.feather','.tsv','.arr','.json','.xml','.txt','.hdf','.parquet'],it.repeat('file')))
    image=dict(zip(['.png','.jpeg','.jpg','.tiff','.bmp','.ico','.gif','.svg'],it.repeat('image')))
    audio=dict(zip(['.wav','.mp3','.aac','.ogg','.wma'],it.repeat('bullhorn')))
    compress=dict(zip(['.zip','.7z','.rar'],it.repeat('file-zip-o')))
    video=dict(zip(['.webm','.mp4','mkv','.flv','.wmv','.mov','.rmvb','.mpg','.avi','.3gp'],it.repeat('video')))
    icone={**image,**file,**audio,**video,**compress,'.ipynb':'book','.rmd':'book','.xls':'file-excel-o','.xlsx':'file-excel-o','.pdf':'file-pdf-o','.xlsm':'file-excel-o'}
    #Copia lista de filhos
    filhos=node.children
    #Vê quais filhos são pastas(terminam com '/') e rearranja priorizando as pastas
    ord=[np.where([filho.id.endswith('/') for filho in filhos])[0],np.where([not filho.id.endswith('/') for filho in filhos])[0]]
    ord=np.concatenate(ord)
    filhos=[filhos[i] for i in ord]
    #Faz loop passando por cada elemento
    for i in filhos:
        #Se for arquivo, cria um botão
        if len(i.children)==0 and not i.id.endswith('/'):
            b=widgets.Button(description=i.address,layout=widgets.Layout(width='100%',text_align='left'))
            func=lambda x:print(i.id)
            b.on_click(func)
            idx=find_nth(i.address,r'.',i.address.count(r'.'))
            forma=i.address[idx:].lower()
            if forma in icone.keys():
                b.icon=icone[forma]
            wg.append(b)
        #Se for pasta, faz recursão dentro da pasta
        else:
            wg.append(tree_to_explorer(i))
    acc=widgets.Accordion([widgets.VBox(children=wg)])
    acc.set_title(0,node.id)
    acc.set_state({'selected_index': None})
    return(acc)

def AWS_explorer(Bucket,Sessão):
    #Qual serviço eu desejo acessar com minha conexão
    cliente=Sessão.client('s3')
    tabela=pd.DataFrame(cliente.list_objects(Bucket=Bucket)['Contents'])
    arvore=df_to_tree(Bucket,tabela)
    explorador=tree_to_explorer(arvore)
    return(explorador)