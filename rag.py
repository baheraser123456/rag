from langchain_community.llms import ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
print('start genrate')
dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(dir,'data','shawshank.txt')
db_dir=os.path.join(dir,'chromadb')
embedding=OllamaEmbeddings(model='mxbai-embed-large')
# loader=TextLoader(file_path,encoding='utf-8')
# doc=loader.load()
# text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# docs=text_spliter.split_documents(doc)
# Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=db_dir)
db=Chroma(embedding_function=embedding, persist_directory=db_dir)
ret=db.as_retriever(search_type='mmr', search_kwargs={'k':4,'fecth_k':20,'lambda_mult':.5})
data=ret.invoke('tell me about shawshank')
for i,d in enumerate(data):
    print(f'doc{i}:\n{d.page_content}\n')










