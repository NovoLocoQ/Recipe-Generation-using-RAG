from typing import List
from langchain_core.load import loads,dumps
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

llm=ChatOllama(model="gemma3:4b")
Ranker=CrossEncoder(model_name_or_path='cross-encoder/ms-marco-MiniLM-L6-v2')
docs=PyPDFLoader(file_path="recipe.pdf").load_and_split()
filtered_docs = [doc for doc in docs if doc.metadata.get("page") != 32]
print(len(filtered_docs))
spliter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800,chunk_overlap=100)
chunks=spliter.split_documents(documents=filtered_docs)
vectorstore=Chroma.from_documents(embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),documents=chunks)
DenseRetriever=vectorstore.as_retriever(search_kwargs={"k":3})
SparseRetriver=BM25Retriever.from_documents(documents=chunks)
retriever=EnsembleRetriever(
    retrievers=[DenseRetriever,SparseRetriver]
)
def get_union(documents:list[list]):
    docs=[dumps(s) for sub in documents for s in sub]
    docs=list(set(docs))
    return [loads(doc) for doc in docs]
#response=retriever.invoke("give recipe and ingredients for Mushroom Bruschetta With Balsamic & Thyme and also for Apple French Toast ")
Mtemplate="""You are multi query generator based on a given question.Break down the given question and generate 4 sentences so as to improve retriever results.Each generated sentence should be concise and unique.
            Question:{question}.Give just the question one by one seperated by a new line and Nothing more."""
Mprompt=PromptTemplate.from_template(template=Mtemplate)
Mchain=(
    {"question":RunnablePassthrough()}
    | Mprompt
    | llm
    | StrOutputParser()
    | (lambda x:x.split("\n"))
)

Rchain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(lambda x: {
        "query": x["query"],
        "subqueries": Mchain.invoke(x["query"])
    })
    | RunnableLambda(lambda x: {
        "query": x["query"],
        "docs": get_union(retriever.map().invoke(x["subqueries"]))
    })
)
def Reranker(docs: List[Document], query: str, top_k=6):
    pairs = [(query, d.page_content) for d in docs]
    scores = Ranker.predict(pairs)
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    top_docs = ranked[:top_k]
    return {
        "query": query,
        "context_docs": [d for d, _ in top_docs],
        "scores": [float(s) for _, s in top_docs]
    }
Rechain=(
    Rchain
    | RunnableLambda(lambda x:Reranker(x["docs"],x["query"]))
)
Mprompt = PromptTemplate.from_template("""
You are a recipe generation assistant.
Use ONLY the context provided below to answer.

Question: {question}

Context:
{context}
""")
def join_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
gen = (
    Rechain
    | RunnableLambda(lambda x: {"question": x["query"],"context": join_docs(x["context_docs"])})
    | Mprompt
    | llm
    | StrOutputParser()
)
response=gen.invoke("give recipe and ingredients for Mushroom Bruschetta With Balsamic & Thyme and also for Apple French Toast")
print(response)
