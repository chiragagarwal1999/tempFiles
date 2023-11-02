#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
import streamlit as st
import ingest
import subprocess
import time
import shutil

# if not load_dotenv():
#     print("Could not load .env file or it is empty. Please check if it exists and is readable.")
#     exit(1)

#embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
#persist_directory = os.environ.get('PERSIST_DIRECTORY')

# model_type = os.environ.get('MODEL_TYPE')
# model_path = os.environ.get('MODEL_PATH')
# model_n_ctx = os.environ.get('MODEL_N_CTX')
# model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
# target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# # Added a paramater for GPU layer numbers
# n_gpu_layers = os.environ.get('N_GPU_LAYERS')
#
# # Added custom directory path for CUDA dynamic library
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/extras/CUPTI/lib64")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")
# os.add_dll_directory("C:/tools/cuda/bin")

from constants import CHROMA_SETTINGS


if "embeddings" not in st.session_state:
    embeddings = HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDINGS_MODEL_NAME"))
    st.session_state["embeddings"] = embeddings

if "chroma_client" not in st.session_state or st.session_state["chroma_client"] is None:
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=os.environ.get('PERSIST_DIRECTORY'))
    st.session_state["chroma_client"] = chroma_client

if "db" not in st.session_state or st.session_state["db"] is None:
    db = Chroma(persist_directory=os.environ.get('PERSIST_DIRECTORY'), embedding_function=st.session_state["embeddings"], client_settings=CHROMA_SETTINGS,
                client=st.session_state["chroma_client"])
    st.session_state["db"] = db

if "retriever" not in st.session_state or st.session_state["retriever"] is None:
    retriever = st.session_state["db"].as_retriever(search_kwargs={"k": int(os.environ.get('TARGET_SOURCE_CHUNKS',4))})
    st.session_state["retriever"] = retriever

if "llm" not in st.session_state:
    llm = GPT4All(model=os.environ.get('MODEL_PATH'), max_tokens=os.environ.get('MODEL_N_CTX'), backend='gptj', n_batch=int(os.environ.get('MODEL_N_BATCH',8)), verbose=False)
    st.session_state["llm"] = llm

if "qa" not in st.session_state or st.session_state["qa"] is None:
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    args = parser.parse_args()
    qa = RetrievalQA.from_chain_type(llm=st.session_state["llm"], chain_type="stuff", retriever=st.session_state["retriever"], return_source_documents=not args.hide_source)
    st.session_state["qa"] = qa

st.title("Ask your documents!")

uploaded_file = st.file_uploader("Upload your file here...")
prompt = st.text_input("Input your query here")

if uploaded_file and not prompt:
    db = st.session_state.get("db")
    dir1 = './source_documents'
    for f in os.listdir(dir1):
        os.remove(os.path.join(dir1, f))
    print("Cleaned source documents")
    # dir2 = './db'
    # for root, dirs, files in os.walk(dir2):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         os.remove(file_path)
    #     for dir in dirs:
    #         dir_path = os.path.join(root, dir)
    #         shutil.rmtree(dir_path)
    print("Removed existing vectorDB")
    upload_file_dir = './source_documents'
    file_path = os.path.join(upload_file_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    print("Uploaded new document!")
    st.success(f"Saved: {uploaded_file.name}")

    st.text("Ingesting document...")
    subprocess_cmd = "python ingest.py"  # Replace 'your_script.py' with the actual script name
    process = subprocess.Popen(subprocess_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
        time.sleep(1)
    st.text("Ingestion complete! Start asking questions.")


if prompt:
    query = prompt
    st.write("Generating results...")
    start = time.time()
    res = st.session_state["qa"](query)
    answer= res['result']
    # , [] if args.hide_source else res['source_documents']
    end = time.time()
    st.write("Here is the result")
    st.write(answer)
    time = round(end - start, 2)
    st.write("The answer took around " + str(time) + " seconds")
    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)
    prompt = None

    # Print the relevant sources used for the answer
    # for document in docs:
    #     print("\n> " + document.metadata["source"] + ":")
    #     print(document.page_content)

