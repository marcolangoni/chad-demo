from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import subprocess
import shutil
import psycopg2

import numpy as np

import streamlit as st

from llama_index.vector_stores import TimescaleVectorStore
from llama_index import StorageContext
from llama_index.indices.vector_store import VectorStoreIndex

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from timescale_vector import client

from typing import List, Tuple

from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding

from llama_index.text_splitter import SentenceSplitter

def create_uuid(date_string: str):
    datetime_obj = datetime.fromisoformat(date_string)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)

# Create a Node object from a single cdr
# date,projectId,direction,duration,estimatedCost,fromE164,result,toE164,contentType
def create_nodes(row):
    text_splitter = SentenceSplitter(chunk_size=128)

    cdr = row.to_dict()
    record_content = (
        "Date: "+ str(cdr["date"])
        + " "
        + "Project: "+ cdr['projectId']
        + " "
        + "Direction: "+ cdr['direction']
        + " "
        + "Cost: "+ str(cdr['estimatedCost'])
        + " "
        + "From: "+ cdr['fromE164']
        + " "
        + "Result: "+ cdr['result']
        + " "
        + "To: "+ cdr['toE164']
        + " "
        + "Type: "+ cdr['contentType']
    )

    text_chunks = text_splitter.split_text(record_content)
    nodes = [TextNode(
        id_=create_uuid(cdr["date"]),
        text=chunk,
        metadata={
            "project": cdr["projectId"],
            "direction": cdr['direction'],
            "date": cdr["date"],
            "cost": cdr["estimatedCost"],
            "from": cdr["fromE164"],
            "result": cdr["result"],
            "to": cdr["toE164"],
            "type": cdr["contentType"]
        },
    ) for chunk in text_chunks]

    return nodes

def load_into_db(df_combined):
    embedding_model = OpenAIEmbedding()
    embedding_model.api_key = st.secrets["OPENAI_API_KEY"]

    ts_vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name="cdr",
        time_partition_interval=timedelta(days=1),
    )

    ts_vector_store._sync_client.drop_table()
    ts_vector_store._sync_client.create_tables()

    cpus = cpu_count()
    min_splits = len(df_combined.index) / 1000 #no more than 1000 rows/split
    num_splits = int(max(cpus, min_splits))


    st.spinner("Processing...")
    progress = st.progress(0, f"Processing, with {num_splits} splits")
    start = time.time()

    nodes_combined = [item for sublist in [create_nodes(row) for _, row in df_combined.iterrows()] for item in sublist]
    node_tasks = np.array_split(nodes_combined, num_splits)
    
    def worker(nodes): 
        start = time.time()
        texts = [n.get_content(metadata_mode="all") for n in nodes] 
        embeddings = embedding_model.get_text_embedding_batch(texts)
        for i, node in enumerate(nodes):
            node.embedding = embeddings[i]
        duration_embedding = time.time()-start
        start = time.time()
        ts_vector_store.add(nodes)
        duration_db = time.time()-start
        return (duration_embedding, duration_db)

    embedding_durations = []
    db_durations = []
    with ThreadPoolExecutor() as executor:
        times = executor.map(worker, node_tasks)

        for index, worker_times in enumerate(times):
            duration_embedding, duration_db = worker_times
            embedding_durations.append(duration_embedding)
            db_durations.append(duration_db)
            progress.progress((index+1)/num_splits, f"Processing, with {num_splits} splits")


    progress.progress(100, f"Processing embeddings took {sum(embedding_durations)}s. Db took {sum(db_durations)}s. Using {num_splits} splits")
    
    st.spinner("Creating the index...")
    progress = st.progress(0, "Creating the index")
    start = time.time()
    ts_vector_store.create_index()
    duration = time.time()-start
    progress.progress(100, f"Creating the index took {duration} seconds")
    st.success("Done")

def get_cdrs(): 
    st.spinner("Reading CDRs...")
    start = time.time()
    progress = st.progress(0, "Reading CDRs.")
    file_path = "data/chad-cdrs.csv"

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    duration = time.time()-start
    progress.progress(100, f"Reading CDRs took {duration} seconds")
    return df


def load_cdrs():
    if st.button("Load CDRs into the database"):
        df = get_cdrs()
        load_into_db(df)

st.set_page_config(page_title="Load CDRs", page_icon="💿")
st.markdown("# Load CDRs for analysis")
st.sidebar.header("Load CDRs")
st.write(
    """Load CDRs!"""
)

load_cdrs()
