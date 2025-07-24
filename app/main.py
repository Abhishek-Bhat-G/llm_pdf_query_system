import os
import json
import pymupdf as fitz
import faiss
import numpy as np
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_wrapper import call_llm