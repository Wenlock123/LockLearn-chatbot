import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# กำหนด path ที่เก็บฐานข้อมูล vector ของ chromadb
CHROMA_PATH = "./chromadb_database_v2"

# สร้าง client chromadb โดยระบุโฟลเดอร์ persist_directory
client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH
    )
)

st.title("LockLearn Chatbot")

# โหลดโมเดล embedding (ตัวอย่างใช้ sentence-transformers)
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embedder = load_embedding_model()

# สมมติชื่อ collection ที่ใช้เก็บข้อมูล vector
COLLECTION_NAME = "recommendations"

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    st.error(f"ไม่พบ collection ชื่อ '{COLLECTION_NAME}' ในฐานข้อมูล: {e}")
    st.stop()

# รับ input จากผู้ใช้
query = st.text_input("ถามคำถามหรือขอคำแนะนำ:")

if query:
    # สร้าง embedding จากข้อความผู้ใช้
    query_embedding = embedder.encode(query).tolist()

    # ค้นหาในฐานข้อมูลโดยใช้ embedding ที่สร้าง
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # จำนวนผลลัพธ์ที่ต้องการ
    )

    # แสดงผลลัพธ์
    st.write("คำแนะนำที่ใกล้เคียง:")
    for i, doc in enumerate(results['documents'][0]):
        st.markdown(f"**{i+1}.** {doc}")
