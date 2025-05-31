import streamlit as st
import chromadb

CHROMA_PATH = "./chromadb_database_v2"  # เปลี่ยนเป็น path ที่เก็บฐานข้อมูลของคุณใน repo

def test_persistent_client():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        st.success("เชื่อมต่อ PersistentClient สำเร็จ")

        collections = client.list_collections()
        st.write(f"Collections ที่พบ: {[c.name for c in collections]}")

        collection_name = "locklearn_recommendations"
        if collection_name in [c.name for c in collections]:
            collection = client.get_collection(collection_name)
            query_text = ["test query"]
            results = collection.query(query_texts=query_text, n_results=3)
            st.write("ผลลัพธ์ query:")
            st.write(results)
        else:
            st.warning(f"Collection '{collection_name}' ไม่พบในฐานข้อมูล")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

st.title("Test ChromaDB PersistentClient in Streamlit Cloud")

if st.button("ทดสอบเชื่อมต่อและ query ChromaDB"):
    test_persistent_client()
