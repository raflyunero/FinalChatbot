from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session, render_template, flash, Response
from flask_cors import CORS
from dotenv import load_dotenv
import bcrypt
import os
from datetime import datetime, timedelta
import sqlite3
import json
import random
import pickle
import hashlib
import time
import threading
from pathlib import Path
from fuzzywuzzy import fuzz
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import re
import requests
import glob
import concurrent.futures
from zhipuai import ZhipuAI

app = Flask(__name__, static_folder="static", static_url_path='')
CORS(app)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY", "default_secret")
app.permanent_session_lifetime = timedelta(hours=2)

# Global variables untuk tracking status dan response
processing_status = {}
response_store = {}
pdf_vectorstore = None  # Inisialisasi sebagai None, akan diisi saat startup

# ---------------- Credentials & API ---------------- #
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_HASHED_PASSWORD = os.getenv("ADMIN_HASHED_PASSWORD", "").encode("utf-8")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

if ZHIPU_API_KEY:
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
else:
    client = None
    print("⚠️ ZHIPU_API_KEY tidak ditemukan. Fitur AI tidak akan berfungsi.")

# ---------------- Helpers ---------------- #
def verify_password(input_password, stored_hash):
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_hash)

def get_jawaban(dosen, nip):
    template = random.choice(jawaban_variasi)
    return template.format(dosen=dosen, nip=nip)

jawaban_variasi = [
    "NIP dari {dosen} itu adalah {nip}",
    "NIP {nip} itu punya {dosen}",
    "{dosen} punya NIP: {nip}"
]

# ---------------- Dataset Loaders ---------------- #
# Dosen
dataset_dosen_data = {}
try:
    with open("dataset_dosen.json", "r", encoding="utf-8") as f:
        dataset_dosen_data = json.load(f)
except Exception as e:
    print("⚠️ Gagal load dataset_dosen.json:", e)

# Fakultas
dataset_fakultas_data = {}
try:
    with open("dataset_fakultas.json", "r", encoding="utf-8") as f:
        dataset_fakultas_data = json.load(f)
    print("✅ dataset_fakultas.json berhasil dimuat, total fakultas:",
          len(dataset_fakultas_data.get("data_fakultas", [])))
except Exception as e:
    print("⚠️ Gagal load dataset_fakultas.json:", e)

# Program Studi
dataset_prodi_data = {}
try:
    with open("dataset_prodi.json", "r", encoding="utf-8") as f:
        dataset_prodi_data = json.load(f)
    print("✅ dataset_prodi.json berhasil dimuat, total fakultas:",
          len(dataset_prodi_data.get("dataset_prodi", [])))
except Exception as e:
    print(f"⚠️ Error loading dataset_prodi.json: {e}")
    dataset_prodi_data = {"dataset_prodi": []}

# ---------------- Fungsi Panggilan Program Studi ---------------- #
def get_prodi_by_fakultas(query: str):
    """
    Fungsi untuk mengambil daftar program studi berdasarkan nama fakultas dari dataset_prodi.json.
    """
    try:
        all_prodi = dataset_prodi_data.get("dataset_prodi", [])
        query_lower = query.lower()

        # Jika user tanya "semua prodi"
        if "semua" in query_lower or "daftar prodi" in query_lower or "seluruh" in query_lower:
            response = "Berikut daftar fakultas dan program studi di Universitas Diponegoro:\n\n"
            for item in all_prodi:
                response += f"🎓 {item['fakultas']}:\n"
                for ps in item["program_studi"]:
                    response += f"  - {ps}\n"
                response += "\n"
            return response.strip()

        # Cek program studi tertentu
        for item in all_prodi:
            for ps in item["program_studi"]:
                if ps.lower() in query_lower or query_lower in ps.lower():
                    response = f"Program studi {ps} berada di {item['fakultas']}."
                    other_prodi = [p for p in item["program_studi"] if p.lower() != ps.lower()]
                    if other_prodi:
                        response += f"\n\nProgram studi lain di {item['fakultas']}:\n"
                        for p in other_prodi:
                            response += f"  - {p}\n"
                    return response

        # Cek fakultas tertentu
        for item in all_prodi:
            faculty_name = item["fakultas"].lower()
            if faculty_name in query_lower or query_lower in faculty_name:
                response = f"Program studi yang ada di {item['fakultas']} adalah:\n"
                for ps in item["program_studi"]:
                    response += f"  - {ps}\n"
                return response.strip()

        return "Maaf, saya tidak menemukan program studi atau fakultas tersebut dalam daftar UNDIP. Coba periksa penulisan nama program studi atau fakultasnya."

    except Exception as e:
        return f"Terjadi kesalahan saat mengambil data program studi: {e}"

# ---------------- Database Setup ---------------- #
def get_db():
    return sqlite3.connect('questions.db')

def create_table_if_not_exists():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_count (
            date TEXT PRIMARY KEY,
            count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_question():
    conn = get_db()
    cursor = conn.cursor()
    today = datetime.today().strftime('%Y-%m-%d')
    cursor.execute("SELECT count FROM question_count WHERE date = ?", (today,))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE question_count SET count = count + 1 WHERE date = ?", (today,))
    else:
        cursor.execute("INSERT INTO question_count (date, count) VALUES (?, ?)", (today, 1))
    conn.commit()
    conn.close()

def get_today_question_count():
    conn = get_db()
    cursor = conn.cursor()
    today = datetime.today().strftime('%Y-%m-%d')
    cursor.execute("SELECT count FROM question_count WHERE date = ?", (today,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0

# ---------------- Optimized Zhipu AI Embeddings with Caching ---------------- #
class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-2", cache_dir="embeddings_cache"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True) # Buat folder cache jika tidak ada

    def _get_cache_path(self, text):
        # Buat nama file cache berdasarkan hash teks
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    embeddings.append(pickle.load(f))
            else:
                try:
                    data = {"model": self.model, "input": text}
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    response = requests.post(self.base_url, headers=headers, json=data)
                    if response.status_code != 200:
                        print("⚠️ Zhipu embedding error:", response.text)
                        embeddings.append([0.0] * 1024) # Fallback zero vector
                        continue
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    embeddings.append(embedding)
                    # Simpan ke cache
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embedding, f)
                except Exception as e:
                    print("⚠️ Error embed text:", e)
                    embeddings.append([0.0] * 1024) # Fallback zero vector
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        cache_path = self._get_cache_path(text)
        
        # Cek cache terlebih dahulu
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Jika tidak ada di cache, buat embedding baru
        try:
            data = {"model": self.model, "input": text}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            # Simpan ke cache
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
                
            return embedding
        except Exception as e:
            print("⚠️ Error embedding query:", e)
            return [0.0] * 1024 # Fallback zero vector

# ---------------- PDF Processing ---------------- #
def clean_pdf_text(text):
    text = re.sub(r'[^\w\s\.\,\-\:\(\)\@]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def init_pdf_vectorstore_from_folder(folder_path):
    try:
        if not ZHIPU_API_KEY:
            print("⚠️ ZHIPU_API_KEY tidak ada, tidak bisa memproses PDF.")
            return None
            
        all_documents = []
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            print(f"Tidak ada file PDF di {folder_path}")
            return None
            
        print(f"Memproses {len(pdf_files)} file PDF...")
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            for doc in documents:
                doc.page_content = clean_pdf_text(doc.page_content)
            all_documents.extend(documents)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        texts = text_splitter.split_documents(all_documents)
        
        # Gunakan Zhipu Embeddings dengan caching
        embeddings = ZhipuEmbeddings(api_key=ZHIPU_API_KEY)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        print(f"Error saat memproses PDFs: {e}")
        return None

# Fungsi untuk memuat vector store saat startup
def load_vectorstore_on_startup():
    global pdf_vectorstore
    if ZHIPU_API_KEY:
        print("Memuat vector store saat startup...")
        pdf_vectorstore = init_pdf_vectorstore_from_folder("pdfs")
        if pdf_vectorstore:
            print("✅ Vector store berhasil dimuat")
        else:
            print("⚠️ Gagal memuat vector store")
    else:
        print("⚠️ ZHIPU_API_KEY tidak ada, melewati pemuatan vector store")

# ---------------- Optimized Retrieval Functions ---------------- #
def retrieve_from_pdf(query: str, k=3):
    context = ""

    # Daftar fakultas (tidak perlu vector store)
    if "fakultas" in query.lower():
        fakultas_list = dataset_fakultas_data.get("data_fakultas", [])
        if fakultas_list:
            context += "\nDaftar fakultas di Universitas Diponegoro:\n"
            for i, f_item in enumerate(fakultas_list, 1):
                context += f"{i}. {f_item['nama_fakultas']} - {f_item['link']}\n"
        return context

    # RAG PDF
    if pdf_vectorstore is None:
        return ""
    
    # Gunakan thread pool untuk pencarian paralel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(pdf_vectorstore.similarity_search, query, k=k)
        docs = future.result()
    
    return "\n".join([doc.page_content for doc in docs])

def retrieve_relevant_info(user_message: str):
    context = ""
    msg = user_message.lower()
    relevant_dosen = []
    keywords = msg.split()
    
    # Batasi jumlah kata kunci untuk pencarian
    keywords = keywords[:5]  # Ambil hanya 5 kata pertama
    
    # Pre-filter dosen berdasarkan kata kunci yang paling relevan
    for item in dataset_dosen_data.get("data_dosen", []):
        nama_dosen = item.get("nama_dosen", "").lower()
        nip = item.get("nip", "")
        
        # Cek NIP terlebih dahulu (lebih cepat)
        if nip.lower() in msg:
            relevant_dosen.append({"nama": item["nama_dosen"], "nip": nip, "score": 1000})
            continue
            
        # Hanya lakukan fuzzy matching untuk nama yang mengandung kata kunci
        for keyword in keywords:
            if keyword in nama_dosen:
                score = fuzz.partial_ratio(keyword, nama_dosen)
                if score > 70:
                    relevant_dosen.append({"nama": item["nama_dosen"], "nip": nip, "score": score})
                    break  # Hanya perlu satu kata kunci yang cocok
    
    if relevant_dosen:
        relevant_dosen.sort(key=lambda x: x["score"], reverse=True)
        context += "Informasi dosen yang relevan:\n"
        # Batasi hanya 3 dosen teratas untuk mempercepat
        for dosen in relevant_dosen[:3]:
            context += f"- {dosen['nama']}, NIP: {dosen['nip']}\n"
    
    # Tambahkan konteks dari PDF
    pdf_context = retrieve_from_pdf(user_message)
    if pdf_context.strip():
        context += "\nInformasi dari dokumen:\n" + pdf_context

    return context

# ---------------- Optimized Chatbot Logic ---------------- #
def handle_zhipu_ai_with_rag(user_message: str, session_id: str):
    if not client:
        return "⚠️ Maaf, layanan AI sedang tidak tersedia."
    
    # Cek pertanyaan singkat terlebih dahulu untuk respons template
    if len(user_message.split()) < 5:
        msg_lower = user_message.lower()
        if "siapa" in msg_lower and "rektor" in msg_lower:
            return "Rektor Universitas Diponegoro saat ini adalah Prof. Dr. Yos Johan Utama, S.H., M.Hum."
        if "jam berapa" in msg_lower or "sekarang jam berapa" in msg_lower:
            return f"Sekarang jam {datetime.now().strftime('%H:%M')}."
        # Tambahkan template lainnya sesuai kebutuhan

    # Cek apakah pertanyaan tentang program studi
    if "prodi" in user_message.lower() or "jurusan" in user_message.lower():
        return get_prodi_by_fakultas(user_message)

    relevant_info = retrieve_relevant_info(user_message)
    
    # Batasi panjang konteks untuk mempercepat pemrosesan AI
    if len(relevant_info) > 2000:
        relevant_info = relevant_info[:2000] + "..."
    
    augmented_prompt = f"""
    Informasi relevan dari database UNDIP:
    {relevant_info}
    Pertanyaan pengguna: {user_message}
    """

    try:
        response = client.chat.completions.create(
            model="glm-4.5-Flash",  # Model yang lebih cepat
            messages=[
                {"role": "system", "content": (
                     "Kamu adalah chatbot akademik Universitas Diponegoro (UNDIP). "
                "Jawablah dengan bahasa gaul, santai, sopan, dan jelas ala mahasiswa 🤙, "
                "tetap fokus pada konteks akademik."
                "berikan jawaban yang sama ketika konteks yang ditanyakan sama seperti sebelumnya"
                )},
                {"role": "user", "content": augmented_prompt}
            ],
            max_tokens=1500  # Batasi panjang respons
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Zhipu AI: {e}")
        return "Maaf, terjadi kesalahan saat memproses permintaan Anda."

# ---------------- Routes ---------------- #
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not client:
        return jsonify({"reply": "⚠️ Layanan AI tidak tersedia."})

    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")
    
    processing_status[session_id] = True
    save_question()
    
    def process_question():
        try:
            reply = handle_zhipu_ai_with_rag(user_message, session_id)
            response_store[session_id] = reply
        except Exception as e:
            print(f"Error processing question: {e}")
            response_store[session_id] = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."
        finally:
            processing_status[session_id] = False
    
    thread = threading.Thread(target=process_question)
    thread.start()
    
    return jsonify({"status": "processing"})

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    
    if processing_status.get(session_id, False):
        return jsonify({"status": "processing"})
    
    if session_id in response_store:
        reply = response_store.pop(session_id)
        return jsonify({"status": "completed", "reply": reply})
    
    return jsonify({"status": "not_found"})

# ---------------- New Streaming Route ---------------- #
@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    if not client:
        return jsonify({"error": "Layanan AI tidak tersedia."})

    data = request.get_json()
    user_message = data.get("message", "")
    
    def generate():
        try:
            # Kirim status processing
            yield f"data: {json.dumps({'status': 'processing'})}\n\n"
            
            # Proses pertanyaan
            reply = handle_zhipu_ai_with_rag(user_message, "stream_session")
            save_question()
            
            # Kirim respons dalam chunk untuk efek streaming
            words = reply.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'status': 'streaming', 'content': chunk})}\n\n"
                time.sleep(0.03) # Delay kecil untuk efek streaming
            
            # Kirim status completed
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            
        except Exception as e:
            print(f"Error in stream: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': 'Terjadi kesalahan.'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

# ---------------- Admin Routes ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == ADMIN_USERNAME and verify_password(password, ADMIN_HASHED_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("admin"))
        else:
            flash("Login gagal! Username atau password salah.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/admin")
def admin():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    create_table_if_not_exists()
    today_question_count = get_today_question_count()
    
    try:
        with open("dataset_dosen.json", "r", encoding="utf-8") as f:
            dosen_data = json.load(f)
    except Exception as e:
        print("⚠️ Gagal load dataset_dosen.json:", e)
        dosen_data = {"data_dosen": []}
    
    return render_template(
        "dashboard.html", 
        question_count=today_question_count,
        dataset_dosen_data=dosen_data,
        now=datetime.now(),
        zhipu_connected=client is not None
    )

# ---------------- Main ---------------- #
if __name__ == "__main__":
    # Muat vector store saat startup untuk percepatan
    load_vectorstore_on_startup()
    create_table_if_not_exists()
    app.run(debug=True)