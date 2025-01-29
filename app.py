import os
import boto3
import json
from io import BytesIO
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# AWS S3 Config
AWS_S3_BUCKET = "testing-bart-1"
s3_client = boto3.client("s3")


def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def save_to_s3_json(data, key):
    """Save data to S3 as a JSON file."""
    json_buffer = BytesIO()
    json_buffer.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
    json_buffer.seek(0)
    s3_client.upload_fileobj(json_buffer, AWS_S3_BUCKET, key)


def load_from_s3_json(key):
    """Load JSON data from S3."""
    try:
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception:
        return None


def save_faiss_to_s3(vectorstore, user_id, pdf_id):
    """Save FAISS index to S3 under user_id/pdf_id."""
    vectorstore.save_local("/tmp/faiss_index")
    s3_client.upload_file("/tmp/faiss_index/index.faiss", AWS_S3_BUCKET, f"{user_id}/{pdf_id}/index.faiss")
    s3_client.upload_file("/tmp/faiss_index/index.pkl", AWS_S3_BUCKET, f"{user_id}/{pdf_id}/index.pkl")


def load_faiss_from_s3(user_id, pdf_id):
    """Load FAISS index from S3 for a specific user and PDF."""
    try:
        os.makedirs("/tmp/faiss_index", exist_ok=True)
        s3_client.download_file(AWS_S3_BUCKET, f"{user_id}/{pdf_id}/index.faiss", "/tmp/faiss_index/index.faiss")
        s3_client.download_file(AWS_S3_BUCKET, f"{user_id}/{pdf_id}/index.pkl", "/tmp/faiss_index/index.pkl")
        return FAISS.load_local("/tmp/faiss_index", OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    except Exception:
        return None


def extract_pdf_text(pdf_path):
    """Extracts text from a single PDF file."""
    pdf_reader = PdfReader(pdf_path)
    return "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])


def split_text_to_chunks(raw_text):
    """Splits text into smaller chunks."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)


def create_vectorstore(text_chunks, user_id, pdf_id):
    """Creates and saves a FAISS vector store."""
    vectorstore = FAISS.from_texts(text_chunks, OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    save_faiss_to_s3(vectorstore, user_id, pdf_id)
    return vectorstore


def generate_conversation_chain(vectorstore, chat_history):
    """Creates a conversational retrieval chain."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    # Convert chat history from JSON (plain text) to LangChain message objects
    for entry in chat_history:
        memory.chat_memory.add_user_message(entry["question"])
        memory.chat_memory.add_ai_message(entry["answer"])

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


@app.route('/chat', methods=['POST'])
def chat():
    """Handles both PDF uploads and questions in a single request."""
    user_id = request.form.get('user_id') or request.json.get('user_id')
    pdf_id = request.form.get('pdf_id') or request.json.get('pdf_id')
    file = request.files.get('file')
    question = request.form.get('question') or request.json.get('question')

    if not user_id or not pdf_id or not question:
        return jsonify({'error': 'User ID, PDF ID, and question are required'}), 400

    # Step 1: Check if FAISS index already exists
    vectorstore = load_faiss_from_s3(user_id, pdf_id)

    if file:
        s3_pdf_key = f"{user_id}/{pdf_id}.pdf"

        # Step 2: If FAISS exists and PDF already exists, skip processing
        if vectorstore and file_exists_in_s3(AWS_S3_BUCKET, s3_pdf_key):
            print("PDF and FAISS index found, skipping reprocessing.")
        else:
            # Process and store the new PDF
            filename = secure_filename(file.filename)
            filepath = f"/tmp/{filename}"

            file_content = file.read()
            with open(filepath, "wb") as f:
                f.write(file_content)

            s3_client.upload_fileobj(BytesIO(file_content), AWS_S3_BUCKET, s3_pdf_key)

            raw_text = extract_pdf_text(filepath)
            text_chunks = split_text_to_chunks(raw_text)
            vectorstore = create_vectorstore(text_chunks, user_id, pdf_id)

            os.remove(filepath)

    if not vectorstore:
        return jsonify({'error': 'No stored data found for this user and PDF'}), 400

    # Step 3: Load chat history
    chat_history_key = f"{user_id}/{pdf_id}/chat_history.json"
    chat_history = load_from_s3_json(chat_history_key) or []

    conversation_chain = generate_conversation_chain(vectorstore, chat_history)

    # Step 4: Get response
    response = conversation_chain.invoke({"question": question})

    # Step 5: Update and save chat history (Only storing plain text, not objects)
    chat_history.append({
        "question": question,
        "answer": response.get("answer", "No answer available")
    })
    save_to_s3_json(chat_history, chat_history_key)

    return jsonify({'response': response.get("answer", "No answer available")}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))

