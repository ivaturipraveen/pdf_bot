import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()

def extract_pdf_text(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return None, str(e)

def split_text_to_chunks(raw_text):
    """Splits text into smaller chunks."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # Smaller chunk size for better context
            chunk_overlap=200,  # Overlap for context continuity
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks
    except Exception as e:
        return None, str(e)

def create_vectorstore(text_chunks):
    """Creates a FAISS vector store using text chunks."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
        return vectorstore
    except Exception as e:
        return None, str(e)

def generate_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    try:
        llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/upload', methods=['POST'])
def upload_pdf_and_ask():
    """
    Upload a PDF file and ask a question about it.
    Expected input:
    - File: A PDF file
    - Question: The question as a form field or JSON payload
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file temporarily
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from PDF
        raw_text = extract_pdf_text(filepath)
        if not raw_text:
            os.remove(filepath)
            return jsonify({'error': 'Failed to extract text from PDF'}), 400

        # Split text into chunks
        text_chunks = split_text_to_chunks(raw_text)
        if not text_chunks:
            os.remove(filepath)
            return jsonify({'error': 'Failed to split text into chunks'}), 400

        # Create FAISS vectorstore
        vectorstore = create_vectorstore(text_chunks)
        if not vectorstore:
            os.remove(filepath)
            return jsonify({'error': 'Failed to create vector store'}), 400

        # Clean up uploaded file
        os.remove(filepath)

        # Get the user's question
        question = request.form.get('question') or request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Generate conversation chain
        conversation_chain = generate_conversation_chain(vectorstore)
        if not conversation_chain:
            return jsonify({'error': 'Failed to create conversation chain'}), 500

        # Generate response
        prompt = (
            f"You are a knowledgeable assistant. Based on the uploaded document, "
            f"answer the following question concisely yet thoroughly: '{question}'"
        )
        try:
            response = conversation_chain.invoke({"question": prompt})
            if "answer" in response:
                return jsonify({'response': response["answer"]}), 200
            else:
                return jsonify({'error': 'No response from the model'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
