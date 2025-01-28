from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import DeepSeekChatEndpoint
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()

# Model API keys
MODEL_KEYS = {
    'gpt4': os.getenv('OPENAI_API_KEY'),
    'gpt4-turbo': os.getenv('OPENAI_API_KEY'),
    'gpt4-32k': os.getenv('OPENAI_API_KEY'),
    'gpt3.5-turbo': os.getenv('OPENAI_API_KEY'),
    'gpt4-vision': os.getenv('OPENAI_API_KEY'),
    'claude3-opus': os.getenv('ANTHROPIC_API_KEY'),
    'claude3-sonnet': os.getenv('ANTHROPIC_API_KEY'),
    'claude3-haiku': os.getenv('ANTHROPIC_API_KEY'),
    'claude2.1': os.getenv('ANTHROPIC_API_KEY'),
    'deepseek-chat': os.getenv('DEEPSEEK_API_KEY'),
    'deepseek-coder': os.getenv('DEEPSEEK_API_KEY'),
}

def get_llm(model_key):
    """Returns the appropriate language model based on the selected model."""
    
    # OpenAI Models
    openai_models = {
        'gpt4': "gpt-4",
        'gpt4-turbo': "gpt-4-turbo-preview",
        'gpt4-32k': "gpt-4-32k",
        'gpt3.5-turbo': "gpt-3.5-turbo",
        'gpt4-vision': "gpt-4-vision-preview"
    }
    
    # Anthropic Models
    claude_models = {
        'claude3-opus': "claude-3-opus-20240229",
        'claude3-sonnet': "claude-3-sonnet-20240229",
        'claude3-haiku': "claude-3-haiku-20240229",
        'claude2.1': "claude-2.1"
    }
    
    # DeepSeek Models
    # deepseek_models = {
    #     'deepseek-chat': "deepseek-chat",
    #     'deepseek-coder': "deepseek-coder"
    # }
    
    try:
        # Handle OpenAI models
        if model_key in openai_models:
            return ChatOpenAI(
                api_key=MODEL_KEYS[model_key],
                model_name=openai_models[model_key],
                temperature=0.7
            )
        
        # Handle Claude models
        elif model_key in claude_models:
            return ChatAnthropic(
                api_key=MODEL_KEYS[model_key],
                model_name=claude_models[model_key],
                temperature=0.7
            )
            
        # Handle DeepSeek models
        # elif model_key in deepseek_models:
        #     return DeepSeekChatEndpoint(
        #         deepseek_api_key=MODEL_KEYS[model_key],
        #         model_name=deepseek_models[model_key],
        #         temperature=0.7,
        #         api_base="https://api.deepseek.com/v1",  # Update with correct API base URL
        #     )
        
        else:
            raise ValueError(f"Invalid model selected: {model_key}")
            
    except Exception as e:
        logger.error(f"Error initializing model {model_key}: {str(e)}")
        raise Exception(f"Failed to initialize model {model_key}: {str(e)}")

def extract_pdf_text(pdf_path):
    """Extracts text from a PDF file with better error handling."""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip(), None
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return None, f"Failed to extract text: {str(e)}"

def split_text_to_chunks(raw_text):
    """Splits text into smaller chunks with improved chunking strategy."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks, None
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return None, f"Failed to split text: {str(e)}"

def create_vectorstore(text_chunks, model_key):
    """Creates a FAISS vector store with better error handling."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=MODEL_KEYS['gpt4'])  # Using OpenAI embeddings for all models
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
        return vectorstore, None
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None, f"Failed to create vector store: {str(e)}"

def generate_conversation_chain(vectorstore, model_key):
    """Creates a conversational retrieval chain with the selected model."""
    try:
        llm = get_llm(model_key)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return conversation_chain, None
    except Exception as e:
        logger.error(f"Error generating conversation chain: {str(e)}")
        return None, f"Failed to generate conversation chain: {str(e)}"

@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to the PDF Chat API", 
        "endpoints": {
            "/upload": "POST - Upload a PDF file",
            "/ask": "POST - Ask a question about the uploaded PDF"
        },
        "available_models": {
            "OpenAI": ["gpt4", "gpt4-turbo", "gpt4-32k", "gpt3.5-turbo", "gpt4-vision"],
            "Anthropic": ["claude3-opus", "claude3-sonnet", "claude3-haiku", "claude2.1"],
            "DeepSeek": ["deepseek-chat", "deepseek-coder"]
        }
    })

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handles PDF upload and text extraction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            text, error = extract_pdf_text(filepath)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if error:
                return jsonify({'error': error}), 400

            return jsonify({'text': text}), 200

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles question answering using the selected model."""
    try:
        data = request.json
        if not all(k in data for k in ['text', 'question', 'model_key']):
            return jsonify({'error': 'Missing required fields'}), 400

        text = data['text']
        question = data['question']
        model_key = data['model_key']

        if model_key not in MODEL_KEYS:
            return jsonify({'error': f'Invalid model selected: {model_key}'}), 400

        if not MODEL_KEYS[model_key]:
            return jsonify({'error': f'API key not configured for {model_key}'}), 400

        chunks, error = split_text_to_chunks(text)
        if error:
            return jsonify({'error': error}), 400

        vectorstore, error = create_vectorstore(chunks, model_key)
        if error:
            return jsonify({'error': error}), 400

        conversation_chain, error = generate_conversation_chain(vectorstore, model_key)
        if error:
            return jsonify({'error': error}), 400

        response = conversation_chain.invoke({
            "question": question,
            "chat_history": []
        })

        return jsonify({
            'response': response['answer'],
            'sources': [doc.page_content[:200] + "..." for doc in response.get('source_documents', [])]
        }), 200

    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles file size limit exceeded error."""
    return jsonify({'error': 'File size exceeds limit (32MB)'}), 413

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
