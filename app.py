import os
import json
import requests
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import time

st.set_page_config(
    page_title="Global AI Assistant",
    page_icon="🌍🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Konfigurasi API Global
# =========================
AVAILABLE_APIS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini"
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": ["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
        "default_model": "mixtral-8x7b-32768"
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "default_model": "claude-3-5-sonnet-20241022"
    },
    "Ollama": {
        "base_url": "http://localhost:11434",
        "env_key": "OLLAMA_API_KEY",
        "models": ["llama3.1", "llama2", "mistral", "neural-chat"],
        "default_model": "llama3.1"
    },
    "Hugging Face": {
        "base_url": "https://api-inference.huggingface.co",
        "env_key": "HF_API_KEY",
        "models": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1"],
        "default_model": "meta-llama/Llama-2-7b-chat-hf"
    }
}

GLOBAL_SYSTEM_PROMPTS = {
    "Global Assistant": """Kamu adalah Global AI Assistant yang sangat kompeten, siap menjawab segala pertanyaan dari berbagai bidang:
- Teknologi & Programming: full-stack development, cloud, DevOps, AI/ML
- Business & Entrepreneurship: strategi, marketing, growth hacking
- Science & Research: riset, analisis data, eksplorasi konsep ilmiah
- Creative Writing: artikel, cerita, copywriting, konten viral
- Personal Development: produktivitas, karir, life hacks
- General Knowledge: history, culture, geography, trivia
- Problem Solving: debug, troubleshooting, strategic thinking

Prinsip:
✓ Ramah, profesional, dan adaptif
✓ Jelaskan dengan detail tapi concise
✓ Gunakan analogi dan contoh real-world
✓ Jika tidak tahu, katakan dengan jujur
✓ Respons dalam bahasa user (Indonesia/English)
✓ Sertakan code, diagram, atau struktur jika relevan
✓ Tawarkan follow-up questions untuk diskusi lebih dalam
""",
    "Expert Coder": """Kamu adalah Expert Software Engineer yang mahir dalam:
- Full-stack development (frontend, backend, database)
- Multiple languages (Python, JavaScript, Go, Rust, Java, C++, etc)
- System design dan architecture
- DevOps & Cloud (AWS, GCP, Azure)
- Testing, debugging, performance optimization
- Best practices dan design patterns

Saat user bertanya tentang kode:
✓ Berikan solusi langsung dengan kode yang siap pakai
✓ Jelaskan logic dan best practices
✓ Sertakan testing strategies
✓ Diskusikan trade-offs dan alternative approaches
✓ Contohkan usage dan edge cases
""",
    "Business Strategist": """Kamu adalah Business & Strategy Consultant yang berpengalaman dalam:
- Business model development
- Market analysis dan competitive strategy
- Go-to-market planning
- Growth hacking dan scaling
- Investment & funding
- Product management
- Team building & leadership

Fokus pada:
✓ Actionable insights
✓ Data-driven recommendations
✓ ROI dan metrics
✓ Risk analysis
✓ Practical implementation roadmap
""",
    "Creative Writer": """Kamu adalah Creative Writing Specialist dengan expertise:
- Content creation untuk berbagai platform
- Storytelling yang engaging
- Copywriting yang persuasif
- SEO optimization
- Brand voice development
- Poetry, fiction, dan creative non-fiction

Tugasmu:
✓ Buat konten yang engaging dan shareable
✓ Pahami target audience
✓ Adaptif dengan tone dan style
✓ Research dan fact-check
✓ Iterasi berdasarkan feedback
""",
    "Research Analyst": """Kamu adalah Research Analyst yang detail dan metodis:
- Literature review dan synthesis
- Data analysis dan interpretation
- Academic writing
- Research methodology
- Trend analysis
- Statistical analysis

Standar kerja:
✓ Cite sources yang kredibel
✓ Critical thinking dan skeptical analysis
✓ Metodologi yang solid
✓ Visualisasi data yang clear
✓ Actionable insights dari research
"""
}

# =========================
# CSS Styling
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .api-status-active {
        color: #00cc44;
        font-weight: bold;
    }
    .api-status-inactive {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Session State Initialization
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_provider" not in st.session_state:
    st.session_state.api_provider = "OpenAI"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

if "api_calls_count" not in st.session_state:
    st.session_state.api_calls_count = 0

# =========================
# Utility Functions
# =========================
def get_env_or_default(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else default

def test_api_connection(provider: str, api_key: str, base_url: str, model: str) -> tuple[bool, str]:
    """Test koneksi ke API"""
    try:
        if provider == "OpenAI" or provider == "Groq":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            endpoint = f"{base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "max_tokens": 10,
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            if response.status_code == 200 or response.status_code == 429:
                return True, "✓ Koneksi berhasil"
            else:
                return False, f"✗ Error: {response.status_code}"
        
        elif provider == "Anthropic":
            headers = {
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            endpoint = "https://api.anthropic.com/v1/messages"
            payload = {
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            if response.status_code == 200 or response.status_code == 429:
                return True, "✓ Koneksi berhasil"
            else:
                return False, f"✗ Error: {response.status_code}"
        
        elif provider == "Ollama":
            endpoint = f"{base_url}/api/tags"
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                return True, "✓ Koneksi berhasil"
            else:
                return False, f"✗ Error: {response.status_code}"
        
        elif provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            endpoint = f"{base_url}/models/{model}"
            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 200 or response.status_code == 401:
                return True, "✓ Koneksi berhasil"
            else:
                return False, f"✗ Error: {response.status_code}"
        
    except requests.exceptions.Timeout:
        return False, "✗ Timeout - Server tidak merespons"
    except Exception as e:
        return False, f"✗ Error: {str(e)[:50]}"

# =========================
# API Call Functions
# =========================
def call_openai_compatible(messages: List[Dict], provider: str, api_key: str, base_url: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI-compatible APIs (OpenAI, Groq)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        error_detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"{provider} API Error: {error_detail}")
    
    return response.json()["choices"][0]["message"]["content"]

def call_anthropic(messages: List[Dict], api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    endpoint = "https://api.anthropic.com/v1/messages"
    
    # Convert messages format
    claude_messages = []
    system_content = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            claude_messages.append({"role": msg["role"], "content": msg["content"]})
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": claude_messages
    }
    
    if system_content:
        payload["system"] = system_content
    
    response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        error_detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"Anthropic API Error: {error_detail}")
    
    return response.json()["content"][0]["text"]

def call_ollama(messages: List[Dict], base_url: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call Ollama local API"""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    response = requests.post(endpoint, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama Error: {response.text[:200]}")
    
    return response.json()["message"]["content"]

def call_hugging_face(messages: List[Dict], api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = f"https://api-inference.huggingface.co/models/{model}"
    
    # Format untuk HF
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"[SYSTEM] {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"[USER] {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"[ASSISTANT] {msg['content']}\n"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    
    response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face Error: {response.text[:200]}")
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")
    return str(result)

def generate_response(user_prompt: str, provider: str, api_key: str, base_url: str, model: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """Generate AI response based on selected provider"""
    
    # Build message list
    chat_messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in st.session_state.messages:
        if msg["role"] in ["user", "assistant"]:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current prompt
    chat_messages.append({"role": "user", "content": user_prompt})
    
    try:
        if provider in ["OpenAI", "Groq"]:
            response = call_openai_compatible(chat_messages, provider, api_key, base_url, model, temperature, max_tokens)
        elif provider == "Anthropic":
            response = call_anthropic(chat_messages, api_key, model, temperature, max_tokens)
        elif provider == "Ollama":
            response = call_ollama(chat_messages, base_url, model, temperature, max_tokens)
        elif provider == "Hugging Face":
            response = call_hugging_face(chat_messages, api_key, model, temperature, max_tokens)
        else:
            response = "Provider tidak valid"
        
        st.session_state.api_calls_count += 1
        return response
    
    except Exception as e:
        raise RuntimeError(f"Gagal generate response: {str(e)}")

# =========================
# Main Layout
# =========================
st.markdown("""
<div class="main-header">
    <h1>🌍 Global AI Assistant</h1>
    <p>Conversational AI yang menjawab segala hal - Multi-Provider, Multi-Personality</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar Configuration
# =========================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# API Provider Selection
st.sidebar.subheader("📡 API Provider")
provider = st.sidebar.selectbox(
    "Pilih Provider AI",
    list(AVAILABLE_APIS.keys()),
    index=list(AVAILABLE_APIS.keys()).index(st.session_state.api_provider)
)
st.session_state.api_provider = provider

provider_info = AVAILABLE_APIS[provider]

# API Key Input
api_key = st.sidebar.text_input(
    f"{provider} API Key",
    type="password",
    value=get_env_or_default(provider_info["env_key"], ""),
    help=f"Masukkan API key untuk {provider}. Bisa juga set env {provider_info['env_key']}"
)

# Base URL (untuk Ollama & custom endpoints)
base_url = st.sidebar.text_input(
    "Base URL",
    value=get_env_or_default("CUSTOM_BASE_URL", provider_info["base_url"]),
    help="Untuk Ollama: http://localhost:11434. Untuk custom proxy/self-hosted"
)

# Model Selection
model = st.sidebar.selectbox(
    "Model",
    provider_info["models"],
    index=0 if provider_info["default_model"] not in provider_info["models"] else provider_info["models"].index(provider_info["default_model"])
)

# Model Parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
with col2:
    max_tokens = st.slider("Max Tokens", 100, 4000, 1500, 100)

st.sidebar.markdown("---")

# AI Personality Selection
st.sidebar.subheader("🎭 AI Personality")
personality = st.sidebar.selectbox(
    "Pilih Personality",
    list(GLOBAL_SYSTEM_PROMPTS.keys()),
    index=0
)

system_prompt = st.sidebar.text_area(
    "System Prompt (Custom)",
    value=GLOBAL_SYSTEM_PROMPTS[personality],
    height=200,
    help="Edit system prompt atau gunakan preset di atas"
)

st.sidebar.markdown("---")

# API Connection Test
st.sidebar.subheader("🔌 API Status")
if st.sidebar.button("Test Koneksi API"):
    with st.spinner("Testing..."):
        is_connected, status_msg = test_api_connection(provider, api_key, base_url, model)
        if is_connected:
            st.sidebar.success(status_msg)
        else:
            st.sidebar.error(status_msg)

st.sidebar.markdown("---")

# Session Info
st.sidebar.subheader("📊 Session Info")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Messages", len(st.session_state.messages))
with col2:
    st.metric("API Calls", st.session_state.api_calls_count)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.api_calls_count = 0
    st.rerun()

# Export Chat Button
if st.sidebar.button("💾 Export Chat"):
    if st.session_state.messages:
        chat_export = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.sidebar.download_button(
            label="Download JSON",
            data=chat_export,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.sidebar.warning("Tidak ada chat untuk di-export")

st.sidebar.markdown("---")
st.sidebar.info("""
### 🌟 Features
- Multi-provider AI (OpenAI, Groq, Anthropic, Ollama, HF)
- 5+ AI Personalities
- Real-time API testing
- Chat history export
- Advanced customization

### 🚀 Supported Providers
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-Turbo
- **Groq**: Mixtral, Llama 3.1 (fast inference)
- **Anthropic**: Claude 3.5 Sonnet, Opus, Haiku
- **Ollama**: Local LLMs (free, offline)
- **Hugging Face**: 1000+ open models
""")

# =========================
# Main Chat Area
# =========================
col_main = st.container()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Tanya apa saja ke Global AI Assistant..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        progress_placeholder = st.empty()
        
        try:
            # Validasi API key
            if not api_key and provider != "Ollama":
                raise RuntimeError(f"⚠️ {provider} API Key belum diisi. Masukkan di sidebar atau set env {provider_info['env_key']}")
            
            if provider == "Ollama" and not api_key:
                placeholder.info("ℹ️ Menghubung ke Ollama lokal di " + base_url)
            
            # Show loading state
            with progress_placeholder.container():
                st.info(f"⏳ Connecting to {provider}...")
            
            # Generate response
            start_time = time.time()
            response = generate_response(
                prompt,
                provider,
                api_key,
                base_url,
                model,
                system_prompt,
                temperature,
                max_tokens
            )
            elapsed_time = time.time() - start_time
            
            # Clear progress placeholder
            progress_placeholder.empty()
            
            # Display response
            placeholder.markdown(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show metadata
            st.caption(f"✓ Response dari {provider} ({model}) dalam {elapsed_time:.2f}s")
        
        except Exception as e:
            progress_placeholder.empty()
            error_msg = f"❌ Error: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =========================
# Quick Actions
# =========================
st.markdown("---")
st.subheader("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💡 Generate Ideas"):
        st.session_state.quick_action = "Generate Ideas"
        st.session_state.quick_prompt = "Berikan 10 ide kreatif dan inovatif untuk:"

with col2:
    if st.button("🔍 Analyze"):
        st.session_state.quick_action = "Analyze"
        st.session_state.quick_prompt = "Analisis secara mendalam tentang:"

with col3:
    if st.button("💻 Code Example"):
        st.session_state.quick_action = "Code"
        st.session_state.quick_prompt = "Buatkan contoh kode untuk:"

with col4:
    if st.button("📝 Write Content"):
        st.session_state.quick_action = "Write"
        st.session_state.quick_prompt = "Tulis artikel/konten tentang:"

# =========================
# Information Panel
# =========================
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📖 How to Use", "🔧 API Info", "💎 Pro Tips"])

with tab1:
    st.markdown("""
    ## Cara Menggunakan Global AI Assistant
    
    1. **Pilih Provider**: Tentukan API provider mana yang ingin digunakan
    2. **Masukkan API Key**: Sediakan credentials (bisa via env variables)
    3. **Pilih Personality**: Sesuaikan personality AI dengan kebutuhan Anda
    4. **Customize**: Ubah temperature, max tokens, atau system prompt
    5. **Chat**: Mulai berbincang dengan AI
    
    ### Tips:
    - Gunakan Groq untuk response cepat
    - Gunakan Claude untuk reasoning kompleks
    - Gunakan Ollama untuk privacy (offline)
    - Export chat untuk dokumentasi atau analisis
    """)

with tab2:
    st.markdown("""
    ## API Provider Information
    
    ### OpenAI
    - **URL**: https://api.openai.com/v1
    - **Get Key**: https://platform.openai.com/api-keys
    - **Best For**: Produksi, reliability tinggi
    
    ### Groq
    - **URL**: https://api.groq.com/openai/v1
    - **Get Key**: https://console.groq.com
    - **Best For**: Ultra-fast inference, streaming
    
    ### Anthropic (Claude)
    - **URL**: https://api.anthropic.com
    - **Get Key**: https://console.anthropic.com
    - **Best For**: Complex reasoning, long context
    
    ### Ollama (Local)
    - **URL**: http://localhost:11434
    - **Setup**: Download dari https://ollama.ai
    - **Best For**: Privacy, offline, free
    
    ### Hugging Face
    - **URL**: https://api-inference.huggingface.co
    - **Get Key**: https://huggingface.co/settings/tokens
    - **Best For**: Open-source models
    """)

with tab3:
    st.markdown("""
    ## 💎 Pro Tips for Best Results
    
    ### Prompt Engineering
    - Berikan context yang jelas
    - Gunakan format structured (JSON, markdown)
    - Sebutkan output format yang diinginkan
    - Gunakan few-shot examples
    
    ### Model Selection
    - GPT-4o untuk tasks kompleks, reasoning
    - GPT-4o-mini untuk speed & cost
    - Claude untuk nuanced writing
    - Mixtral (Groq) untuk creative tasks
    - Ollama untuk privacy-sensitive work
    
    ### Temperature Settings
    - **0.0-0.3**: Precise, consistent (coding, analysis)
    - **0.5-0.7**: Balanced (general chat)
    - **0.8-1.0**: Creative (writing, ideas)
    - **1.0+**: Very creative (brainstorming)
    
    ### Cost Optimization
    - Gunakan Groq (free tier available)
    - Gunakan Ollama untuk volume tinggi
    - Combine APIs: Ollama + OpenAI backup
    - Monitor API calls di sidebar
    
    ### Privacy & Security
    - Use Ollama untuk sensitive data
    - Jangan share API keys di public repos
    - Gunakan env variables
    - Regular audit logs
    """)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    <p>🌍 Global AI Assistant v1.0 | Multi-Provider • Multi-Personality • Production-Ready</p>
    <p>Powered by OpenAI, Groq, Anthropic, Ollama & Hugging Face</p>
    <p>Made with ❤️ for developers, creators, and thinkers</p>
</div>
""", unsafe_allow_html=True)
