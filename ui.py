import os
import time
import json
import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from fpdf import FPDF
from chat_engine import ChatEngine
from auth import login_user, register_user
from user_data import (
    handle_pdf_upload,
    save_user_data_from_session, # Now importing this to save user config
    delete_pdf_for_user,
    create_new_chat_session,
    delete_chat_session,
    archive_chat_session,
    restore_chat_session
)
# REMOVED: from config import save_config - no longer needed here
from voice import speak_text, transcribe_audio
from quiz_generator import QuizGenerator

# --- UI Enhancement Functions ---

def add_bg_from_local(image_file):
    """Adds a background image from a local file to the Streamlit app."""
    if not os.path.exists(image_file):
        st.warning(f"Background image not found: {image_file}. Using default background.", icon="🖼️")
        return
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp .st-emotion-cache-16idsys p {{
            color: #FFFFFF !important;
        }}
        /* Specific styling for bordered containers to ensure background is visible */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-of-type(2) > div,
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div.st-emotion-cache-1e5z8os,
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2) > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div.st-emotion-cache-1e5z8os,
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div.st-emotion-cache-1e5z8os {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }}

        .stTextInput label, .stSelectbox label, .stRadio label, .stSlider label {{
            color: #FFFFFF !important;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            color: #D0D0D0 !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: #FFFFFF !important;
            border-bottom: 2px solid #FFFFFF;
        }}
        /* Chat messages */
        .st-chat-message-container.st-chat-message-container-user {{
            background-color: rgba(0, 128, 255, 0.7) !important;
            color: white !important;
        }}
        .st-chat-message-container.st-chat-message-container-ai {{
            background-color: rgba(50, 50, 50, 0.8) !important;
            color: white !important;
        }}
        .st-chat-message-container p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- PDF Generation Helper ---
def create_quiz_pdf(quiz_data):
    """Generates a two-page PDF with questions and an answer key."""
    pdf = FPDF()

    # --- Page 1: Questions ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Quiz Questions', 0, 1, 'C')
    pdf.ln(10)

    # Get the initial left margin for indentation control
    initial_x = pdf.get_x()
    indent = 5 # Indent by 5 units

    for i, q in enumerate(quiz_data):
        # Use a consistent 'latin-1' encoding with replacement for unsupported characters
        question_text = f"Q{i+1}: {q['question']}".encode('latin-1', 'replace').decode('latin-1')
        options = [opt.encode('latin-1', 'replace').decode('latin-1') for opt in q['options']]

        pdf.set_font("Arial", 'B', 12)
        pdf.set_x(initial_x) # Ensure we start at the margin
        pdf.multi_cell(0, 6, question_text)
        pdf.ln(2)

        pdf.set_font("Arial", '', 12)
        for opt in options:
            pdf.set_x(initial_x)
            pdf.cell(indent)
            pdf.multi_cell(0, 6, f"- {opt}")
        pdf.ln(6)

    # --- Page 2: Answer Key ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Answer Key', 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    for i, q in enumerate(quiz_data):
        answer_text = f"Q{i+1}: {q['answer']}".encode('latin-1', 'replace').decode('latin-1')
        pdf.set_x(initial_x)
        pdf.multi_cell(0, 6, answer_text)
        pdf.ln(4)

    return bytes(pdf.output())


# --- Authentication UI ---
def show_login_form():
    """Displays a modern, centered login and registration form using tabs."""
    add_bg_from_local('background.jpg')
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True):
            st.markdown("<h1 style='text-align: center;'>🤖 Dialogix</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-bottom: 20px;'>Your AI-powered learning assistant.</p>", unsafe_allow_html=True)
            login_tab, register_tab = st.tabs(["**Login**", "**Register**"])
            with login_tab:
                with st.form(key="login_form"):
                    st.text_input("Username", placeholder="Enter your username", key="login_user")
                    st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                    st.markdown("</br>", unsafe_allow_html=True)
                    if st.form_submit_button(label="Login", use_container_width=True, type="primary"):
                        success, message = login_user(st.session_state["login_user"], st.session_state["login_pass"])
                        if success:
                            st.toast(message, icon="🎉")
                            st.rerun()
                        else:
                            st.error(message, icon="❌")
            with register_tab:
                with st.form(key="register_form"):
                    st.text_input("Username", placeholder="Create a new username", key="reg_user")
                    st.text_input("Password", type="password", placeholder="Create a strong password", key="reg_pass")
                    st.markdown("</br>", unsafe_allow_html=True)
                    if st.form_submit_button(label="Register", use_container_width=True):
                        success, message = register_user(st.session_state["reg_user"], st.session_state["reg_pass"])
                        if success:
                            st.success(message, icon="✅")
                        else:
                            st.error(message, icon="❌")

# --- Sidebar UI Components ---

def sidebar_session_selector():
    """Manages the chat session selection, renaming, and deletion in the sidebar."""
    st.sidebar.title("💬 My Chats")

    if "confirming_action_index" not in st.session_state:
        st.session_state.confirming_action_index = None
    if "editing_chat_index" not in st.session_state:
        st.session_state.editing_chat_index = None

    def handle_rename(index):
        new_name = st.session_state[f"rename_input_{index}"]
        if new_name and new_name != st.session_state.chat_session_names[index]:
            st.session_state.chat_session_names[index] = new_name
            save_user_data_from_session(st.session_state.username)
            st.toast(f"Chat renamed to '{new_name}'", icon="✏️")
        st.session_state.editing_chat_index = None
        st.rerun()

    def adjust_current_chat_after_action(acted_on_index):
        if st.session_state.current_chat == acted_on_index:
            st.session_state.current_chat = max(0, acted_on_index - 1)
            if not st.session_state.chat_session_names:
                create_new_chat_session(st.session_state.username)
                st.session_state.current_chat = 0
        elif st.session_state.current_chat > acted_on_index:
            st.session_state.current_chat -= 1

    for i, name in enumerate(st.session_state.chat_session_names):
        if st.session_state.confirming_action_index == i:
            with st.sidebar.container(border=True):
                st.warning(f"Action for **'{name}'**?")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Delete", key=f"confirm_delete_{i}", use_container_width=True, type="primary"):
                        deleted_name = st.session_state.chat_session_names[i]
                        delete_chat_session(st.session_state.username, i)
                        adjust_current_chat_after_action(i)
                        st.session_state.confirming_action_index = None
                        st.toast(f"Chat '{deleted_name}' deleted.", icon="🗑️")
                        st.rerun()
                with col2:
                    if st.button("Archive", key=f"confirm_archive_{i}", use_container_width=True):
                        archived_name = st.session_state.chat_session_names[i]
                        archive_chat_session(st.session_state.username, i)
                        adjust_current_chat_after_action(i)
                        st.session_state.confirming_action_index = None
                        st.toast(f"Chat '{archived_name}' archived.", icon="🗄️")
                        st.rerun()
                with col3:
                    if st.button("Cancel", key=f"cancel_action_{i}", use_container_width=True):
                        st.session_state.confirming_action_index = None
                        st.rerun()
        else:
            col1, col2, col3 = st.sidebar.columns([0.7, 0.15, 0.15])
            with col1:
                if st.session_state.get("editing_chat_index") == i:
                    st.text_input("Rename chat", value=name, key=f"rename_input_{i}", on_change=handle_rename, args=(i,), label_visibility="collapsed")
                else:
                    button_type = "primary" if st.session_state.current_chat == i and st.session_state.page == "chat" else "secondary"
                    if st.button(name, key=f"chat_{i}", use_container_width=True, type=button_type):
                        st.session_state.current_chat = i
                        st.session_state.page = "chat"
                        st.session_state.editing_chat_index = None
                        st.session_state.confirming_action_index = None
                        st.rerun()
            with col2:
                if st.session_state.get("editing_chat_index") == i:
                    pass
                else:
                    if st.button("✏️", key=f"edit_{i}", help="Rename chat"):
                        st.session_state.editing_chat_index = i
                        st.session_state.confirming_action_index = None
                        st.rerun()
            with col3:
                if st.session_state.get("editing_chat_index") != i:
                    if st.button("🗑️", key=f"delete_archive_{i}", help="Delete or Archive chat"):
                        st.session_state.confirming_action_index = i
                        st.session_state.editing_chat_index = None
                        st.rerun()

    if st.sidebar.button("➕ New Chat", use_container_width=True):
        create_new_chat_session(st.session_state.username)
        st.session_state.page = "chat"
        st.session_state.editing_chat_index = None
        st.session_state.confirming_action_index = None
        st.rerun()

    if st.session_state.get("archived_sessions"):
        with st.sidebar.expander("🗄️ Archived Chats"):
            if not st.session_state.archived_sessions:
                st.caption("No archived chats.")
            else:
                for idx, archived in enumerate(list(st.session_state.archived_sessions)):
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.write(f"{archived['name']}")
                    with col2:
                        if st.button("⬆️", key=f"restore_{idx}", help="Restore this chat"):
                            restore_chat_session(st.session_state.username, idx)
                            st.toast(f"Restored chat '{archived['name']}'!", icon="🎉")
                            st.rerun()

def show_pdf_manager_in_sidebar(state):
    """Renders the PDF uploader and manager in a sidebar expander."""
    if not hasattr(state, 'chat_pdf_paths') or not state.chat_pdf_paths:
        st.sidebar.expander("📄 PDF Management").info("Please create a chat session first to manage PDFs.")
        return

    idx = state.current_chat
    if idx >= len(state.chat_pdf_paths):
        state.current_chat = 0
        idx = 0
        st.rerun()

    chat_engine = state.chat_engines[idx]

    with st.sidebar.expander("📄 PDF Management", expanded=False):
        st.subheader("Add PDF to this Chat")

        uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"], key=f"pdf_uploader_{idx}")

        if uploaded_file:
            processed_pdf_names = [os.path.basename(p) for p in state.chat_pdf_paths[idx]]
            if uploaded_file.name not in processed_pdf_names:
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    handle_pdf_upload(state.username, uploaded_file, idx)
                    save_user_data_from_session(state.username)
                st.success(f"✅ PDF '{uploaded_file.name}' added and processed.")
                st.rerun()
            else:
                st.info(f"PDF '{uploaded_file.name}' is already associated with this chat.")

        st.markdown("---")
        st.subheader("Activate & Manage PDFs")

        if state.chat_pdf_paths and state.chat_pdf_paths[idx]:
            pdf_options = [os.path.basename(p) for p in state.chat_pdf_paths[idx]]

            active_pdf_name = None
            if hasattr(chat_engine, 'rag') and chat_engine.rag and chat_engine.rag.pdf_path:
                active_pdf_name = os.path.basename(chat_engine.rag.pdf_path)

            try:
                active_pdf_index = pdf_options.index(active_pdf_name) if active_pdf_name in pdf_options else 0
            except ValueError:
                active_pdf_index = 0

            selected_pdf_name = st.selectbox("Select a PDF to make it active:", pdf_options, index=active_pdf_index, key=f"select_active_pdf_{idx}")

            selected_pdf_path = next((p for p in state.chat_pdf_paths[idx] if os.path.basename(p) == selected_pdf_name), None)

            if selected_pdf_path and (not hasattr(chat_engine, 'rag') or not chat_engine.rag or chat_engine.rag.pdf_path != selected_pdf_path):
                with st.spinner(f"Activating '{selected_pdf_name}'..."):
                    ok, msg = chat_engine.attach_pdf(selected_pdf_path)
                    st.toast(f"Activated '{selected_pdf_name}'" if ok else msg, icon="✅" if ok else "❌")
                    if not ok:
                        st.error(f"Failed to activate PDF: {msg}")
            elif selected_pdf_path and hasattr(chat_engine, 'rag') and chat_engine.rag and chat_engine.rag.pdf_path == selected_pdf_path:
                st.info(f"'{selected_pdf_name}' is currently active.")
            elif not selected_pdf_path:
                st.warning("Selected PDF path could not be resolved.")

            st.markdown("---")

            st.write(f"**Delete selected PDF:** `{selected_pdf_name}`")
            if st.button("🗑️ Delete this PDF", type="secondary", use_container_width=True, key=f"delete_pdf_{idx}_{selected_pdf_name}"):
                if selected_pdf_path:
                    if hasattr(chat_engine, 'rag') and chat_engine.rag and chat_engine.rag.pdf_path == selected_pdf_path:
                        chat_engine.rag = None
                        st.toast(f"Deactivated '{selected_pdf_name}' as it's being deleted.", icon="ℹ️")

                    success, message = delete_pdf_for_user(selected_pdf_path)
                    if success:
                        state.chat_pdf_paths[idx].remove(selected_pdf_path)
                        save_user_data_from_session(state.username)
                        st.toast(message, icon="🗑️")
                        st.rerun()
                    else:
                        st.toast(message, icon="❌")

                else:
                    st.warning("No PDF selected for deletion or path not found.")
        else:
            st.info("No PDFs have been uploaded for this chat yet. Upload one above!")


def sidebar_navigation():
    """Adds navigation buttons to the sidebar for settings and quiz generator."""
    st.sidebar.markdown("---")
    if st.sidebar.button("💬 Back to Chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()

    if st.sidebar.button("🧠 Quiz Generator", use_container_width=True):
        st.session_state.page = "quiz"
        st.rerun()

    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state.page = "settings"
        st.rerun()


# --- Main Page UI ---

def stream_response(response):
    """Yields words from a response string with a delay to simulate typing."""
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def show_chat_page(state):
    """Renders the main chat interface, including messages and input controls."""

    # --- Initialize session state ---
    if "response_handled" not in st.session_state:
        st.session_state.response_handled = False
    if "audio_played" not in st.session_state:
        st.session_state.audio_played = False
    if "generated_response" not in st.session_state:
        st.session_state.generated_response = None

    # --- Ensure valid chat index ---
    if state.current_chat >= len(state.chat_engines):
        if len(state.chat_engines) > 0:
            state.current_chat = 0
        else:
            create_new_chat_session(state.username)
            state.current_chat = 0
            st.rerun()

    chat_index = state.current_chat
    chat_engine = state.chat_engines[chat_index]

    # --- Show welcome message if needed ---
    if not state.chat_sessions[chat_index]:
        welcome_message()

    # --- Show chat history ---
    for msg in state.chat_sessions[chat_index]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- User text input ---
    user_text_input = st.chat_input("Type your message here...", key="chat_text_input")
    if user_text_input:
        state.chat_sessions[chat_index].append({"role": "user", "content": user_text_input})
        st.session_state.response_handled = False
        st.session_state.generated_response = None
        st.session_state.audio_played = False
        st.rerun()

    # --- Generate assistant response only if needed ---
    if (
        state.chat_sessions[chat_index]
        and state.chat_sessions[chat_index][-1]["role"] == "user"
        and not st.session_state.response_handled
    ):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.get_response(state.chat_sessions[chat_index][-1]["content"])
                st.write_stream(stream_response(response))

        state.chat_sessions[chat_index].append({"role": "assistant", "content": response})
        save_user_data_from_session(state.username)

        st.session_state.generated_response = response
        st.session_state.response_handled = True
        st.session_state.audio_played = False
        st.rerun()

    # --- Playback assistant audio (after rerun only) ---
    if st.session_state.generated_response and not st.session_state.audio_played:
        if state.user_config.get("elevenlabs_api"):
            try:
                speak_text(st.session_state.generated_response, state.user_config["elevenlabs_api"])
                st.session_state.audio_played = True
            except Exception as e:
                st.error(f"Failed to play AI voice response: {e}")

    # --- Voice input section ---
    with st.container(border=True):
        st.markdown("##### Or speak your message:")
        wav_audio_data = audio_recorder(text="", key="audio_recorder")

        if wav_audio_data is not None:
            with st.spinner("Transcribing audio..."):
                whisper_model_to_use = state.user_config.get("whisper_model", "tiny")
                transcription = transcribe_audio(wav_audio_data, whisper_model_to_use)

            if transcription:
                state.chat_sessions[chat_index].append({"role": "user", "content": transcription})
                st.session_state.response_handled = False
                st.session_state.generated_response = None
                st.session_state.audio_played = False
                st.rerun()


# --- Settings and Quiz Page UI ---

def show_settings_page(state):
    """Renders the user-specific application settings on a dedicated page."""
    st.title("⚙️ Your Settings")

    with st.container(border=True):
        st.header("System Prompt")
        # MODIFIED: Use st.session_state.user_config instead of st.session_state.config
        state.user_config["system_prompt"] = st.text_area(
            "This prompt guides the AI's personality and responses.",
            value=state.user_config.get("system_prompt", "You are a helpful AI assistant."),
            height=150,
            key="system_prompt_settings"
        )

    with st.container(border=True):
        st.header("Whisper Model")
        whisper_models = ["tiny", "base", "small", "medium", "large", "large-v2"]
        # MODIFIED: Use st.session_state.user_config for whisper model
        current_whisper_model = state.user_config.get("whisper_model", "tiny")
        if current_whisper_model not in whisper_models:
            current_whisper_model = "tiny"

        state.user_config["whisper_model"] = st.selectbox(
            "Select the model size for audio transcription. Larger models are more accurate but slower.",
            whisper_models,
            index=whisper_models.index(current_whisper_model),
            key="whisper_model_settings"
        )
        st.caption("Note: Larger models require more memory and may take longer to download and process.")

    with st.container(border=True):
        st.header("ElevenLabs API Key")
        # MODIFIED: Use st.session_state.user_config for ElevenLabs API key
        state.user_config["elevenlabs_api"] = st.text_input(
            "Enter your API key for text-to-speech functionality.",
            value=state.user_config.get("elevenlabs_api", ""),
            type="password",
            key="elevenlabs_api_settings"
        )
        st.caption("Get your ElevenLabs API key from [ElevenLabs](https://elevenlabs.io/)", unsafe_allow_html=True)


    if st.button("💾 Save Settings", use_container_width=True, type="primary"):
        # MODIFIED: Call save_user_data_from_session to save user-specific config
        save_user_data_from_session(state.username)
        st.success("✅ Settings saved successfully.")
        st.toast("Settings have been updated!", icon="👍")


def show_quiz_page(state):
    """Renders the quiz generation page and handles quiz logic."""
    st.title("🧠 Quiz Generator")
    st.markdown("Create a quiz from a topic or a PDF document.")

    if 'page' in st.session_state and st.session_state.page != 'quiz':
        keys_to_clear = ['quiz_data', 'user_answers', 'show_score']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = 'quiz'

    if not st.session_state.get('quiz_data'):
        with st.container(border=True):
            st.subheader("1. Configure Your Quiz")
            source_type = st.radio("Quiz Source:", ("Topic", "PDF"), horizontal=True, key="quiz_source")
            col1, col2 = st.columns(2)
            with col1:
                difficulty = st.select_slider("Difficulty:", ("Easy", "Medium", "Hard"), value="Medium", key="quiz_difficulty")
            with col2:
                num_questions = st.number_input("Number of Questions:", min_value=3, max_value=15, value=5, key="quiz_num_questions")

            source_input_valid = False
            topic = ""
            pdf_path = None

            if source_type == "Topic":
                topic = st.text_input("Enter quiz topic:", placeholder="e.g., 'The History of Space Exploration'", key="quiz_topic_input")
                if topic:
                    source_input_valid = True
            else:
                current_chat_pdfs = st.session_state.chat_pdf_paths[st.session_state.current_chat]
                if not current_chat_pdfs:
                    st.warning("No PDFs in this chat. Please upload one via the sidebar.", icon="⚠️")
                else:
                    pdf_options = {os.path.basename(p): p for p in current_chat_pdfs}
                    selected_pdf_name = st.selectbox("Select a PDF:", options=list(pdf_options.keys()), key="quiz_pdf_select")
                    pdf_path = pdf_options.get(selected_pdf_name)
                    if pdf_path:
                        source_input_valid = True

            # MODIFIED: Pass st.session_state.user_config to QuizGenerator
            if st.button("✨ Generate Quiz", use_container_width=True, type="primary", disabled=not source_input_valid):
                generator = QuizGenerator(st.session_state.user_config)
                with st.spinner("Generating your quiz... This may take a moment."):
                    if source_type == "Topic":
                        response = generator.generate_from_topic(topic, difficulty, num_questions)
                    else:
                        response = generator.generate_from_pdf(pdf_path, difficulty, num_questions)

                try:
                    cleaned_str = response.strip()
                    if cleaned_str.startswith("```json"):
                        cleaned_str = cleaned_str[len("```json"):].strip()
                    if cleaned_str.endswith("```"):
                        cleaned_str = cleaned_str[:-len("```")].strip()

                    quiz_data = json.loads(cleaned_str)

                    if isinstance(quiz_data, list) and all('question' in q and 'options' in q and 'answer' in q for q in quiz_data):
                        st.session_state.quiz_data = quiz_data
                        st.session_state.user_answers = [None] * len(quiz_data)
                        st.session_state.show_score = False
                        st.rerun()
                    else:
                        st.error("The AI returned an invalid quiz format. Please ensure it contains questions, options, and answers.", icon="🚨")
                        st.code(response)
                except (json.JSONDecodeError, TypeError) as e:
                    st.error(f"Failed to parse the quiz from the AI's response: {e}. Check the AI's output format.", icon="🚨")
                    st.code(response)

    if st.session_state.get('quiz_data') and not st.session_state.get('show_score'):
        with st.container(border=True):
            st.subheader("2. Take the Quiz")
            with st.form("quiz_form"):
                user_answers = []
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**Question {i+1}: {q['question']}**")
                    options = q.get('options', [])
                    if q['answer'] not in options:
                        options.append(q['answer'])
                    import random
                    random.shuffle(options)
                    user_choice = st.radio("Options:", options, key=f"q_{i}", label_visibility="collapsed")
                    user_answers.append(user_choice)

                st.markdown("---")
                if st.form_submit_button("Submit Answers", use_container_width=True, type="primary"):
                    st.session_state.user_answers = user_answers
                    st.session_state.show_score = True
                    st.rerun()

    if st.session_state.get('show_score'):
        with st.container(border=True):
            st.subheader("3. Your Results")
            quiz_data = st.session_state.quiz_data
            user_answers = st.session_state.user_answers
            score = sum(1 for i, u_ans in enumerate(user_answers) if u_ans == quiz_data[i]['answer'])
            st.metric(label="Your Score", value=f"{score}/{len(quiz_data)}", delta=f"{score/len(quiz_data)*100:.1f}%")
            with st.expander("📝 Review Your Answers", expanded=False):
                for i, q in enumerate(quiz_data):
                    user_ans = user_answers[i]
                    correct_ans = q['answer']
                    if user_ans == correct_ans:
                        st.success(f"**Q{i+1}: Correct!** {q['question']}", icon="✅")
                    else:
                        st.error(f"**Q{i+1}: Incorrect!** {q['question']}", icon="❌")
                        st.info(f"Your answer: `{user_ans}` | Correct answer: `{correct_ans}`")
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("↻ Retry Quiz", use_container_width=True):
                    keys_to_clear = ['user_answers', 'show_score']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            with col2:
                if st.button("🎉 Create New Quiz", use_container_width=True, type="primary"):
                    keys_to_clear = ['quiz_data', 'user_answers', 'show_score']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            with col3:
                pdf_data = create_quiz_pdf(st.session_state.quiz_data)
                st.download_button(label="📄 Download PDF", data=pdf_data, file_name="quiz_with_answers.pdf", mime="application/pdf", use_container_width=True)

def welcome_message():
    """Displays a welcome message in the main chat area."""
    st.markdown("## 👋 Welcome to Dialogix!")
    st.markdown("Start a conversation by typing below, or upload a PDF in the sidebar to chat with your document.")