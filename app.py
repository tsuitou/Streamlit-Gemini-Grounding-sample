import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import content_types
from dotenv import load_dotenv
from pathlib import Path

# 拡張子 → MIME の対応表
EXTENSION_TO_MIME = {
    # ドキュメント系
    "pdf": "application/pdf",
    "js": "application/x-javascript",  # または "text/javascript"
    "py": "text/x-python",            # または "application/x-python"
    "css": "text/css",
    "md": "text/md",
    "csv": "text/csv",
    "xml": "text/xml",
    "rtf": "text/rtf",
    "txt": "text/plain",

    # 画像
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",

    # 動画
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "mov": "video/mov",      # 一般的には "video/quicktime" も利用される
    "avi": "video/avi",      # 一般的には "video/x-msvideo" も利用される
    "flv": "video/x-flv",
    "mpg": "video/mpg",
    "webm": "video/webm",
    "wmv": "video/wmv",      # 一般的には "video/x-ms-wmv" も利用される
    "3gpp": "video/3gpp",

    # 音声
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}

# 環境変数の読み込み (.env に GOOGLE_API_KEY, MODELS 等を記載)
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# data/ フォルダがなければ作成
os.makedirs('data/', exist_ok=True)

# 過去チャットの読み込み（あれば）
try:
    past_chats = joblib.load('data/past_chats_list')
except Exception:
    past_chats = {}

# -----------------------------
# サイドバー
# -----------------------------
with st.sidebar:
    # グラウンディング切り替えボタン
    if 'grounding_enabled' not in st.session_state:
        st.session_state.grounding_enabled = False
    if st.button("グラウンディング：" + ("有効" if st.session_state.grounding_enabled else "無効")):
        st.session_state.grounding_enabled = not st.session_state.grounding_enabled
        st.rerun()

    new_chat_id = f'{time.time()}'
    chat_ids = sorted(past_chats.keys(), key=float, reverse=True)
    chat_options = ['New Chat'] + chat_ids

    if 'selected_chat' not in st.session_state:
        st.session_state.selected_chat = 'New Chat'
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    selected_chat = st.selectbox(
        label='履歴',
        options=chat_options,
        index=st.session_state.selected_index,
        format_func=lambda x: past_chats.get(x, 'New Chat') if x != 'New Chat' else 'New Chat'
    )

    st.session_state.selected_chat = selected_chat

    if selected_chat == 'New Chat':
        st.session_state.chat_id = new_chat_id
        st.session_state.chat_title = f'ChatSession-{new_chat_id}'
    else:
        st.session_state.chat_id = selected_chat
        st.session_state.chat_title = past_chats[selected_chat]

    if selected_chat != 'New Chat':
        if st.button('Delete this chat'):
            try:
                os.remove(f'data/{selected_chat}-st_messages')
                os.remove(f'data/{selected_chat}-gemini_messages')
            except FileNotFoundError:
                pass
            del past_chats[selected_chat]
            joblib.dump(past_chats, 'data/past_chats_list')

            # 削除後は最初の項目（New Chat）を選択
            st.session_state.selected_chat = 'New Chat'
            st.session_state.selected_index = 0
            st.rerun()
    # ====================================
    # ファイルアップローダー
    # ====================================
    allowed_extensions = list(EXTENSION_TO_MIME.keys())
    new_file = st.file_uploader(
        label="添付",
        type= allowed_extensions,
        accept_multiple_files=False,
    )


# モデル選択機能
try:
    api_models = genai.list_models()
    api_model_names = [m.name for m in api_models]
except Exception as e:
    st.error("モデル一覧の取得に失敗しました: " + str(e))
    api_model_names = []

additional_models = os.environ.get("MODELS", "").split(",")
combined_models = sorted(set(api_model_names + [m.strip() for m in additional_models if m.strip()]))

default_index = 0
selected_model = st.selectbox("モデル選択", combined_models, index=default_index)

# チャット履歴の読み込み
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except Exception:
    st.session_state.messages = []
    st.session_state.gemini_history = []

# モデルとチャットセッションの初期化
st.session_state.model = genai.GenerativeModel(selected_model)
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

# 過去メッセージの表示
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(name=message['role'], avatar=message.get('avatar')):
        st.markdown(message['content'])
    # 「削除」ボタン
    if selected_chat != 'New Chat':
        if st.button(f"Delete", key=f"reset_{i}"):
            if i == 0:
                try:
                    os.remove(f'data/{selected_chat}-st_messages')
                    os.remove(f'data/{selected_chat}-gemini_messages')
                except FileNotFoundError:
                    pass
                del past_chats[selected_chat]
                joblib.dump(past_chats, 'data/past_chats_list')
                st.session_state.selected_chat = 'New Chat'
                st.session_state.selected_index = 0
                st.rerun()
            st.session_state.messages = st.session_state.messages[:i]
            st.session_state.gemini_history = st.session_state.gemini_history[:i]
            joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
            joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
            st.rerun()

# ユーザー入力欄
if prompt := st.chat_input('Your message here...'):
    # New Chat の場合、タイトルを設定
    if st.session_state.chat_id not in past_chats:
        chat_title = prompt[:20]
        past_chats[st.session_state.chat_id] = chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

        st.session_state.selected_chat = st.session_state.chat_id
        st.session_state.selected_index = 1

    # 1. ユーザーメッセージの表示
    with st.chat_message('user'):
        # もしアップロードされたファイルがあれば、会話に表示する
        file_list_markdown = ''
        if new_file:
            file_list_markdown = "\n - " + new_file.name
            
        st.markdown(prompt + file_list_markdown)
        
    st.session_state.messages.append({'role': 'user', 'content': prompt + file_list_markdown})


    # 2. モデルへ問い合わせ
    if st.session_state.grounding_enabled:
        response = st.session_state.chat.send_message(prompt, stream=True, tools='google_search_retrieval')
    else:
        if new_file:
            file_data = new_file.read()  # バイナリデータを読み取り
            # ファイル名・拡張子から MIME タイプを推定
            extension = Path(new_file.name).suffix.lower().lstrip('.')
            mime_type = EXTENSION_TO_MIME.get(extension, "application/octet-stream")
            # Part.from_bytes でコンテンツオブジェクトを作成（1つ目がファイルデータ）
            file_part = content_types.to_part({
                "mime_type": mime_type,
                "data": file_data,
            })
            contents = [file_part, prompt]
            response = st.session_state.chat.send_message(content=contents, stream=True)
        else:
            response = st.session_state.chat.send_message(prompt, stream=True)

    # 3. 応答をストリーミングで受け取り、チャットに表示
    with st.chat_message(name='ai', avatar='✨'):
        message_placeholder = st.empty()
        full_response = ''
        response_chunks = []
        for chunk in response:
            response_chunks.append(chunk)
            full_response += chunk.text
            time.sleep(0.05)
            message_placeholder.write(full_response + '▌')
        message_placeholder.write(full_response)

        # groundingのメタデータ対応が必要な場合の例
        if st.session_state.grounding_enabled:
            all_grounding_links = ""
            all_grounding_queries = ""
            for chunk in response_chunks:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            for i, grounding_chunk in enumerate(metadata.grounding_chunks):
                                if hasattr(grounding_chunk, 'web') and grounding_chunk.web:
                                    all_grounding_links += f"[{i + 1}][{grounding_chunk.web.title}]({grounding_chunk.web.uri}) "
                        if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                            for query in metadata.web_search_queries:
                                all_grounding_queries += f"{query} / "
            all_grounding_queries = " / ".join(sorted(set(all_grounding_queries.rstrip(" /").split(" / "))))
            formatted_metadata = "\n\n---\n"
            if all_grounding_links:
                formatted_metadata += all_grounding_links + "\n"
            if all_grounding_queries:
                formatted_metadata += "Query: " + all_grounding_queries + "\n"
            full_response += formatted_metadata
            message_placeholder.write(full_response)

    st.session_state.messages.append({'role': 'ai', 'content': full_response, 'avatar': '✨'})
    st.session_state.gemini_history = st.session_state.chat.history

    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
    st.rerun()
