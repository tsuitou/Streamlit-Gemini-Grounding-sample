import time
import os
import joblib
import json
import bcrypt
import re
import streamlit as st
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv
from pathlib import Path

# 拡張子 → MIME の対応表
EXTENSION_TO_MIME = {
    # ドキュメント系
    'pdf': 'application/pdf',
    'js': 'application/x-javascript',  # または 'text/javascript'
    'py': 'text/x-python',            # または 'application/x-python'
    'css': 'text/css',
    'md': 'text/md',
    'csv': 'text/csv',
    'xml': 'text/xml',
    'rtf': 'text/rtf',
    'txt': 'text/plain',

    # 画像
    'png': 'image/png',
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'webp': 'image/webp',
    'heic': 'image/heic',
    'heif': 'image/heif',

    # 動画
    'mp4': 'video/mp4',
    'mpeg': 'video/mpeg',
    'mov': 'video/mov',      # 一般的には 'video/quicktime' も利用される
    'avi': 'video/avi',      # 一般的には 'video/x-msvideo' も利用される
    'flv': 'video/x-flv',
    'mpg': 'video/mpg',
    'webm': 'video/webm',
    'wmv': 'video/wmv',      # 一般的には 'video/x-ms-wmv' も利用される
    '3gpp': 'video/3gpp',

    # 音声
    'wav': 'audio/wav',
    'mp3': 'audio/mp3',
    'aiff': 'audio/aiff',
    'aac': 'audio/aac',
    'ogg': 'audio/ogg',
    'flac': 'audio/flac',
}

# 環境変数の読み込み (.env に GOOGLE_API_KEY, MODELS 等を記載)
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
st.session_state.client = genai.Client(api_key=GOOGLE_API_KEY)
google_search_tool = Tool(
    google_search = GoogleSearch()
)

# data/ フォルダがなければ作成
os.makedirs('data/', exist_ok=True)

# ユーザー情報を保存するJSONファイル
ACCOUNT_FILE = 'data/accounts.json'

def load_accounts():
    if os.path.exists(ACCOUNT_FILE):
        with open(ACCOUNT_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return {}  # ファイルが空だった場合
    return {}


def save_accounts(accounts):
    with open(ACCOUNT_FILE, 'w') as f:
        json.dump(accounts, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, password):
    if username == '' or password == '':
        st.error("ユーザー名かパスワードが空欄です")
        return False
    if any(not re.match(r'^[a-zA-Z0-9]*$', field) for field in (username, password)):
        st.error("英数字以外の文字が含まれています。")
        return False
    accounts = load_accounts()
    if username in accounts:
        return False  # 既存のユーザー名
    accounts[username] = hash_password(password)
    save_accounts(accounts)
    return True

def authenticate(username, password):
    accounts = load_accounts()
    if username in accounts and verify_password(password, accounts[username]):
        return True
    return False

def find_gemini_index(messages, target_user_messages):
    user_count = 0
    for idx, content in enumerate(messages):
        if content.role == 'user':
            user_count += 1
            if user_count == target_user_messages:
                # ユーザーメッセージに対応する応答の最後まで含める
                while idx + 1 < len(messages) and messages[idx + 1].role == 'model':
                    idx += 1
                return idx + 1
    return len(messages)

# ユーザー認証
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ''

# ログイン画面の表示
if not st.session_state.authenticated:
    st.title('ログイン')
    username = st.text_input('ユーザー名')
    password = st.text_input('パスワード', type='password')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('ログイン'):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error('ログイン失敗')
    
    with col2:
        if st.button('新規登録'):
            if register_user(username, password):
                st.success('登録完了！ログインしてください。')
            else:
                st.error('エラー')
    
    st.stop()

# ユーザーデータ保存ディレクトリ
user_dir = f'data/{st.session_state.username}/'
os.makedirs(user_dir, exist_ok=True)

# 過去チャットの読み込み（ユーザーごとに保存）
past_chats_file = f'{user_dir}past_chats_list'
try:
    past_chats = joblib.load(past_chats_file)
except Exception:
    past_chats = {}

# モデル選択機能
try:
    api_models = st.session_state.client.models.list()
    api_model_names = [m.name for m in api_models]
except Exception as e:
    st.error('モデル一覧の取得に失敗しました: ' + str(e))
    api_model_names = []

additional_models = os.environ.get('MODELS', '').split(',')
combined_models = sorted(set(api_model_names + [m.strip() for m in additional_models if m.strip()]))

default_index = 0
selected_model = st.selectbox('モデル選択', combined_models, index=default_index)

# -----------------------------
# サイドバー
# -----------------------------
with st.sidebar:
    # グラウンディング切り替えボタン
    grounding_flag = False
    if 'grounding_enabled' not in st.session_state:
        st.session_state.grounding_enabled = False
    if st.button('グラウンディング：' + ('有効' if st.session_state.grounding_enabled else '無効')):
        st.session_state.grounding_enabled = not st.session_state.grounding_enabled
        grounding_flag = True

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
        if st.button('このチャットを削除'):
            try:
                os.remove(f'{user_dir}{selected_chat}-st_messages')
                os.remove(f'{user_dir}{selected_chat}-gemini_messages')
            except FileNotFoundError:
                pass
            del past_chats[selected_chat]
            joblib.dump(past_chats, f'{user_dir}past_chats_list')

            # 削除後は最初の項目（New Chat）を選択
            st.session_state.selected_chat = 'New Chat'
            st.session_state.selected_index = 0
            st.rerun()
    # ====================================
    # ファイルアップローダー
    # ====================================

    allowed_extensions = list(EXTENSION_TO_MIME.keys())
    new_file = st.file_uploader(
        label='添付',
        type= allowed_extensions,
        accept_multiple_files=False,
    )
    if new_file:
        file_data = new_file.read()  # バイナリデータを読み取り
        # ファイル名・拡張子から MIME タイプを推定
        extension = Path(new_file.name).suffix.lower().lstrip('.')
        mime_type = EXTENSION_TO_MIME.get(extension, 'application/octet-stream')
        # Part.from_bytes でコンテンツオブジェクトを作成（1つ目がファイルデータ）
        file_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type,
        )
        if st.button('トークン数確認'):
            st.session_state.token = '添付ファイルのトークン：' + str(st.session_state.client.models.count_tokens(model=selected_model,contents=file_part,).total_tokens)
    else:
        st.session_state.token = ''
    
    st.markdown(st.session_state.token)
    if grounding_flag:
        st.rerun()

# チャット履歴の読み込み
try:
    st.session_state.messages = joblib.load(f'{user_dir}{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'{user_dir}{st.session_state.chat_id}-gemini_messages')
except Exception:
    st.session_state.messages = []
    st.session_state.gemini_history = []

# モデルとチャットセッションの初期化
st.session_state.chat = st.session_state.client.chats.create(model=selected_model, history=st.session_state.gemini_history)

# 過去メッセージの表示
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(name=message['role'], avatar=message.get('avatar')):
        st.markdown(message['content'])
    # 「削除」ボタン
    if selected_chat != 'New Chat':
        if message['role'] == 'user':
            if st.button(f'Delete', key=f'reset_{i}'):
                if i == 0:
                    try:
                        os.remove(f'{user_dir}{selected_chat}-st_messages')
                        os.remove(f'{user_dir}{selected_chat}-gemini_messages')
                    except FileNotFoundError:
                        pass
                    del past_chats[selected_chat]
                    joblib.dump(past_chats, f'{user_dir}past_chats_list')
                    st.session_state.selected_chat = 'New Chat'
                    st.session_state.selected_index = 0
                    st.rerun()
                st.session_state.messages = st.session_state.messages[:i]
                target_user_messages = sum(1 for msg in st.session_state.messages[:i] if msg['role'] == 'user')
                gemini_index = find_gemini_index(st.session_state.gemini_history, target_user_messages)
                st.session_state.gemini_history = st.session_state.gemini_history[:gemini_index]
                
                joblib.dump(st.session_state.messages, f'{user_dir}{st.session_state.chat_id}-st_messages')
                joblib.dump(st.session_state.gemini_history, f'{user_dir}{st.session_state.chat_id}-gemini_messages')
                st.rerun()

# ユーザー入力欄
if prompt := st.chat_input('Your message here...'):
    # New Chat の場合、タイトルを設定
    if st.session_state.chat_id not in past_chats:
        chat_title = prompt[:20]
        past_chats[st.session_state.chat_id] = chat_title
        joblib.dump(past_chats, f'{user_dir}past_chats_list')

        st.session_state.selected_chat = st.session_state.chat_id
        st.session_state.selected_index = 1

    # 1. ユーザーメッセージの表示
    with st.chat_message('user'):
        # もしアップロードされたファイルがあれば、会話に表示する
        file_list_markdown = ''
        if new_file:
            file_list_markdown = '\n - ' + new_file.name
        st.markdown(prompt + file_list_markdown)
        
    st.session_state.messages.append({'role': 'user', 'content': prompt + file_list_markdown})


    # 2. モデルへ問い合わせ
    if st.session_state.grounding_enabled:
        configs = GenerateContentConfig(tools=[google_search_tool],response_modalities=["TEXT"],)
        response = st.session_state.chat.send_message_stream(message=prompt,config=configs)
        if new_file:
            contents = [file_part, prompt]
            response = st.session_state.chat.send_message_stream(message=contents,config=configs)
        else:
            response = st.session_state.chat.send_message_stream(message=prompt,config=configs)
    else:
        if new_file:
            contents = [file_part, prompt]
            response = st.session_state.chat.send_message_stream(message=contents)
        else:
            response = st.session_state.chat.send_message_stream(message=prompt)

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
            all_grounding_links = ''
            all_grounding_queries = ''
            for chunk in response_chunks:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            for i, grounding_chunk in enumerate(metadata.grounding_chunks):
                                if hasattr(grounding_chunk, 'web') and grounding_chunk.web:
                                    all_grounding_links += f'[{i + 1}][{grounding_chunk.web.title}]({grounding_chunk.web.uri}) '
                        if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                            for query in metadata.web_search_queries:
                                all_grounding_queries += f'{query} / '
            all_grounding_queries = ' / '.join(sorted(set(all_grounding_queries.rstrip(' /').split(' / '))))
            formatted_metadata = '\n\n---\n'
            if all_grounding_links:
                formatted_metadata += all_grounding_links + '\n'
            if all_grounding_queries:
                formatted_metadata += '\nクエリ：' + all_grounding_queries + '\n'
            full_response += formatted_metadata
            message_placeholder.write(full_response)
            #if st.session_state.chat._curated_history and st.session_state.chat._curated_history[-1].role == "model":
            #    # parts が複数ある可能性を考慮して、ループで処理
            #    for part in st.session_state.chat._curated_history[-1].parts:
            #        if hasattr(part, 'text'): # text 属性があるか確認
            #            part.text += formatted_metadata

    st.session_state.messages.append({'role': 'ai', 'content': full_response, 'avatar': '✨'})
    st.session_state.gemini_history = st.session_state.chat._curated_history

    joblib.dump(st.session_state.messages, f'{user_dir}{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'{user_dir}{st.session_state.chat_id}-gemini_messages')
    st.rerun()
