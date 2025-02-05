import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

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



# サイドバー：過去チャットの一覧を表示
with st.sidebar:
    # -----------------------------
    # グラウンディングの有効化・無効化ボタン
    # -----------------------------
    if 'grounding_enabled' not in st.session_state:
        st.session_state.grounding_enabled = False
    # ボタンラベルは現在の状態に応じて「有効化」または「無効化」と表示
    if st.button("Grounding : \n" + ("Enabled" if st.session_state.grounding_enabled else "Disabled")):
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
        label='History',
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

    # 履歴削除ボタン
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


# モデル選択機能一時的に無効化
#try:
#    api_models = genai.list_models()
#    api_model_names = [m.name for m in api_models]
#except Exception as e:
#    st.error("モデル一覧の取得に失敗しました: " + str(e))
#    api_model_names = []

additional_models = os.environ.get("MODELS", "").split(",")
#combined_models = sorted(set(api_model_names + [m.strip() for m in additional_models if m.strip()]))

default_index = 0
selected_model = st.selectbox("Model", additional_models, index=default_index)


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

        # 「削除」ボタンを追加
        if st.button(f"Delete", key=f"reset_{i}"):
            # 選択されたメッセージ以降の履歴を削除
            st.session_state.messages = st.session_state.messages[:i]
            st.session_state.gemini_history = st.session_state.gemini_history[:i]

            # 履歴の保存
            joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
            joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

            # 再描画
            st.rerun()

# ユーザー入力
if prompt := st.chat_input('Your message here...'):
    # 新しいチャットならタイトルを更新して保存
    if st.session_state.chat_id not in past_chats:
        chat_title = prompt[:20]
        past_chats[st.session_state.chat_id] = chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

        # コンボボックスの更新
        st.session_state.selected_chat = st.session_state.chat_id
        st.session_state.selected_index = 1  # 新しいチャットが2番目に表示されるため

    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # グラウンディングが有効なら、tools 引数に 'google_search_retrieval' を追加
    if st.session_state.grounding_enabled:
        response = st.session_state.chat.send_message(prompt, stream=True, tools='google_search_retrieval')
    else:
        response = st.session_state.chat.send_message(prompt, stream=True)

    with st.chat_message(name='ai', avatar='✨'):
        message_placeholder = st.empty()
        full_response = ''
        response_chunks = []  # 各チャンクオブジェクトを保持
        # ストリーミングでチャンクを受信し、full_response を構築
        for chunk in response:
            response_chunks.append(chunk)
            full_response += chunk.text
            time.sleep(0.05)
            message_placeholder.write(full_response + '▌')
        message_placeholder.write(full_response)

        # response_chunks からメタデータを探す
        if st.session_state.grounding_enabled:
            # 複数チャンクにわたるメタデータを統合
            all_grounding_links = ""
            all_grounding_queries = ""

            for chunk in response_chunks:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata

                        # リンク情報の収集
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            for i, grounding_chunk in enumerate(metadata.grounding_chunks):
                                if hasattr(grounding_chunk, 'web') and grounding_chunk.web:
                                    all_grounding_links += f"[{i + 1}][{grounding_chunk.web.title}]({grounding_chunk.web.uri}) "

                        # クエリ情報の収集
                        if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                            for query in metadata.web_search_queries:
                                all_grounding_queries += f"{query} / "

            # 重複を排除
            all_grounding_queries = " / ".join(sorted(set(all_grounding_queries.rstrip(" /").split(" / "))))

            # メタデータの成形
            formatted_metadata = "\n\n---\n"
            if all_grounding_links:
                formatted_metadata += all_grounding_links + "\n"
            if all_grounding_queries:
                formatted_metadata += "Query: " + all_grounding_queries + "\n"

            # 応答メッセージにメタデータを付加
            full_response += formatted_metadata
            message_placeholder.write(full_response)

    st.session_state.messages.append({'role': 'ai', 'content': full_response, 'avatar': '✨'})
    st.session_state.gemini_history = st.session_state.chat.history


    # 履歴の保存
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

    # 最後に再描画して選択状態を維持
    st.rerun()
