import streamlit as st
from modelo import respuesta
from streamlit_chat import message

st.markdown(
    """
    <style>
    div[data-baseweb="card"] {
        background-color: #c3e6e5;
    }
    div[data-baseweb="input"] {
        background-color: #eef5f4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

avatar = {
    'user': 'lorelei',
    'assistant': 'pixel-art'
}

st.title("Asistente de Seguros 🛡️")

# Divide la pantalla en dos columnas
col1, col2 = st.columns([1, 1])

# Coloca el chat en la columna izquierda
with col1:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hola! soy tu robot personal ¿En qué te puedo ayudar hoy?"}]

    with st.form("chat_input", clear_on_submit=True):
        user_input = st.text_input(
            key='user_message',
            label="Tu consulta:",
            placeholder="Escribe aquí tu consulta",
            )
        st.form_submit_button("ENVIAR", use_container_width=True)

    placeholder = st.empty()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with placeholder.container():
            with st.spinner('Cargando...'):
                msg = respuesta(user_input)
        st.session_state.messages.append({'role': 'assistant', 'content': msg})

    for i, msg in enumerate(reversed(st.session_state.messages)):
        message(msg["content"],
                is_user=msg["role"] == "user",
                key=str(i),
                avatar_style=avatar[msg["role"]])

with col2:
    st.image('./seguros-figura-615x.webp')
