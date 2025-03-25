import os
import yaml
import chainlit as cl
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI  
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import asyncio

# 🔹 Chargement sécurisé des variables d'environnement
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")  # Fallback sécurisé

if not MISTRAL_API_KEY:
    raise ValueError("❌ Clé API Mistral manquante. Ajoutez-la dans votre fichier .env")

# 🔹 Chargement des prompts depuis YAML
with open("prompts.yaml", "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)["prompts"]

# 🔹 Initialisation de la mémoire conversationnelle
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔹 Gestion du streaming avec un Callback asynchrone
async def interact_with_llm(user_input):
    history = memory.load_memory_variables({}).get("chat_history", [])
    
    # Formatage des messages pour LangChain
    messages = [SystemMessage(content=prompts["system"]["content"])] + history
    messages.append(HumanMessage(content=user_input))

    # Création du callback pour gérer le streaming
    callback = AsyncIteratorCallbackHandler()
    
    # Initialisation du modèle avec le callback
    llm = ChatMistralAI(model="mistral-large-2411", api_key=MISTRAL_API_KEY, streaming=True, callbacks=[callback])

    # Envoi de la requête en arrière-plan
    task = asyncio.create_task(llm.ainvoke(messages))

    # Création d'un message Chainlit en mode streaming
    msg = cl.Message(content="")
    await msg.send()  # Envoi du message initial pour préparer l'affichage

    full_response = ""  # Stocke la réponse complète

    # Lecture des tokens au fur et à mesure qu'ils arrivent
    async for chunk in callback.aiter():
        if chunk:
            full_response += chunk
            await msg.stream_token(chunk)  # Affichage dynamique des tokens

    # Attente de la fin de la requête
    await task

    # Sauvegarde dans la mémoire conversationnelle
    memory.save_context({"input": user_input}, {"output": full_response})

    return full_response

# 🔹 Initialisation de Chainlit
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=prompts["initialization"]["assistant"]).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        await cl.Message(content=prompts["goodbye"]["assistant"].format(nom="Cher utilisateur")).send()
        return
    
    await interact_with_llm(user_input)
