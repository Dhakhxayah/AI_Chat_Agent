import os
import queue
import json
import time
from datetime import datetime

import sounddevice as sd
import vosk
import pyttsx3
import webbrowser
import psutil
import pyjokes
import requests

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# API KEYS
os.environ["GOOGLE_API_KEY"] = "Your gemini api"
SERP_API_KEY = "Your serp api"
weather_api = "Your weather api"

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Gemini + LangChain Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load Vosk Model
model_path = r"D:\\chat-agent\\vosk-model-en-us-0.22"
model = vosk.Model(model_path)
q = queue.Queue()

listening = False
chat_mode_enabled = False

def callback(indata, frames, time, status):
    if status:
        print(status)
    if listening:
        q.put(bytes(indata))

def speak(text):
    if not chat_mode_enabled:
        global listening
        listening = False
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.5)
        listening = True

def web_search(query):
    try:
        url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"
        response = requests.get(url)
        results = response.json()
        snippet = results.get("organic_results", [{}])[0].get("snippet", "No summary found.")
        return snippet
    except Exception as e:
        return f"Web search error: {e}"

def ask_gemini(question, context_chunks):
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    prompt = f"Answer this using context. If not helpful, use your own knowledge:\n\nContext:\n{context}\n\nQuestion: {question}"
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f" Gemini error: {e}"

def export_chat_log(user_input, agent_response):
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Agent: {agent_response}\n")
        f.write("-" * 40 + "\n")

def build_vector_db():
    file_path = input("\U0001F4C1 Enter path to your .txt file: ").strip()
    if not os.path.isfile(file_path):
        speak("File not found!")
        exit(1)
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    db = FAISS.from_documents(splits, embedding)
    db.save_local("rag_models/faiss_index")
    return db

vector_db = build_vector_db()

def get_tools(db):
    return [
        Tool("Answer Question", lambda q: ask_gemini(q, db.similarity_search(q, k=4)), "QA"),
        Tool("Search Web", web_search, "Live web info"),
        Tool("Summarize Text", lambda q: llm.invoke(f"Summarize this:\n{q}").content, "Summarization")
    ]

tools = get_tools(vector_db)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

def execute_command(command):
    command = command.lower().strip()

    if "open notepad" in command:
        speak("Opening Notepad")
        os.system("notepad.exe")
    elif "open google" in command:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
    elif "shutdown" in command:
        speak("Shutting down")
        os.system("shutdown /s /t 10")
    elif "restart" in command:
        speak("Restarting system")
        os.system("shutdown /r /t 10")
    elif "battery" in command:
        percent = psutil.sensors_battery().percent
        speak(f"Battery is at {percent} percent.")
    elif "time" in command:
        speak(datetime.now().strftime("It's %I:%M %p"))
    elif "date" in command:
        speak(datetime.now().strftime("Today is %A, %B %d"))
    elif "joke" in command:
        speak(pyjokes.get_joke())
    elif "weather" in command:
        city = "Coimbatore"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api}&units=metric"
        res = requests.get(url).json()
        if "main" in res:
            temp = res["main"]["temp"]
            condition = res["weather"][0]["description"]
            speak(f"{temp}°C with {condition} in {city}")
        else:
            speak("Couldn't fetch weather.")
    elif any(word in command for word in ["chapter", "document", "summary", "file"]):
        response = ask_gemini(command, vector_db.similarity_search(command, k=4))
        print("Chat Agent:\n", response)
        speak(response)
        export_chat_log(command, response)
    else:
        print("Forwarding to Chat Agent...")
        try:
            response = agent.invoke({"input": command})["output"]
            print("Chat Agent:\n", response)
            speak(response)
            export_chat_log(command, response)
        except Exception as e:
            if "429" in str(e):
                print("Agent error: Quota exceeded. Try again later.")
                speak("I'm currently out of request quota. Please try again in some time.")
            else:
                speak("Sorry, I couldn't answer that.")
                print("Agent error:", e)

def recognize_speech():
    global listening
    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16', channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, 16000)
        print("Listening...")
        listening = True

        while True:
            try:
                data = q.get(timeout=5)
            except queue.Empty:
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                command = result.get("text", "")
                if command:
                    print(f"You said: {command}")
                    execute_command(command)

def chat_mode():
    global chat_mode_enabled
    chat_mode_enabled = True
    while True:
        cmd = input(" Ask: ").strip()
        if cmd.lower() == "exit":
            speak("Exiting chat.")
            break
        execute_command(cmd)

if __name__ == "__main__":
    speak("Welcome. Would you like to chat or speak?")
    choice = input("Type 'chat' or 'speak': ").strip().lower()

    if choice == "speak":
        chat_mode_enabled = False
        recognize_speech()
    else:
        chat_mode()