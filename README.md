# ✈️ FlightAI Chatbot

**FlightAI** is an AI-powered chatbot built with Gradio that assists users in booking flights. It supports **multilingual interactions**, **voice input/output**, and integrates **OpenAI's API** for intelligent conversations.

---

## 🌟 Features

- 🗣️ **Voice Input & Output**: Speak to the assistant and hear its response.
- 🌍 **Multilingual Support**: Choose your preferred language, and the chatbot will translate both user inputs and assistant replies.
- 💬 **Dual Chat UI**: English chat alongside translated language chat.
- 🤖 **OpenAI Integration**: Uses GPT-based responses for smart conversations.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FlightAI-Chatbot.git
cd FlightAI-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_key_here
```

Alternatively, you can export it directly in your terminal:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the chatbot

```bash
python flightai_chatbot.py
```

## 📁 File Overview

| File                 | Description                                  |
|----------------------|----------------------------------------------|
| `flightai_chatbot.py`| Main Gradio chatbot with logic and UI        |
| `requirements.txt`   | Python dependencies                          |
| `.env.example`       | Example for environment variables            |
| `README.md`          | Setup and documentation                      |
| `assets/` (optional) | Images or media used in demo or UI           |

---

## 📸 Demo

*(Optional: Add screenshots or GIFs here to showcase the chatbot UI in action)*

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🤝 Contributions

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## ✉️ Contact

For questions, suggestions, or collaborations, feel free to open a GitHub issue.
