import gradio as gr
import time
from langfuse import Langfuse
from langfuse.decorators import observe
from smolagents import OpenAIServerModel, ToolCallingAgent
import git
import os

langfuse = Langfuse(
    public_key="pk-lf-1e88a751-e94e-411a-8df4-6f351be3b82b",  
    secret_key="sk-lf-7fd9c71f-d0dc-4ae6-87a9-a6603a1afe1b",  
    host="https://cloud.langfuse.com"  
)

model = OpenAIServerModel(
    model_id="local-model",
    api_base="http://localhost:1234/v1",
    api_key="not-needed",
)

agent = ToolCallingAgent(
    name="LocalLLMAgent",
    model=model,
    tools=[],
)

messages = []

def process_mcp_context(context_data):
    if not context_data:
        return ""
    if "context" in context_data and isinstance(context_data["context"], dict):
        context_str = "Контекст (MCP):\n"
        for key, value in context_data["context"].items():
            context_str += f"- {key}: {value}\n"
        return context_str
    else:
        print("Предоставленный контекст не соответствует формату MCP.")
        return ""

def get_files_and_folders(repo_path, base_path="", level=0):
    files_and_folders = []
    for item in os.listdir(os.path.join(repo_path, base_path)):
        if item == ".git":
            continue
        full_path = os.path.join(base_path, item)
        indent = "  " * level
        if os.path.isfile(os.path.join(repo_path, full_path)):
            files_and_folders.append(f"{indent}📄 {full_path}")
        elif os.path.isdir(os.path.join(repo_path, full_path)):
            files_and_folders.append(f"{indent}📁 {full_path}")
            files_and_folders.extend(get_files_and_folders(repo_path, full_path, level + 1))
    return files_and_folders

def get_files_content(repo_path, base_path=""):
    files_content = []
    for item in os.listdir(os.path.join(repo_path, base_path)):
        if item == ".git":
            continue
        full_path = os.path.join(base_path, item)
        file_path = os.path.join(repo_path, full_path)
        if os.path.isfile(file_path):
            if is_text_file(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                    files_content.append({
                        "path": full_path,
                        "content": content
                    })
            else:
                print(f"Пропускаем медиафайл: {full_path}")
        elif os.path.isdir(file_path):
            files_content.extend(get_files_content(repo_path, full_path))
    return files_content

def is_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file.read(1024)
        return True
    except UnicodeDecodeError:
        return False

def clone_and_analyze_repo(repo_url):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    local_path = f"./{repo_name}"
    if os.path.exists(local_path):
        print("Репозиторий уже существует, пропускаем клонирование.")
    else:
        print("Клонирование репозитория...")
        try:
            repo = git.Repo.clone_from(repo_url, local_path)
            print("Репозиторий успешно клонирован!")
        except git.exc.GitCommandError as e:
            return f"Ошибка при клонировании репозитория: {e}"
    files_and_folders = get_files_and_folders(local_path)
    files_content = get_files_content(local_path)
    report = "СПИСОК ФАЙЛОВ И ПАПОК:\n"
    report += "\n".join(files_and_folders)
    report += "\nСОДЕРЖИМОЕ ФАЙЛОВ:\n"
    for file in files_content:
        report += f"\n{'='*50}\nФайл: {file['path']}\n{'='*50}\n"
        report += file['content']
        report += f"\n{'-'*50}\n"
    return report

@observe
def generate_response(github_link, user_input, system_prompt=None, temperature=0.7, max_tokens=512, top_p=0.9, mcp_context=None):
    global messages
    mcp_context_str = process_mcp_context(mcp_context)
    if mcp_context_str:
        user_input = f"{mcp_context_str}\n{user_input}"
    if github_link.strip():
        try:
            repo_data = clone_and_analyze_repo(github_link)
            user_input = f"Ссылка на GitHub: {github_link}\n{repo_data}\n{user_input}"
        except Exception as e:
            return f"Ошибка при обработке репозитория: {str(e)}"
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + [
            msg for msg in messages if msg["role"] != "system"
        ]
    messages.append({"role": "user", "content": user_input})
    start_time = time.time()
    try:
        response = agent.run(user_input)
    except Exception as e:
        return f"Ошибка при генерации ответа: {str(e)}"
    generation_time = time.time() - start_time
    if hasattr(response, 'text'):
        assistant_reply = response.text
    elif hasattr(response, 'content'):
        assistant_reply = response.content
    elif isinstance(response, dict):
        assistant_reply = response.get("answer", "Ошибка: ответ модели пуст.")
    else:
        assistant_reply = str(response)
    messages.append({"role": "assistant", "content": assistant_reply})
    trace = langfuse.trace(name="smolagents-example")
    trace.generation(
        name="chat-completion",
        input=user_input,
        output=assistant_reply,
        metadata={
            "generation_time": generation_time,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "messages": messages
        }
    )
    return f"Модель ({generation_time:.2f} сек): {assistant_reply}"

def new_chat():
    global messages
    messages = []
    return None

with gr.Blocks(css="""
    body { background-color: #1E1E1E; color: #5E5E61; }
    .sidebar { background-color: #2E2D31; color: white; padding: 35px; border-radius: 4px; }
    .nav-button {
        background-color: transparent;
        color: white;
        border: none;
        padding: 5px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .nav-button img { transition: transform 0.3s ease; }
    .nav-button:hover { background-color: #4a494e; }
    .nav-button:hover img { transform: scale(1.1); }
    .nav-button:hover span { color: white; }
""") as demo:
    with gr.Row():
        with gr.Sidebar(elem_classes="sidebar"):
            gr.HTML("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <svg width="43" height="43" viewBox="0 0 43 43" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <g filter="url(#filter0_d_8_98)">
                            <path d="M19.979 12.1198L21.4276 17.0453H9.87437L11.3593 12.1198H19.979Z" fill="white"/>
                            <path d="M12.5182 17.9507L17.4799 19.1459L11.7214 29.1418L8.20833 25.4114L12.5182 17.9507Z" fill="white"/>
                            <path d="M13.8219 27.3309L17.335 23.6006L23.1297 33.5965L18.1318 34.7917L13.8219 27.3309Z" fill="white"/>
                            <path d="M31.2062 30.844H22.5866L21.1379 25.9547H32.6912" fill="white"/>
                            <path d="M34.357 17.5524L30.0472 25.0493L25.0855 23.8541L30.844 13.8582" fill="white"/>
                            <path d="M28.7434 15.6691L25.2303 19.3994L19.4356 9.36727L24.4335 8.20833L28.7434 15.6691Z" fill="white"/>
                        </g>
                        <defs>
                            <filter id="filter0_d_8_98" x="-1.1" y="-1.1" width="45.2" height="45.2" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
                                <feFlood flood-opacity="0" result="BackgroundImageFix"/>
                                <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
                                <feOffset/>
                                <feGaussianBlur stdDeviation="4.05"/>
                                <feComposite in2="hardAlpha" operator="out"/>
                                <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0"/>
                                <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_8_98"/>
                                <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow_8_98" result="shape"/>
                            </filter>
                        </defs>
                    </svg>
                    <div style="display: flex; flex-direction: column;">
                        <h1 style="margin: 0; font-size: 30px;">SPECIAL</h1>
                        <div style="display: flex; gap: 2px;">
                            <h1 style="margin: 0; font-size: 30px;">GIT</h1>
                            <h1 style="margin: 0; font-size: 30px; color: #ffffff54;">GPT</h1>
                        </div>
                    </div>
                </div>
            """)
            with gr.Column():
                new_chat_button = gr.Button(
                    "Новый чат",
                    elem_classes="nav-button",
                    variant="secondary"
                )
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Диалог с моделью", type="messages")
            github_link = gr.Textbox(
                label="Введите ссылку на GitHub/GitLab",
                placeholder="https://github.com/example",
                elem_classes="input-field"
            )
            user_input = gr.Textbox(
                label="Введите ваше сообщение",
                placeholder="Введите текст...",
                elem_classes="input-field"
            )
            with gr.Accordion("Настройка гиперпараметров", open=False):
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                    label="Температура (Temperature)",
                    info="Контролирует случайность генерации (ниже - более детерминировано)"
                )
                max_tokens = gr.Slider(
                    minimum=16, maximum=40000, value=1024, step=16,
                    label="Максимальное количество токенов (Max Tokens)",
                    info="Максимальная длина ответа модели"
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.1,
                    label="Top-P Sampling",
                    info="Фильтрует токены, которые покрывают заданную вероятность"
                )
            submit_button = gr.Button("Отправить", elem_classes="submit-button")
    
    def update_chat(github_link, user_message, temperature, max_tokens, top_p, chat_history):
        global messages
        response = generate_response(
            github_link=github_link,
            user_input=user_message,
            system_prompt="Ты — эксперт по анализу Git-репозиториев. Отвечай только на РУССКОМ ЯЗЫКЕ!!! "
                          "Проанализируй предоставленные данные репозитория, но основывайся ТОЛЬКО на содержимом файлов (код, конфигурации, документация). "
                          "Не используй метаданные репозитория (например, историю коммитов или авторов). Предоставь обзор в следующем формате: "
                          "1. **Структура проекта**: "
                          "- Опиши основные директории и файлы в репозитории. "
                          "- Укажи, какие типы файлов присутствуют (например, исходный код, тесты, конфигурации, документация). "
                          "2. **Анализ кода и файлов**: "
                          "- Выдели ключевые особенности кода (например, используемые языки программирования, фреймворки, библиотеки). "
                          "- Опиши основные функции или модули, если они выделяются в файлах. "
                          "- Если есть конфигурационные файлы (например, `.env`, `Dockerfile`, `package.json`), укажи их назначение. "
                          "3. **Потенциальные проблемы и уязвимости**: "
                          "- Выяви возможные проблемы в коде (например, устаревшие библиотеки, отсутствие обработки ошибок). "
                          "- Укажи потенциальные уязвимости (например, жестко закодированные пароли, небезопасные методы). "
                          "- Отметь наличие или отсутствие важных файлов (например, `.gitignore`, `README.md`, тестов). "
                          "4. **Рекомендации**: "
                          "- Предложи конкретные улучшения для кода или структуры файлов. "
                          "- Укажи, какие файлы или части кода требуют особого внимания. ",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})
        return "", "", temperature, max_tokens, top_p, chat_history
    
    new_chat_button.click(
        new_chat,
        inputs=[],
        outputs=[chatbot]
    )

    submit_button.click(
        update_chat,
        inputs=[github_link, user_input, temperature, max_tokens, top_p, chatbot],
        outputs=[github_link, user_input, temperature, max_tokens, top_p, chatbot]
    )

demo.launch(share=True)
