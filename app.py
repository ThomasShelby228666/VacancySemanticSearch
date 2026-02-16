import gradio as gr
import logging
from search import load_resources, semantic_search, get_vacancies_by_ids

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("vacancy_generator.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DB_PATH = "data/vacancies.db"
TOP_K = 5

logger.info("Запуск веб-интерфейса для поиска вакансий")

model, index, vacancy_ids = load_resources()

def search_interface(query):
    logger.info(f"Поиск вакансий по запросу: {query}")

    if not query.strip():
        return "🔍 Введите запрос!"

    try:
        found_ids = semantic_search(query, model, index, vacancy_ids, TOP_K)

        if not found_ids:
            return "❌ Ничего не найдено."

        vacancies = get_vacancies_by_ids(found_ids, DB_PATH)

        output = ""
        for i, (vid, title, desc) in enumerate(vacancies, 1):
            output += f"**{i}. {title}**\n\n{desc[:200]}...\n\n---\n\n"

        logger.info(f"Найдено {len(vacancies)} вакансий")
        return output

    except Exception as e:
        logger.error(f"Ошибка при поиске вакансий: {e}")
        return "❌ Возникла ошибка при поиске вакансий."

demo = gr.Interface(
    fn=search_interface,
    inputs=gr.Textbox(
        label="🔎 Опишите желаемую работу",
        placeholder="Например: «Хочу работать data scientist с опытом в NLP»"
    ),
    outputs=gr.Markdown(label="🎯 Результаты поиска"),
    title="Семантический поиск по вакансиям",
    description="Поиск по смыслу, а не по ключевым словам",
    examples=[
        ["ML-инженер с опытом в трансформерах"],
        ["Разработчик Python для backend-систем"],
        ["Аналитик данных с навыками визуализации"]
    ],
    submit_btn="🔍 Найти вакансии",
    clear_btn="🗑️ Очистить",
    flagging_mode="never"
)

if __name__ == "__main__":
    logger.info("Запуск веб-сервера на http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)