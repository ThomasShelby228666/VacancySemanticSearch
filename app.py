import gradio as gr
import logging
from search import load_resources, semantic_search, get_vacancies_by_ids

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("vacancy_generator.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
DB_PATH = "data/vacancies.db" # Путь к базе данных
TOP_K = 5 # Количество результатов по умолчанию

logger.info("Запуск веб-интерфейса для поиска вакансий")

# Загрузка ресурсов (модель, индекс, маппинг) при старте приложения
model, index, vacancy_ids = load_resources()

def search_interface(query: str) -> str:
    """
    Функция обработки поискового запроса для Gradio интерфейса.
    Args:
       query (str): Поисковый запрос от пользователя
    Returns:
       str: Отформатированный Markdown текст с результатами поиска
            или сообщение об ошибке/отсутствии результатов
    Логика работы:
       1. Проверка на пустой запрос
       2. Семантический поиск ID вакансий
       3. Получение полной информации о вакансиях из БД
       4. Форматирование результатов в Markdown
       5. Возврат отформатированного текста
    """
    logger.info(f"Поиск вакансий по запросу: {query}")

    # Проверка на пустой запрос
    if not query.strip():
        return "Введите запрос!"

    try:
        # Выполнение семантического поиска
        found_ids = semantic_search(query, model, index, vacancy_ids, TOP_K)

        # Проверка наличия результатов
        if not found_ids:
            return "Ничего не найдено."

        # Получение полных данных о вакансиях
        vacancies = get_vacancies_by_ids(found_ids, DB_PATH)

        # Форматирование результатов в Markdown
        output = ""
        for i, (vid, title, desc) in enumerate(vacancies, 1):
            output += f"**{i}. {title}**\n\n{desc[:300]}...\n\n---\n\n"

        logger.info(f"Найдено {len(vacancies)} вакансий")
        return output

    except Exception as e:
        logger.error(f"Ошибка при поиске вакансий: {e}")
        return "Возникла ошибка при поиске вакансий."

# Создание Gradio интерфейса
demo = gr.Interface(
    fn=search_interface,  # Функция обработки запросов
    inputs=gr.Textbox(
        label="Опишите желаемую работу",
        placeholder="Например: «Хочу работать data scientist с опытом в NLP»"
    ),
    outputs=gr.Markdown(label="Результаты поиска"),
    title="Семантический поиск по вакансиям",
    description="Поиск по смыслу, а не по ключевым словам",
    examples=[
        ["ML-инженер с опытом в трансформерах"],
        ["Разработчик Python для backend-систем"],
        ["Аналитик данных с навыками визуализации"]
    ],
    submit_btn="Найти вакансии", # Кастомная кнопка отправки
    clear_btn="🗑Очистить", # Кастомная кнопка очистки
    flagging_mode="never" # Отключение кнопки Flag (если включить - появится возможность скачивать в csv выводимый список вакансий)
)

if __name__ == "__main__":
    """
    Запуск веб-сервера приложения.
    Параметры запуска:
       server_name="0.0.0.0" - доступ с любого устройства в сети
       server_port=7860 - порт для доступа к приложению
    После запуска приложение доступно по адресу:
    - Локально: http://localhost:7860
    """
    logger.info("Запуск веб-сервера на http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)