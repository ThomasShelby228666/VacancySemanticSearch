import sqlite3
import random
import os
import logging

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
DB_PATH = "../data/vacancies.db"    # Относительный путь к базе данных
NUM_VACANCIES = 200     # Количество генерируемых вакансий
USE_RUSSIAN = True  # True - русские описания, False - английские

# Синтетическая генерация данных на русском языке
if USE_RUSSIAN:
    TITLES = [
        "Data Scientist", "ML-инженер", "Аналитик данных", "Backend-разработчик",
        "Frontend-разработчик", "DevOps-инженер", "Product Manager",
        "QA-инженер", "Data Engineer", "NLP-специалист", "Computer Vision Engineer",
        "Разработчик Python", "Разработчик Java", "Разработчик C++",
        "Системный аналитик", "BI-аналитик", "MLOps-инженер", "Research Scientist"
    ]
    DESCRIPTIONS = [
        "Компания ищет специалиста с опытом в {tech}. Основные задачи: разработка и поддержка ML-моделей, работа с большими данными.",
        "Требуется опытный {tech}-разработчик. Проект связан с масштабируемыми распределёнными системами.",
        "Ищем аналитика для работы с бизнес-метриками. Нужны навыки в {tech} и визуализации данных.",
        "Вакансия для инженера, который умеет деплоить и мониторить модели в продакшене с использованием {tech}.",
        "Проект по обработке естественного языка. Требуется опыт в {tech} и понимание современных архитектур нейросетей.",
        "Разработка высоконагруженного backend на {tech}. Опыт работы с микросервисами обязателен.",
        "Команда ищет специалиста по компьютерному зрению. Опыт с {tech} и OpenCV приветствуется.",
        "Нужен QA-инженер с автоматизацией тестов на {tech}. Знание CI/CD — плюс."
    ]
    TECH_STACK =[
        "Python, pandas, scikit-learn", "PyTorch, Transformers", "TensorFlow, Keras",
        "SQL, PostgreSQL", "Docker, Kubernetes", "Airflow, Spark",
        "React, TypeScript", "Node.js, Express", "Java, Spring",
        "C++, OpenCV", "LangChain, LLM", "FAISS, Elasticsearch",
        "Git, GitLab CI", "AWS, GCP", "Linux, Bash", "FastAPI, Flask"
    ]
# Синтетическая генерация данных на английском языке
else:
    TITLES = [
        "Data Scientist", "ML engineer", "Data Analyst", "Backend developer",
        "Frontend Developer", "DevOps Engineer", "Product Manager",
        "QA Engineer", "Data Engineer", "NLP Specialist", "Computer Vision Engineer",
        "Python Developer", "Java Developer", "C++ Developer",
        "System Analyst", "BI Analyst", "MLOps Engineer", "Research Scientist"
    ]
    DESCRIPTIONS = [
        "The company is looking for a specialist with experience in {tech}. Main tasks: development and support of ML models, working with big data.",
        "An experienced {tech} developer is required. The project is related to scalable distributed systems.",
        "We are looking for an analyst to work with business metrics. I need skills in {tech} and data visualization.",
        "A vacancy for an engineer who knows how to deploy and monitor models in production using {tech}.",
        "A natural language processing project. It requires experience in {tech} and an understanding of modern neural network architectures.",
        "Development of a high-load backend on {tech}. Experience working with microservices is required.",
        "The team is looking for a computer vision specialist. Experience with {tech} and OpenCV is welcome.",
        "We need a QA engineer with test automation for {tech}. CI/CD knowledge is a plus."
    ]
    TECH_STACK = [
        "Python, pandas, scikit-learn", "PyTorch, Transformers", "TensorFlow, Keras",
        "SQL, PostgreSQL", "Docker, Kubernetes", "Airflow, Spark",
        "React, TypeScript", "Node.js, Express", "Java, Spring",
        "C++, OpenCV", "LangChain, LLM", "FAISS, Elasticsearch",
        "Git, GitLab CI", "AWS, GCP", "Linux, Bash", "FastAPI, Flask"
    ]

def generate_vacancy(vacancy_id: int) -> tuple:
    """
    Генерирует одну вакансию со случайными данными.
    Args:
        vacancy_id (int): Уникальный идентификатор вакансии
    Returns:
        tuple: Кортеж (id, title, description)
    Example:
        >>> generate_vacancy(1)
        (1, 'ML-инженер', 'Компания ищет специалиста с опытом в PyTorch...')
    """
    title = random.choice(TITLES)
    tech = random.choice(TECH_STACK)
    description = random.choice(DESCRIPTIONS).format(tech=tech)
    return (vacancy_id, title, description)

def main() -> None:
    """
    Основная функция создания базы данных вакансий.
    Выполняет следующие шаги:
    1. Проверяет/создаёт папку data/
    2. Удаляет существующую БД (если есть)
    3. Создаёт новую БД с таблицей vacancies
    4. Генерирует указанное количество вакансий
    5. Сохраняет их в БД
    Raises:
        sqlite3.Error: При ошибках работы с SQLite
        Exception: При любых других непредвиденных ошибках
    """
    logger.info("Начало синтетической генерации базы данных вакансий")

    # Создание папки для БД
    os.makedirs("../data", exist_ok=True)

    # Удаление существующей БД
    if os.path.exists(DB_PATH):
        logger.info(f"Удаление существующей БД: {DB_PATH}")
        os.remove(DB_PATH)
        logger.debug("Файл БД успешно удален")

    try:
        # Подключение к БД
        logger.debug(f"Подключение к БД: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Создание таблицы
        logger.info("Создание таблицы vacancies")
        cursor.execute("""
            CREATE TABLE vacancies (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL)
        """)
        logger.debug("Таблица vacancies создана успешно")

        # Генерация вакансий
        logger.info(f"Генерация {NUM_VACANCIES} вакансий")
        vacancies = [generate_vacancy(i) for i in range(1, NUM_VACANCIES + 1)]

        # Вставка вакансий
        logger.debug(f"Вставка {NUM_VACANCIES} сгенерированных вакансий в таблицу")
        cursor.executemany("INSERT INTO vacancies (id, title, description) VALUES (?, ?, ?)", vacancies)

        # Сохранение изменений
        conn.commit()
        logger.info(f"Успешно добавлено {cursor.rowcount} записей в БД по пути: {DB_PATH}")

    except sqlite3.Error as e:
        logger.error(f"Ошибка sqlite: {e}")
        raise

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        raise

    finally:
        # Закрытие соединения
        if "conn" in locals():
            conn.close()
            logger.debug("Соединение с БД закрыто")

if __name__ == "__main__":
    main()




