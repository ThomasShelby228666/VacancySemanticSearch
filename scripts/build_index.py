import numpy as np
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging
from typing import Tuple, List

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
DB_PATH = "../data/vacancies.db" # Путь к базе данных
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Мультиязычная модель
EMBEDDINGS_FILE = "../embeddings/embeddings.npy" # Файл с эмбеддингами
ID_MAP_FILE = "../embeddings/id_map.pkl" # Файл с маппингом ID
FAISS_INDEX_FILE = "../index/faiss_vacancy.index" # Файл FAISS индекса

def load_vacancies(db_path: str) -> Tuple[List[str], List[str]]:
    """
    Загружает вакансии из SQLite базы данных.
    Args:
        db_path (str): Путь к файлу SQLite базы данных
    Returns:
        Tuple[List[int], List[str]]: Кортеж, содержащий:
            - Список ID вакансий
            - Список текстов для эмбеддинга (формат: "Название. Описание")
    Raises:
        FileNotFoundError: Если файл БД не существует
        sqlite3.Error: При ошибках работы с SQLite
    Example:
        >>> ids, texts = load_vacancies("../data/vacancies.db")
        >>> print(f"Загружено {len(ids)} вакансий")
    """
    logger.info(f"Чтение вакансий из: {db_path}")

    # Проверка наличия базы данных по указанному пути
    if not os.path.exists(db_path):
        logger.error(f"База данных не найдена по пути: {db_path}")
        raise FileNotFoundError

    try:
        # Подключение к БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description FROM vacancies ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

        # Загрузка вакансий
        vacancy_ids = [row[0] for row in rows]
        texts = [f"{row[1]}. {row[2]}" for row in rows]
        logger.info(f"Загружено {len(texts)} вакансий из БД")
        logger.debug(f"Первые 5 ID вакансий: {vacancy_ids[:5]}")

        return vacancy_ids, texts

    except sqlite3.Error as e:
        logger.error(f"Ошибка sqlite во время загрузки данных: {e}")
        raise

    except Exception as e:
        logger.error(f"Непредвиденная ошибка во время загрузки данных: {e}")
        raise

def generate_embeddings(texts: List[str], model: str) -> np.ndarray:
    """
    Генерирует эмбеддинги для списка текстов с помощью модели MiniLM
    Args:
        texts (List[str]): Список текстов для преобразования в эмбеддинги
        model (str): Название модели из Hugging Face
    Returns:
        np.ndarray: Массив эмбеддингов размерностью (n_texts, embedding_dim)
    Raises:
        Exception: При ошибках загрузки модели или создании эмбеддингов
    Example:
        >>> embeddings = generate_embeddings(["Python разработчик"], "paraphrase-multilingual-MiniLM-L12-v2")
        >>> print(embeddings.shape)
        (1, 384)
    """
    logger.info(f"Генерация эмбеддингов для {len(texts)} текстов с помощью модели {model}")

    try:
        # Загрузка модели
        model = SentenceTransformer(model)
        logger.debug(f"Модель {model} загружена успешно")

        # Генерация эмбеддингов
        logger.info(f"Начало генерации эмбеддингов (может длиться несколько минут)")
        embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Сгенерированы эмбеддинги с размерностью: {embeddings.shape}")

        # Нормализация эмбеддингов
        logger.debug("Нормализация эмбеддингов для косинусного сходства")
        faiss.normalize_L2(embeddings)
        logger.debug("Нормализация завершена")

        return embeddings

    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддингов: {e}")
        raise

def save_embeddings(embeddings: np.ndarrray, ids: List[int], emb_file: str, id_file: str) -> None:
    """
       Сохраняет эмбеддинги и маппинг ID в файлы.
       Args:
           embeddings (np.ndarray): Массив эмбеддингов
           ids (List[int]): Список ID вакансий в том же порядке, что и эмбеддинги
           emb_file (str): Путь для сохранения эмбеддингов (.npy)
           id_file (str): Путь для сохранения маппинга ID (.pkl)
       Raises:
           Exception: При ошибках сохранения файлов
       Example:
           >>> save_embeddings(embeddings, [1,2,3], "embeddings.npy", "id_map.pkl")
       """
    logger.info(f"Сохранение эмбеддингов в {emb_file}")

    try:
        # Создание папки для файлов
        os.makedirs(os.path.dirname(emb_file), exist_ok=True)

        # Сохранение эмбеддингов
        np.save(emb_file, embeddings)
        logger.debug(f"Эмбеддинги сохранены, размерность: {embeddings.shape}")

        # Сохранение маппинга ID
        with open(id_file, "wb") as f:
            pickle.dump(ids, f)
        logger.debug(f"Маппинг ID сохранен, их количество: {len(ids)}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        raise

def build_faiss_index(embeddings: np.ndarray, index_file: str) -> None:
    """
    Строит FAISS индекс для быстрого семантического поиска.
    Args:
        embeddings (np.ndarray): Нормализованные эмбеддинги для индексации
        index_file (str): Путь для сохранения FAISS индекса
    Raises:
        Exception: При ошибках построения или сохранения индекса
    Example:
        >>> build_faiss_index(embeddings, "faiss_vacancy.index")
    """
    logger.info("Построение FAISS индекса")

    try:
        # Создание папки для индекса
        os.makedirs(os.path.dirname(index_file), exist_ok=True)

        # Создание индекса на основе косинусного сходства
        logger.info(f"Создание индекса размерностью {embeddings.shape[1]}")
        index = faiss.IndexFlatIP(embeddings.shape[1])

        # Добавление векторов в индекс
        logger.info(f"Добавление {len(embeddings)} векторов в индекс")
        index.add(embeddings.astype(np.float32))
        logger.debug(f"Индекс содержит {index.ntotal} векторов, он сохраняется в файл: {index_file}")

        # Сохранение индекса
        faiss.write_index(index, index_file)
        logger.info("FAISS индекс сохранен успешно")

        # Логирование размера файла
        file_size = os.path.getsize(index_file) / (1024 * 1024)  # в MB
        logger.debug(f"Размер файла индекса: {file_size:.2f} MB")

    except Exception as e:
        logger.error(f"Ошибка при построении FAISS индекса: {e}")
        raise

def main() -> None:
    """
    Основная функция построения индекса вакансий.
    Выполняет следующие шаги:
    1. Загружает вакансии из SQLite БД
    2. Генерирует эмбеддинги с помощью SBERT модели
    3. Сохраняет эмбеддинги и маппинг ID
    4. Строит и сохраняет FAISS индекс
    Требует предварительного создания базы данных скриптом create_vacancy_db.py
    Raises:
        FileNotFoundError: Если база данных не найдена
        Exception: При любых других ошибках в процессе
    """
    logger.info("Начало построения индекса вакансий")

    try:
        # Загрузка данных
        logger.info("Шаг 1. Загрузка данных из БД")
        vacancy_ids, texts = load_vacancies(DB_PATH)

        if not texts:
            logger.warning("База данных пуста")
            return

        # Генерация эмбеддингов
        logger.info("Шаг 2. Генерация эмбеддингов")
        embeddings = generate_embeddings(texts, MODEL_NAME)

        # Сохранение эмбеддингов
        logger.info("Шаг 3. Сохранение эмбеддингов")
        save_embeddings(
            embeddings,
            vacancy_ids,
            EMBEDDINGS_FILE,
            ID_MAP_FILE
        )

        # Построение FAISS индекса
        logger.info("Шаг 4. Построение FAISS индекса")
        build_faiss_index(embeddings, FAISS_INDEX_FILE)

        logger.info(f"Эмбеддинги сохранены в: {EMBEDDINGS_FILE}")
        logger.info(f"Маппинг ID сохранен в: {ID_MAP_FILE}")
        logger.info(f"FAISS индекс сохранен в: {FAISS_INDEX_FILE}")

    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}. Нужно сначала запустить скрипт генерации базы данных")

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        raise

if __name__ == "__main__":
    main()


