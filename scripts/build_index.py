import numpy as np
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("vacancy_generator.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DB_PATH = "../data/vacancies.db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_FILE = "../embeddings/embeddings.npy"
ID_MAP_FILE = "../embeddings/id_map.pkl"
FAISS_INDEX_FILE = "../index/faiss_vacancy.index"

def load_vacancies(db_path):
    logger.info(f"Чтение вакансий из: {db_path}")

    if not os.path.exists(db_path):
        logger.error(f"База данных не найдена по пути: {db_path}")
        raise FileNotFoundError

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description FROM vacancies ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

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

def generate_embeddings(texts, model):
    logger.info(f"Генерация эмбеддингов для {len(texts)} текстов с помощью модели {model}")

    try:
        model = SentenceTransformer(model)
        logger.debug(f"Модель {model} загружена успешно")

        logger.info(f"Начало генерации эмбеддингов (может длиться несколько минут)")
        embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Сгенерированы эмбеддинги с размерностью: {embeddings.shape}")

        logger.debug("Нормализация эмбеддингов для косинусного сходства")
        faiss.normalize_L2(embeddings)
        logger.debug("Нормализация завершена")

        return embeddings

    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддингов: {e}")
        raise

def save_embeddings(embeddings, ids, emb_file, id_file):
    logger.info(f"Сохранение эмбеддингов в {emb_file}")

    try:
        os.makedirs(os.path.dirname(emb_file), exist_ok=True)

        np.save(emb_file, embeddings)
        logger.debug(f"Эмбеддинги сохранены, размерность: {embeddings.shape}")

        with open(id_file, "wb") as f:
            pickle.dump(ids, f)
        logger.debug(f"Маппинг ID сохранен, их количество: {len(ids)}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        raise

def build_faiss_index(embeddings, index_file):
    logger.info("Построение FAISS индекса")

    try:
        os.makedirs(os.path.dirname(index_file), exist_ok=True)

        logger.info(f"Создание индекса размерностью {embeddings.shape[1]}")
        index = faiss.IndexFlatIP(embeddings.shape[1])

        logger.info(f"Добавление {len(embeddings)} векторов в индекс")
        index.add(embeddings.astype(np.float32))
        logger.debug(f"Индекс содержит {index.ntotal} векторов, он сохраняется в файл: {index_file}")

        faiss.write_index(index, index_file)
        logger.info("FAISS индекс сохранен успешно")

        file_size = os.path.getsize(index_file) / (1024 * 1024)  # в MB
        logger.debug(f"Размер файла индекса: {file_size:.2f} MB")

    except Exception as e:
        logger.error(f"Ошибка при построении FAISS индекса: {e}")
        raise

def main():
    logger.info("Начало построения индекса вакансий")

    try:
        logger.info("Шаг 1. Загрузка данных из БД")
        vacancy_ids, texts = load_vacancies(DB_PATH)

        if not texts:
            logger.warning("База данных пуста")
            return

        logger.info("Шаг 2. Генерация эмбеддингов")
        embeddings = generate_embeddings(texts, MODEL_NAME)

        logger.info("Шаг 3. Сохранение эмбеддингов")
        save_embeddings(
            embeddings,
            vacancy_ids,
            EMBEDDINGS_FILE,
            ID_MAP_FILE
        )

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


