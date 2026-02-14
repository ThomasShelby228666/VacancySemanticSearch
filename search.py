import faiss
import numpy as np
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
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

DB_PATH = "data/vacancies.db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ID_MAP_FILE = "embeddings/id_map.pkl"
FAISS_INDEX_FILE = "index/faiss_vacancy.index"
TOP_K = 5

def load_resources():
    logger.info("Загрузка ресурсов для поиска")

    if not os.path.exists(DB_PATH):
        logger.error(f"База данных не найдена: {DB_PATH}. Сначала надо запустить create_vacancy_db.py")
        raise FileNotFoundError(f"БД не найдена: {DB_PATH}")

    if not os.path.exists(ID_MAP_FILE):
        logger.error(f"Маппинг ID не найден: {ID_MAP_FILE}")
        raise FileNotFoundError(f"Маппинг не найден: {ID_MAP_FILE}")

    if not os.path.exists(FAISS_INDEX_FILE):
        logger.error(f"FAISS индекс не найден: {FAISS_INDEX_FILE}. Сначала надо запустить build_index.py")
        raise FileNotFoundError(f"Индекс не найден: {FAISS_INDEX_FILE}")

    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Модель {model} загружена на устройство: {model.device}")

    index = faiss.read_index(FAISS_INDEX_FILE)
    logger.info(f"Индекс загружен, он содержит {index.ntotal} векторов")

    with open(ID_MAP_FILE, "rb") as f:
        vacancy_ids = pickle.load(f)
    logger.info(f"Маппинг загружен, содержит {len(vacancy_ids)} ID")

    if index.ntotal != len(vacancy_ids):
        logger.warning(f"Несоответствие: в индексе {index.ntotal} векторов, но в маппинге {len(vacancy_ids)} ID")

    return model, index, vacancy_ids

def semantic_search(query, model, index, vacancy_ids, k=TOP_K):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    logger.debug(f"Эмбеддинг запроса сгенирован, размерность: {query_embedding.shape}")

    distances, indices = index.search(query_embedding.astype(np.float32), k)
    logger.debug(f"Найдено {len(indices[0])} результатов")

    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        logger.debug(f"Результат {i+1}: позиция {idx}, расстояние {dist:.4f}")

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(vacancy_ids):
            vid = vacancy_ids[idx]
            results.append(vid)
            logger.info(f"ID вакансии: {vid}, позиция в индексе: {idx}")
        else:
            logger.warning(f"Индекс {idx} вне диапазона маппинга (0-{len(vacancy_ids)-1})")

    return results

def get_vacancies_by_ids(ids, db_path):
    if not ids:
        logger.warning("Нет ID для загрузки из БД")
        return []

    try:
        conn = sqlite3.connect(db_path)
        placeholders = ",".join("?" * len(ids))
        query = f"select id, title, description from vacancies where id in ({placeholders})"

        id_to_result = {}
        for row in conn.execute(query, ids).fetchall():
            id_to_result[row[0]] = row

        conn.close()

        results = [id_to_result[vid] for vid in ids if vid in id_to_result]

        if len(results) < len(ids):
            missing = set(ids) - {r[0] for r in results}
            logger.warning(f"Не найдены в БД ID: {missing}")

        logger.debug(f"Загружено {len(results)} вакансий из БД")
        return results

    except sqlite3.Error as e:
        logger.error(f"Ошибка sqlite: {e}")
        raise

def print_results(vacancies, query):

    print("=" * 50)
    print(f"Результаты поиска: '{query}'")
    print("=" * 50)

    if not vacancies:
        print("Ничего не найдено")
        logger.warning("Вакансии не найдены")
        return

    for i, (vid, title, desc) in enumerate(vacancies, 1):
        print(f"\n{i}. [{vid}] {title.upper()}")
        print("=" * 50)
        if len(desc) > 200:
            print(f"{desc[:200]}...")
        else:
            print(desc)

    print("\n" + "=" * 50)

def main():
    try:
        model, index, vacancy_ids = load_resources()

        logger.info("Система готова к поиску. Введите 'exit' для выхода.")
        print("\n" + "=" * 50)
        print("СИСТЕМА ПОИСКА ВАКАНСИЙ ГОТОВА")
        print("=" * 50)
        print("Введите текст запроса (или 'exit' для выхода):")

        while True:
            query = input("\n Введите запрос: ").strip()

            if query.lower() in ["quit", "exit", "выход"]:
                logger.info("Выход из программы")
                print("До свидания")
                break

            if not query:
                logger.info("Игнорирование пустого запроса")
                continue

            try:
                top_ids =  semantic_search(query, model, index, vacancy_ids)

                vacancies = get_vacancies_by_ids(top_ids, DB_PATH)

                print_results(vacancies, query)

            except Exception as e:
                logger.error(f"Ошибка при поиске: {e}")
                print("Возникла ошибка при поиске")

    except FileNotFoundError as e:
        logger.error(f"{e}")
        print(f"\nВозникла ошибка: {e}")
        print("\nСначала запустите:")
        print("  1. create_vacancy_db.py - создание БД с вакансиями")
        print("  2. build_index.py - построение индекса")

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        print(f"\nВозникла ошибка: {e}")

if __name__ == "__main__":
    main()