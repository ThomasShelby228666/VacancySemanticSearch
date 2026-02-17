import faiss
import numpy as np
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
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
DB_PATH = "data/vacancies.db" # Путь к базе данных
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Мультиязычная модель
ID_MAP_FILE = "embeddings/id_map.pkl" # Файл с маппингом ID
FAISS_INDEX_FILE = "index/faiss_vacancy.index" # Файл FAISS индекса
TOP_K = 5 # Количество результатов по умолчанию

def load_resources() -> Tuple[SentenceTransformer, faiss.Index, List[int]]:
    """
    Загружает все необходимые ресурсы для поиска: модель, индекс и маппинг ID.
    Returns:
        Tuple[SentenceTransformer, faiss.Index, List[int]]: Кортеж содержащий:
            - Загруженную SBERT модель
            - FAISS индекс для поиска
            - Список ID вакансий в порядке соответствия индексу
    Raises:
        FileNotFoundError: Если любой из необходимых файлов не найден
    Example:
        >>> model, index, vacancy_ids = load_resources()
        >>> print(f"Загружено {len(vacancy_ids)} вакансий")
    """
    logger.info("Загрузка ресурсов для поиска")

    # Проверка существования всех необходимых файлов
    if not os.path.exists(DB_PATH):
        logger.error(f"База данных не найдена: {DB_PATH}. Сначала надо запустить create_vacancy_db.py")
        raise FileNotFoundError(f"БД не найдена: {DB_PATH}")

    if not os.path.exists(ID_MAP_FILE):
        logger.error(f"Маппинг ID не найден: {ID_MAP_FILE}")
        raise FileNotFoundError(f"Маппинг не найден: {ID_MAP_FILE}")

    if not os.path.exists(FAISS_INDEX_FILE):
        logger.error(f"FAISS индекс не найден: {FAISS_INDEX_FILE}. Сначала надо запустить build_index.py")
        raise FileNotFoundError(f"Индекс не найден: {FAISS_INDEX_FILE}")

    # Загрузка модели
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Модель {model} загружена на устройство: {model.device}")

    # Загрузка FAISS индекса
    index = faiss.read_index(FAISS_INDEX_FILE)
    logger.info(f"Индекс загружен, он содержит {index.ntotal} векторов")

    # Загрузка маппинга ID
    with open(ID_MAP_FILE, "rb") as f:
        vacancy_ids = pickle.load(f)
    logger.info(f"Маппинг загружен, содержит {len(vacancy_ids)} ID")

    # Проверка соответствия индекса и маппинга
    if index.ntotal != len(vacancy_ids):
        logger.warning(f"Несоответствие: в индексе {index.ntotal} векторов, но в маппинге {len(vacancy_ids)} ID")

    return model, index, vacancy_ids

def semantic_search(query: str, model: SentenceTransformer, index: faiss.index,
                    vacancy_ids: List[int], k: int = TOP_K):
    """
    Выполняет семантический поиск вакансий по текстовому запросу.
    Args:
        query (str): Текстовый запрос пользователя
        model (SentenceTransformer): Загруженная SBERT модель
        index (faiss.Index): FAISS индекс для поиска
        vacancy_ids (List[int]): Маппинг позиций индекса к ID вакансий
        k (int): Количество результатов для возврата (по умолчанию TOP_K)
    Returns:
        List[int]: Список ID найденных вакансий
    Example:
        >>> ids = semantic_search("Python разработчик", model, index, vacancy_ids)
        >>> print(ids)
        [42, 15, 73, 91, 28]
    """
    # Генерация эмбеддинга для запроса
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    logger.debug(f"Эмбеддинг запроса сгенирован, размерность: {query_embedding.shape}")

    # Поиск в FAISS индексе
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    logger.debug(f"Найдено {len(indices[0])} результатов")

    # Логирование расстояний (чем меньше, тем ближе)
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        logger.debug(f"Результат {i+1}: позиция {idx}, расстояние {dist:.4f}")

    # Преобразование позиций индекса в ID вакансий
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(vacancy_ids):
            vid = vacancy_ids[idx]
            results.append(vid)
            logger.info(f"ID вакансии: {vid}, позиция в индексе: {idx}")
        else:
            logger.warning(f"Индекс {idx} вне диапазона маппинга (0-{len(vacancy_ids)-1})")

    return results

def get_vacancies_by_ids(ids: List[int], db_path: str) -> List[Tuple]:
    """
    Получает полную информацию о вакансиях из базы данных по их ID.
    Args:
        ids (List[int]): Список ID вакансий для загрузки
        db_path (str): Путь к файлу SQLite базы данных
    Returns:
        List[Tuple]: Список кортежей (id, title, description) в порядке запрошенных ID
    Raises:
        sqlite3.Error: При ошибках работы с SQLite
    Example:
        >>> vacancies = get_vacancies_by_ids([42, 15], "data/vacancies.db")
        >>> for vid, title, desc in vacancies:
        ...     print(f"{vid}: {title}")
    """
    if not ids:
        logger.warning("Нет ID для загрузки из БД")
        return []

    try:
        # Подключение к базе данных
        conn = sqlite3.connect(db_path)
        placeholders = ",".join("?" * len(ids))
        query = f"select id, title, description from vacancies where id in ({placeholders})"

        # Создание словаря для сохранения порядка
        id_to_result = {}
        for row in conn.execute(query, ids).fetchall():
            id_to_result[row[0]] = row

        conn.close()

        # Возврат результатов в том же порядке, что и запрошенные ID
        results = [id_to_result[vid] for vid in ids if vid in id_to_result]

        if len(results) < len(ids):
            missing = set(ids) - {r[0] for r in results}
            logger.warning(f"Не найдены в БД ID: {missing}")

        logger.debug(f"Загружено {len(results)} вакансий из БД")
        return results

    except sqlite3.Error as e:
        logger.error(f"Ошибка sqlite: {e}")
        raise

def print_results(vacancies: List[Tuple], query: str) -> None:
    """
    Выводит результаты поиска в форматированном виде.
    Args:
        vacancies (List[Tuple]): Список вакансий (id, title, description)
        query (str): Исходный поисковый запрос
    Example:
        >>> print_results(vacancies, "Python разработчик")
    """
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
        # Обрезка длинного описания
        if len(desc) > 300:
            print(f"{desc[:300]}...")
        else:
            print(desc)

    print("\n" + "=" * 50)

def main():
    """
    Основная функция для интерактивного поиска вакансий.

    Загружает ресурсы и предоставляет цикл ввода запросов от пользователя.
    Поддерживает команды выхода: 'exit', 'quit', 'выход'.

    Raises:
        FileNotFoundError: Если необходимые файлы не найдены
        Exception: При любых других ошибках выполнения
    """
    try:
        # Загрузка всех ресурсов
        model, index, vacancy_ids = load_resources()

        logger.info("Система готова к поиску. Введите 'exit' для выхода.")
        print("\n" + "=" * 50)
        print("СИСТЕМА ПОИСКА ВАКАНСИЙ ГОТОВА")
        print("=" * 50)
        print("Введите текст запроса (или 'exit' для выхода):")

        # Интерактивный цикл поиска
        while True:
            query = input("\n Введите запрос: ").strip()

            # Проверка команд выхода
            if query.lower() in ["quit", "exit", "выход"]:
                logger.info("Выход из программы")
                print("До свидания")
                break

            # Пропуск пустых запросов
            if not query:
                logger.info("Игнорирование пустого запроса")
                continue

            try:
                # Выполнение поиска
                top_ids =  semantic_search(query, model, index, vacancy_ids)

                # Получение полных данных о вакансиях
                vacancies = get_vacancies_by_ids(top_ids, DB_PATH)

                # Вывод результатов
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