from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to both local databases
db_baseline = Chroma(
    persist_directory="./chroma_db_baseline", embedding_function=embeddings
)
db_semantic = Chroma(
    persist_directory="./chroma_db_semantic", embedding_function=embeddings
)


def run_experiment(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")

    # Query Baseline
    print("--- BASELINE RETRIEVAL (Pipeline A) ---")
    baseline_results = db_baseline.similarity_search(query, k=1)
    for doc in baseline_results:
        print(f"File: {doc.metadata['source']}")
        print("Content Snippet:")
        print(
            doc.page_content[:300] + "...\n"
        )  # Print first 300 chars to see if it's broken

    # Query Semantic
    print("--- SEMANTIC RETRIEVAL (Pipeline B) ---")
    semantic_results = db_semantic.similarity_search(query, k=1)
    for doc in semantic_results:
        print(f"File: {doc.metadata['source']}")
        print("Content Snippet:")
        print(doc.page_content[:300] + "...\n")


# --- ENTER YOUR TEST QUESTIONS HERE ---
# Look at the Flask code you downloaded and ask specific questions about it.
queries = [
    "Where is the `get_send_file_max_age` method defined and how does it determine the cache value?",
    "Explain the `open_resource` method signature and parameters in the `Flask` class.",
    "How does `create_url_adapter` configure the MapAdapter when subdomain matching is disabled?",
    "What are the default configuration values initialized in `default_config` of the Flask application?",
    "How does the `send_static_file` method differ between `Flask` and `Blueprint` classes?",
    "How does the `copy_current_request_context` decorator preserve the request environment for background tasks?",
    "What is the purpose of `_AppCtxGlobals` and how does it implement property access?",
    "Explain the lifecycle and teardown process inside the `AppContext.pop()` method.",
    "How does `has_request_context()` verify if a request is currently active?",
    "In the `Request` class, how does the `max_form_memory_size` property fallback to the application configuration?",
    "Explain how the `blueprint` property determines the currently active blueprint from the endpoint name.",
    "What happens when `on_json_loading_failed` catches a `BadRequest` during JSON parsing in debug mode?",
    "How does the `Response` class configure its `max_cookie_size` attribute differently inside and outside an app context?",
    "Describe the process `find_best_app` uses to locate a Flask application instance within a module.",
    "How does the `ScriptInfo.load_app` method handle loading an application if `create_app` is not provided?",
    "What does the `with_appcontext` decorator do for Click commands?",
    "How does `load_dotenv` prioritize between a provided file path, `.env`, and `.flaskenv`?",
    "Explain how the `Config.from_prefixed_env` method parses nested dictionary structures from environment variables.",
    "How does the `ConfigAttribute` descriptor fetch and optionally convert configuration values?",
    "What is the difference between `from_pyfile` and `from_object` in terms of how they load configuration values?",
]

for q in queries:
    run_experiment(q)
