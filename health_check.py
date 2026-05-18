"""Quick env health check — run once, delete after."""
import sys

print(f'Python: {sys.version.split()[0]}')
print('---')

packages = [
    'langgraph', 'langchain_core', 'langchain_groq', 'groq',
    'qdrant_client', 'sentence_transformers', 'pandas', 'pydantic'
]
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {version}')
    except Exception as e:
        print(f'  {pkg}: ERROR - {type(e).__name__}: {e}')

print('---')
print('Smoke test imports:')

try:
    from langgraph.graph import StateGraph, START, END
    print('  langgraph.graph: OK')
except Exception as e:
    print(f'  langgraph.graph: FAIL - {e}')

try:
    from langchain_core.messages import HumanMessage, AIMessage
    print('  langchain_core.messages: OK')
except Exception as e:
    print(f'  langchain_core.messages: FAIL - {e}')

try:
    from langchain_groq import ChatGroq
    print('  langchain_groq: OK')
except Exception as e:
    print(f'  langchain_groq: FAIL - {e}') 