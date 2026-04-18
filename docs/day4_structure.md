enterprise-rag/
├── src/
│   ├── __init__.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py              ← Day 2
│   │   └── pdf_loader.py        ← Day 2 (Day 4 改了正则)
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── recursive.py         ← Day 2
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bge.py               ← Day 3
│   ├── retrievers/
│   │   ├── __init__.py
│   │   └── qdrant_store.py      ← Day 4
│   ├── generators/
│   │   ├── __init__.py
│   │   └── llm.py               ← Day 4
│   └── pipelines/
│       ├── __init__.py
│       └── naive_rag.py         ← Day 4
├── data/
│   ├── raw/                     ← 4 份 PDF
│   └── processed/               ← Day 3 向量化结果
├── scripts/
│   ├── 01-05                    ← Day 2 实验
│   ├── 06-10                    ← Day 3 实验
│   └── 11-13                    ← Day 4 实验
├── docs/
│   ├── 01-mental-model.md       ← Day 1
│   ├── day02-summary.md         ← Day 2
│   ├── day03-summary.md         ← Day 3
│   └── day04-summary.md         ← Day 4
├── pyproject.toml
├── .env
└── .gitignore