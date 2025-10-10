from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"document": doc_txt, "question": question})

    assert docs is not None
    assert len(docs) > 0
    assert hasattr(docs[0], 'page_content')
    assert res is not None
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"document": doc_txt, "question": "how to make a sandwich"})

    assert docs is not None
    assert len(docs) > 0
    assert hasattr(docs[0], 'page_content')
    assert res is not None
    assert res.binary_score == "no"
