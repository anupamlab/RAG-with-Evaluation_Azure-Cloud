import streamlit as st

from main import (
    validate_env_variables,
    load_documents,
    split_documents,
    create_embeddings,
    create_vector_store,
    create_retriever,
    create_llm,
    create_rag_chain,
    run_ragas_evaluation,
)


@st.cache_resource(show_spinner=False)
def initialize_pipeline():
    validate_env_variables()
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store, chunk_source_map = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store)
    llm = create_llm()
    qa_chain = create_rag_chain(retriever, llm, chunk_source_map=chunk_source_map)
    return embeddings, llm, qa_chain


def display_source_details(answer, sources):
    answer_text = answer.lower()
    not_in_document = any(
        phrase in answer_text
        for phrase in [
            "does not contain any information",
            "context is insufficient",
            "insufficient context",
            "not available in the context",
            "not in the context",
        ]
    )

    if not (sources or not_in_document):
        return

    st.subheader("Source Details")
    if not_in_document:
        st.markdown("- File Name: Not available in the document")
        st.markdown("  Page Number: Not available in the document")
        return

    primary_source = next(
        (
            s
            for s in sources
            if s.get("source_file") != "Unknown" or s.get("page_number") != "Unknown"
        ),
        sources[0],
    )
    st.markdown(f"- File Name: {primary_source['source_file']}")
    st.markdown(f"  Page Number: {primary_source['page_number']}")


def format_evaluation_result(result):
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    def _to_float(value):
        try:
            return float(value)
        except Exception:
            return None

    def _build_output(metric_map):
        output = {}
        for name in metric_names:
            value = metric_map.get(name)
            numeric = _to_float(value)
            output[name] = f"{numeric:.4f}" if numeric is not None else "Not available"
        return output

    if result is None:
        return None

    try:
        result_dict = dict(result)
        if result_dict and any(k in result_dict for k in metric_names):
            return _build_output(result_dict)
    except Exception:
        pass

    # Fallback for Ragas versions where per-sample metrics are exposed via result.scores.
    try:
        scores = getattr(result, "scores", None)
        if isinstance(scores, list) and scores:
            aggregated = {}
            for name in metric_names:
                vals = [_to_float(row.get(name)) for row in scores if isinstance(row, dict)]
                vals = [v for v in vals if v is not None]
                aggregated[name] = (sum(vals) / len(vals)) if vals else None
            return _build_output(aggregated)

        if hasattr(scores, "to_pandas"):
            df = scores.to_pandas()
            if df is not None and not df.empty:
                aggregated = {}
                for name in metric_names:
                    if name in df.columns:
                        aggregated[name] = _to_float(df[name].mean())
                    else:
                        aggregated[name] = None
                return _build_output(aggregated)
    except Exception:
        return None

    return None


def main():
    st.set_page_config(page_title="anupamLab RAG+Evaluation", layout="wide")
    st.title("Azure RAG with Evaluation by anupamLab")

    if "session_evaluation_rows" not in st.session_state:
        st.session_state.session_evaluation_rows = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    try:
        with st.spinner("Initializing RAG pipeline..."):
            embeddings, llm, qa_chain = initialize_pipeline()
    except Exception as exc:
        st.error(f"Failed to initialize pipeline: {exc}")
        return

    with st.form("qa_form", clear_on_submit=False):
        query = st.text_input("Ask a question")
        ask_clicked = st.form_submit_button("Submit")

    if ask_clicked and query.strip():
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"input": query.strip()})
            source_documents = response.get("source_documents", [])
            sources = response.get("sources", [])

            st.session_state.session_evaluation_rows.append(
                {
                    "user_input": query.strip(),
                    "response": response["answer"],
                    "retrieved_contexts": [doc.page_content for doc in source_documents],
                    "retrieved_sources": sources,
                }
            )

            st.session_state.last_answer = response["answer"]
            st.session_state.last_sources = sources

    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)
        display_source_details(st.session_state.last_answer, st.session_state.last_sources)

        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                result = run_ragas_evaluation(
                    st.session_state.session_evaluation_rows, llm, embeddings
                )
            formatted_result = format_evaluation_result(result)
            if formatted_result:
                st.subheader("Ragas Evaluation Results:")
                st.write(f"faithfulness: {formatted_result['faithfulness']}")
                st.write(f"answer_relevancy: {formatted_result['answer_relevancy']}")
                st.write(f"context_precision: {formatted_result['context_precision']}")
                st.write(f"context_recall: {formatted_result['context_recall']}")
                st.write(f"answer_correctness: {formatted_result['answer_correctness']}")
            elif result is not None:
                st.warning("Evaluation completed, but metric summary could not be formatted.")


if __name__ == "__main__":
    main()
