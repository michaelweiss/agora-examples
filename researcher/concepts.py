import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from tinydb import TinyDB, where

from agora.embedding import EmbeddingModel, SentenceTransformerModel
from concept_graph import ConceptGraph, default_min_cluster_size

DATABASE = "db/papers.json"

@st.cache_data()
def search_papers(search_terms):
    db = TinyDB(DATABASE)
    papers = [doc for doc in db.table("metadata").search(
        (where("type") == "research paper") | (where("type") == "conference paper"))]
    for keyword in search_terms.split():
        papers = [p for p in papers if keyword.lower() in p["title"].lower() or
            keyword.lower() in p["abstract"].lower()]
    paper_ids = [p["id"] for p in papers]
    abstracts = [doc for doc in db.table("abstracts").all()
        if doc["id"] in paper_ids and
        not doc["problem"].startswith("The abstract does not specify")]
    documents = [doc["problem"] for doc in abstracts]
    sources = [doc["id"] for doc in abstracts]
    db.close()

    return documents, sources

def generate_embeddings(documents):
    # Generate embeddings
    embedding_model = SentenceTransformerModel()
    embeddings = embedding_model.encode(documents)

    # Apply PCA for dimensionality reduction
    n_components = min(30, len(embeddings))
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)
    return embeddings

def main():
    st.set_page_config(page_title="Concepts", layout="wide")
    st.title("Concepts")
    
    # Initialize session state
    if 'concept_graph' not in st.session_state:
        st.session_state.concept_graph = None
    
    # Get documents
    st.sidebar.header("📄 Documents")
    search_terms = st.sidebar.text_input("Search terms (optional)", "")
    documents, sources = search_papers(search_terms)

    # Sidebar for parameters
    st.sidebar.header("⚙️ Parameters")

    n_docs = len(documents)
    default_cluster_size = default_min_cluster_size(n_docs)

    min_cluster_size = st.sidebar.slider(
        "Min Cluster Size",
        min_value=2,
        max_value=15,
        value=default_cluster_size,
        help="Minimum size of clusters"
    )

    min_samples = st.sidebar.slider(
        "Min Samples",
        min_value=1,
        max_value=5,
        value=max(1, min(5, default_cluster_size // 2)),
        help="Minimum samples in a neighborhood"
    )
    
    edge_threshold = st.sidebar.slider(
        "Edge Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Minimum similarity for creating edges"
    )
    
    # Build graph
    if st.sidebar.button("Build", type="primary"):
        if len(documents) < 2:
            st.error("Please provide at least 2 documents")
        else:
            with st.spinner("Building concept graph..."):
                # Generate embeddings
                embeddings = generate_embeddings(documents)

                # Create concept graph
                cg = ConceptGraph()
                cg.add_documents(documents, embeddings, sources)

                with st.spinner("Clustering documents..."):
                    # HDBSCAN needs min_samples <= number of documents
                    cg.cluster_documents(min_cluster_size, min(min_samples, n_docs))

                with st.spinner("Building graph..."):
                    cg.build_graph(edge_threshold=edge_threshold)
                
                st.session_state.concept_graph = cg
                st.success("✅ Concept graph built successfully!")
                     
    # Display results
    if st.session_state.concept_graph is not None:
        cg = st.session_state.concept_graph
        stats = cg.get_statistics()
        
        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Documents", stats['num_documents'])
        col2.metric("Concepts", stats['num_clusters'])
        col3.metric("Edges", stats['num_edges'])
        col4.metric("Noise", stats['num_noise'])
        col5.metric("Avg Cluster Size", f"{stats['avg_cluster_size']:.1f}")
        
        # Visualization (left) and details (right)
        col_left, col_right = st.columns([0.6, 0.4])

        # Visualization
        with col_left:
            selected = cg.visualize_graph()

        # Concept details
        with col_right:
            st.subheader("Concepts")
            selected_clusters = [int(node.replace("C", "")) for node in selected["nodes"]]
            for cluster_id in sorted(selected_clusters):
                with st.expander(f"Concept {cluster_id} ({len(cg.cluster_docs[cluster_id])} docs)"):
                    if cluster_id in cg.cluster_themes:
                        st.markdown(f"> {cg.cluster_themes[cluster_id]}")
                    elif st.button(f"Analyze theme", key=f"summarize_{cluster_id}"):
                        theme = cg.analyze_theme(cluster_id)
                        st.rerun()

                    for doc_info in cg.cluster_docs[cluster_id]:
                        st.markdown(f"- {doc_info['document']} Source: {doc_info['source']}")
        
if __name__ == "__main__":
    main()