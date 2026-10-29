import streamlit as st
from tinydb import TinyDB, where
import re
import json
import os

DATABASE = "db/papers.json"

def get_all_keywords(papers):
    """Extract all unique keywords from papers"""
    keywords = set()
    for paper in papers:
        for keyword in paper.get('keywords', []):
            keywords.add(keyword.lower())
    return sorted(list(keywords))

def search_papers(_db, search_terms="", selected_keywords=None):
    """Filter papers using TinyDB API by search term and keywords"""
    if selected_keywords is None:
        selected_keywords = []
    
    table = _db.table("metadata")

    # If no search criteria, don't return any papers
    if not search_terms and not selected_keywords:
        return []

    # Build search conditions
    conditions = []
    
    # Text search across multiple fields using TinyDB where conditions
    if search_terms:
        # Split search terms by whitespace and apply AND logic
        search_words = search_terms.strip().split()
        if search_words:
            # For each word, create conditions to check all fields
            for word in search_words:
                word_lower = word.lower()
                word_conditions = [
                    where('id').test(lambda x, w=word_lower: w in str(x).lower()),
                    where('title').test(lambda x, w=word_lower: w in str(x).lower()),
                    where('authors').test(lambda x, w=word_lower: w in str(x).lower()),
                    where('abstract').test(lambda x, w=word_lower: w in str(x).lower()),
                ]
                # Combine with OR logic for this word (word can be found in any field)
                word_condition = word_conditions[0]
                for cond in word_conditions[1:]:
                    word_condition = word_condition | cond
                conditions.append(word_condition)
    
    # Keyword filtering using TinyDB with OR logic
    if selected_keywords:
        # Any of the selected keywords must be present (OR logic)
        keyword_conditions = []
        for keyword in selected_keywords:
            keyword_conditions.append(
                where('keywords').test(lambda x, kw=keyword: kw.lower() in [k.lower() for k in x])
            )
        # Combine keyword conditions with OR logic
        if keyword_conditions:
            keyword_condition = keyword_conditions[0]
            for cond in keyword_conditions[1:]:
                keyword_condition = keyword_condition | cond
            conditions.append(keyword_condition)
    
    # Combine all conditions with AND logic
    if len(conditions) == 1:
        final_condition = conditions[0]
    else:
        final_condition = conditions[0]
        for cond in conditions[1:]:
            final_condition = final_condition & cond
    
    return table.search(final_condition)

@st.cache_resource()
def load_abstracts(path, mtime):
    """Read the abstracts table ONCE and index it by paper id"""
    db = TinyDB(path)
    try:
        return { row.get("id"): row for row in db.table("abstracts").all() }
    finally:
        db.close()

def get_abstract(paper_id):
    """Retrieve abstract elements for a given paper"""
    # TinyDB re-parses the whole JSON file on every query, so looking up
    # papers one at a time is very slow. So instead of this:
    #
    # abstracts_table = _db.table("abstracts")
    # return abstracts_table.get(where("id") == paper_id)
    #
    # we load all abstracts once and then look up the paper by its id.

    abstracts = load_abstracts(DATABASE, os.path.getmtime(DATABASE))
    return abstracts.get(paper_id)

def get_authors(paper):
    """Helper function to format authors list"""
    def guess_author(author):
        if isinstance(author, dict):
            return author.get('name', 'Unknown author')
        return str(author)
    authors = [guess_author(author) for author in paper.get('authors', [])]
    return ', '.join(authors)

def display_paper(paper):
    """Display a single paper's metadata"""
    with st.container():
        display_save_status_and_title(paper)
        with st.expander("More ..."):
            if paper.get('authors'):
                st.markdown(f"*Authors:* {get_authors(paper)}")

            if paper.get('abstract'):
                st.markdown(f"*Abstract:* {paper.get('abstract')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if paper.get('num_pages'):
                    st.markdown(f"*Pages:* {paper.get('num_pages')}")
            with col2:
                if paper.get('doi'):
                    st.markdown(f"*DOI:* {paper.get('doi')}")
            with col3:
                st.markdown(f"*ID:* {paper.get('id', 'N/A')}")
            
            if paper.get('keywords'):
                st.markdown("*Keywords:* " + " • ".join(paper.get('keywords')))

            display_elements(paper)

def display_save_status_and_title(paper):
    paper_id = paper.get('id')
    paper_title = f"{paper.get('title', 'Untitled')}"
    is_saved = paper_id in st.session_state.saved_search_results
    save_paper = st.checkbox(paper_title, 
        value=is_saved, key=f"save_{paper_id}")
    if save_paper and not is_saved:
        st.session_state.saved_search_results[paper_id] = {
            'id': paper_id,
            'title': paper.get('title', 'Untitled'),
            'authors': paper.get('authors', []),
            'doi': paper.get('doi', ''),
            'abstract': paper.get('abstract', ''),
            'keywords': paper.get('keywords', []),
            'problem': paper.get('problem', ''),
            'problem owner': paper.get('problem owner', ''),
            'approach': paper.get('approach', ''),
            'solution': paper.get('solution', ''),
            'consequences': paper.get('consequences', '')
        }
        st.rerun()
    elif not save_paper and is_saved:
        del st.session_state.saved_search_results[paper_id]
        st.rerun()

def display_elements(paper):
    """Display the abstract elements of a paper if available"""
    for key in ['problem', 'problem owner', 'approach', 'solution', 'consequences']:
        value = paper.get(key)
        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

def display_saved_papers():
    """Display saved papers section in sidebar"""
    st.sidebar.markdown("---")
    
    saved_count = len(st.session_state.saved_search_results)
    st.sidebar.caption(f"Saved papers ({saved_count}):")
    
    if saved_count > 0:
        with st.sidebar.expander("View all", expanded=False):
            for paper_id, paper_data in st.session_state.saved_search_results.items():
                title = paper_data.get('title', 'Untitled')
                # Truncate long titles for display
                # display_title = title[:50] + "..." if len(title) > 50 else title
                st.markdown(f"- {paper_id}")
        
        cols = st.sidebar.columns(2)
        with cols[0]:
            export_data = export_saved_papers()
            if export_data:
                st.download_button(
                    label="Export",
                    data=export_data,
                    file_name="saved_papers.txt",
                    mime="text/plain",
                    help="Download saved paper details"
                )
        with cols[1]:
            if st.button("Clear", help="Clear saved papers"):
                st.session_state.saved_search_results.clear()
                st.rerun()
    else:
        st.sidebar.info("None")

def export_saved_papers():
    """Generate export data for saved papers"""
    if not st.session_state.saved_search_results:
        return None
    
    # Create a simple text export with paper titles
    export_lines = []
    for i, (paper_id, paper_data) in enumerate(st.session_state.saved_search_results.items(), 1):
        title = paper_data.get('title', 'Untitled')
        authors = paper_data.get('authors', [])
        authors_str = get_authors(paper_data) if authors else 'Unknown authors'
        problem = paper_data.get('problem', 'N/A')
        problem_owner = paper_data.get('problem owner', 'N/A')
        approach = paper_data.get('approach', 'N/A')
        solution = paper_data.get('solution', 'N/A')
        consequences = paper_data.get('consequences', 'N/A')
        export_lines.append(f"# {paper_id}")
        export_lines.append(f"Title: {title}")
        # export_lines.append(f"   Authors: {authors_str}")
        export_lines.append(f"Problem: {problem}")
        export_lines.append(f"Problem Owner: {problem_owner}")
        export_lines.append(f"Approach: {approach}")
        export_lines.append(f"Solution: {solution}")
        export_lines.append(f"Consequences: {consequences}")
        export_lines.append("")
    
    return "\n".join(export_lines)

def app():
    st.title("Papers")
    
    # Initialize session state for saved search results
    if 'saved_search_results' not in st.session_state:
        st.session_state.saved_search_results = {}
    
    # Load data
    db = TinyDB(DATABASE)
    all_papers = db.table("metadata").all()
    all_keywords = get_all_keywords(all_papers)
    
    # Search interface
    search_terms = st.sidebar.text_input("Search papers:", placeholder="Enter search terms...")
    
    # Keyword selection
    selected_keywords = st.sidebar.multiselect(
        "Filter by keywords:",
        options=all_keywords,
        placeholder="Select keywords to filter by..."
    )
    
    # Display saved papers in sidebar
    display_saved_papers()
    
    # Filter papers
    filtered_papers = search_papers(db, search_terms, selected_keywords)

    # Merge with abstract elements
    for paper in filtered_papers:
        abstract_data = get_abstract(paper.get("id"))
        if abstract_data:
            paper.update(abstract_data)
    
    # Results summary
    st.markdown(f"**Found {len(filtered_papers)} papers** (out of {len(all_papers)} total)")
    
    # Display papers
    if filtered_papers:
        for paper in filtered_papers:
            display_paper(paper)
    else:
        if search_terms or selected_keywords:
            st.info("No papers found matching your search criteria.")
        elif not search_terms and not selected_keywords:
            st.info("No search criteria provided.")
        else:
            st.info("No papers in database.")

if __name__ == "__main__":
    app()