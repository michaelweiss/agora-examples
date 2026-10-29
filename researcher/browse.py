import streamlit as st
from tinydb import TinyDB, where
import os

DATABASE = "db/papers.json"

def search_papers(search_terms=""):
    """Filter papers using TinyDB API by search term"""
    # Split search terms by whitespace and apply AND logic
    search_words = search_terms.split()

    # If no search terms, don't return any papers
    if not search_words:
        return []

    # Build search conditions: one condition per word
    conditions = []
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

    # Combine all words with AND logic
    final_condition = conditions[0]
    for cond in conditions[1:]:
        final_condition = final_condition & cond

    # Search the metadata table
    db = TinyDB(DATABASE)
    try:
        table = db.table("metadata")
        return table.search(final_condition)
    finally:
        db.close()

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
            
            display_elements(paper)

def display_save_status_and_title(paper):
    """Display the save status checkbox and title for a paper"""
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

def export_elements(paper):
    """Generate export data for the abstract elements of a paper"""
    lines = []
    for key in ['problem', 'problem owner', 'approach', 'solution', 'consequences']:
        value = paper.get(key, 'N/A')
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return lines

def export_saved_papers():
    """Generate export data for saved papers"""
    if not st.session_state.saved_search_results:
        return None
    
    # Create a simple text export with paper titles
    export_lines = []
    for i, (paper_id, paper_data) in enumerate(st.session_state.saved_search_results.items(), 1):
        export_lines.append(f"# {paper_id}")
        export_lines.extend(export_elements(paper_data))
        export_lines.append("")
    
    return "\n".join(export_lines)

def app():
    st.title("Papers")
    
    # Initialize session state for saved search results
    if 'saved_search_results' not in st.session_state:
        st.session_state.saved_search_results = {}
    
    # Search interface
    search_terms = st.sidebar.text_input("Search papers:", placeholder="Enter search terms...")

    # Display saved papers in sidebar
    display_saved_papers()

    with st.spinner("Searching papers..."):
        # Filter papers
        filtered_papers = search_papers(search_terms)

        # Merge with abstract elements
        for paper in filtered_papers:
            abstract_data = get_abstract(paper.get("id"))
            if abstract_data:
                paper.update(abstract_data)
        
        # Display papers
        if filtered_papers:
            for paper in filtered_papers:
                display_paper(paper)
        else:
            if search_terms.strip():
                st.info("No papers found matching your search criteria.")
            else:
                st.info("No search criteria provided.")

if __name__ == "__main__":
    app()