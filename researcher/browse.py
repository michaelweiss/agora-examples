import streamlit as st
from tinydb import TinyDB, where
import os

DATABASE = "db/papers.json"

PAGE_SIZE = 50

SEARCH_FIELDS = ('id', 'title', 'authors', 'abstract')

def search_papers(search_terms=""):
    """Filter papers: every search word must appear in id, title, authors or abstract"""
    words = search_terms.lower().split()
    if not words:
        return []

    def matches(paper):
        text = " ".join(str(paper.get(f, "")) for f in SEARCH_FIELDS).lower()
        return all(word in text for word in words)

    db = TinyDB(DATABASE)
    try:
        return db.table("metadata").search(matches)
    finally:
        db.close()

def get_abstracts(paper_ids):
    """Fetch the abstracts for multiple papers"""
    db = TinyDB(DATABASE)
    try:
        rows = db.table("abstracts").search(where("id").one_of(set(paper_ids)))
    finally:
        db.close()
    return {row["id"]: row for row in rows}

def get_authors(paper):
    """Helper function to format authors list"""
    def guess_author(author):
        if isinstance(author, dict):
            return author.get('name', 'Unknown author')
        return str(author)
    authors = [guess_author(author) for author in paper.get('authors', [])]
    return ', '.join(authors)

def paginate(items, query):
    """Show |< < > >| controls and return the slice of items for the current page.

    The current page is kept in st.session_state and goes back to the first
    page whenever `query` changes. Call this where the controls should appear.
    """
    n_pages = max(1, -(-len(items) // PAGE_SIZE))  # ceiling division

    if st.session_state.get("page_query") != query:
        st.session_state.page_query = query
        st.session_state.page = 0
    page = min(st.session_state.get("page", 0), n_pages - 1)
    st.session_state.page = page

    def go_to(new_page):  # button callback: runs before the script reruns
        st.session_state.page = new_page

    if n_pages > 1:
        first, prev, nxt, last, info = st.columns([1, 1, 1, 1, 8])
        first.button("|<", key="page_first", help="First page",
                     on_click=go_to, args=(0,), disabled=page == 0)
        prev.button("<", key="page_prev", help="Previous page",
                    on_click=go_to, args=(page - 1,), disabled=page == 0)
        nxt.button(">", key="page_next", help="Next page",
                   on_click=go_to, args=(page + 1,), disabled=page == n_pages - 1)
        last.button(">|", key="page_last", help="Last page",
                    on_click=go_to, args=(n_pages - 1,), disabled=page == n_pages - 1)

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(items))
    if n_pages > 1:
        info.caption(f"Showing {start + 1}–{end} of {len(items)}")
    return items[start:end]

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
        st.markdown(f"**{key.title()}:** {value}")

def display_saved_papers():
    """Display saved papers section in sidebar"""
    st.sidebar.markdown("---")
    
    saved_count = len(st.session_state.saved_search_results)
    st.sidebar.caption(f"Saved papers ({saved_count}):")
    
    if saved_count > 0:
        with st.sidebar.expander("View all", expanded=False):
            for paper_id in st.session_state.saved_search_results:
                st.markdown(f"- {paper_id}")

        cols = st.sidebar.columns(2)
        with cols[0]:
            st.download_button(
                label="Export",
                data=export_saved_papers(),
                file_name="saved_papers.txt",
                mime="text/plain",
                help="Download saved paper details"
            )
        with cols[1]:
            if st.button("Clear", help="Clear saved papers"):
                # Untick the checkboxes too; otherwise a paper still shown in
                # the results keeps its ticked state and is saved again.
                for paper_id in st.session_state.saved_search_results:
                    st.session_state[f"save_{paper_id}"] = False
                st.session_state.saved_search_results.clear()
                st.rerun()
    else:
        st.sidebar.info("None")

def export_elements(paper):
    """Generate export data for the abstract elements of a paper"""
    lines = []
    for key in ['problem', 'problem owner', 'approach', 'solution', 'consequences']:
        value = paper.get(key) or 'N/A'
        lines.append(f"{key.title()}: {value}")
    return lines

def export_saved_papers():
    """Generate export data for saved papers"""
    export_lines = []
    for paper_id, paper_data in st.session_state.saved_search_results.items():
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
        st.markdown(f"Found **{len(filtered_papers)}** papers matching your search.")

        # Display papers, one page at a time
        if filtered_papers:
            query = " ".join(search_terms.lower().split())
            papers_on_page = paginate(filtered_papers, query)
            abstracts = get_abstracts(p["id"] for p in papers_on_page)
            for paper in papers_on_page:
                # Merge with abstract elements (only for papers being shown)
                abstract_data = abstracts.get(paper.get("id"))
                if abstract_data:
                    paper.update(abstract_data)
                display_paper(paper)
        else:
            if search_terms.strip():
                st.info("No papers found matching your search criteria.")
            else:
                st.info("No search criteria provided.")

if __name__ == "__main__":
    app()