import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from st_cytoscape import cytoscape

from agora.llm import ChatgptLLMProvider

class ConceptGraph:
    """
    A concept graph that clusters document embeddings and builds a 
    knowledge graph based on inter-cluster similarity.
    """

    def __init__(self):
        """Initialize the ConceptGraph."""
        self.documents = []
        self.embeddings = []
        self.sources = []
        self.clusters = None
        self.graph = nx.Graph()
        self.cluster_docs = {}
        self.cluster_centroids = {}
        self.cluster_themes = {}
        self.local_llm = ChatgptLLMProvider()

    def add_documents(self, documents, embeddings, sources):
        """Add documents and their embeddings to the graph."""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.sources.extend(sources)

    def cluster_documents(self, min_cluster_size, min_samples):
        """Cluster documents using HDBSCAN."""
        clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
            metric='euclidean', copy=True)
        self.clusters = clustering.fit_predict(self.embeddings)
        
        # Group documents by cluster
        self.cluster_docs = {}
        for idx, cluster_id in enumerate(self.clusters):
            if cluster_id not in self.cluster_docs:
                self.cluster_docs[cluster_id] = []
            self.cluster_docs[cluster_id].append({
                'index': idx,
                'document': self.documents[idx],
                'embedding': self.embeddings[idx],
                'source': self.sources[idx]
            })
        
        return self.clusters
    
    def compute_cluster_centroids(self):
        """Compute centroid embeddings for each cluster."""
        self.cluster_centroids = {}
        
        for cluster_id, docs in self.cluster_docs.items():
            if cluster_id == -1:  # Skip noise
                continue
            embeddings = np.array([doc['embedding'] for doc in docs])
            centroid = np.mean(embeddings, axis=0)
            self.cluster_centroids[cluster_id] = centroid
            
        return self.cluster_centroids
    
    def build_graph(self, edge_threshold = 0.5):
        """Build the concept graph with nodes and edges"""
        if not self.cluster_centroids:
            self.compute_cluster_centroids()
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes for each cluster
        for cluster_id in self.cluster_centroids.keys():
            num_docs = len(self.cluster_docs[cluster_id])
            self.graph.add_node(
                cluster_id,
                label=f"Concept {cluster_id}",
                num_docs=num_docs,
                documents=[doc['document'] for doc in self.cluster_docs[cluster_id]]
            )
        
        # Add edges based on inter-cluster similarity.
        # Compute all centroid-to-centroid similarities in ONE call; calling
        # cosine_similarity() once per pair of clusters is far slower.
        cluster_ids = list(self.cluster_centroids.keys())
        if len(cluster_ids) > 1:
            centroids = np.vstack([self.cluster_centroids[c] for c in cluster_ids])
            similarity = cosine_similarity(centroids)

            # Each pair once: the upper triangle, excluding the diagonal
            rows, cols = np.triu_indices(len(cluster_ids), k=1)
            keep = similarity[rows, cols] >= edge_threshold
            for i, j in zip(rows[keep], cols[keep]):
                self.graph.add_edge(cluster_ids[i], cluster_ids[j],
                                    weight=float(similarity[i, j]))

        return self.graph
    
    def get_related_concepts(self, cluster_id):
        """Get related concepts (neighboring nodes) for a given cluster."""
        if cluster_id not in self.graph:
            raise ValueError(f"Cluster {cluster_id} not in graph")
        
        neighbors = [
            {
                'cluster_id': neighbor, 
                'weight': self.graph[cluster_id][neighbor].get('weight', None)
            }
            for neighbor in self.graph.neighbors(cluster_id)
        ]
        return neighbors
    
    def get_statistics(self):
        """Get statistics about the concept graph."""
        noise_count = len(self.cluster_docs.get(-1, []))
        num_clusters = len([c for c in self.cluster_docs.keys() if c != -1])
        
        return {
            'num_documents': len(self.documents),
            'num_clusters': num_clusters,
            'num_noise': noise_count,
            'num_edges': self.graph.number_of_edges(),
            'avg_cluster_size': np.mean([len(docs) for cid, docs in self.cluster_docs.items() if cid != -1]) if num_clusters > 0 else 0
        }

    def visualize_graph_cytoscape(self):
        """Visualize the graph using st-cytoscape."""
        stylesheet = [
            {
                "selector": "node", 
                "style": {
                    "label": "data(id)", 
                    "width": "data(size)",
                    "height": "data(size)"
                }
            },
            {
                "selector": "edge",
                "style": {
                    "width": 3,
                    "curve-style": "bezier",
                    "target-arrow-shape": None,
                },
            },
        ]

        elements = self.networkx_to_cytoscape(self.graph)

        layout = {
            "name": "fcose",  # Use a force-directed layout for cytoscape
            "fit": True,
            "padding": 30,
            "animate": True,
        }
 
        return cytoscape(elements, stylesheet, layout=layout, key="graph", 
            min_zoom=1/2, max_zoom=2, height="400px")
    
    def networkx_to_cytoscape(self, G):
        """Convert a NetworkX graph to Cytoscape elements."""
        elements = []

        for node in G.nodes():
            # Node size proportional to the number of docs relative to max cluster size
            max_docs = max([G.nodes[n]['num_docs'] for n in G.nodes()])
            proportion = G.nodes[node]['num_docs'] / max_docs if max_docs > 0 else 1
            size = 10 + 40 * proportion  # min size 10, max size 50

            elements.append(
                {
                    "data": {
                        "id": f"C{node}",
                        "documents": G.nodes[node]['documents'],
                        "size": size,
                    },
                    "selected": False,
                    "selectable": True,
                }
            )

        for edge in G.edges():
            elements.append(
                {
                    "data": {
                        "source": f"C{edge[0]}",
                        "target": f"C{edge[1]}",
                        "id": f"{edge[0]}-{edge[1]}",
                        "weight": G[edge[0]][edge[1]]['weight'],
                    },
                    "selectable": True,
                }
            )

        return elements

    def analyze_theme(self, cluster_id):
        """Use LLM to analyze the common theme of a cluster."""
        if cluster_id in self.cluster_themes:
            return self.cluster_themes[cluster_id]
        
        # Find the three closest documents to the centroid
        centroid = self.cluster_centroids[cluster_id]
        sims = []
        for doc_info in self.cluster_docs[cluster_id]:
            sim = cosine_similarity(
                [centroid],
                [doc_info['embedding']]
            )[0][0]
            sims.append((sim, doc_info))
        sims.sort(reverse=True, key=lambda x: x[0])
        n_top_docs = int(max(3, (len(sims) * 0.05) + 1))  # top 5%, but min 3
        # st.markdown(f"{n_top_docs} of {len(sims)}")
        top_docs = sims[:n_top_docs]
        docs = "\n".join(f"- {top_docs[i][1]['document']}\n" for i in range(len(top_docs)))
        prompt = (
            f"Give a concise description (~ 10 words) of the common theme or concept "
            f"that the following statements represent:\n{docs}\n"
            "Also generate a short, catchy name for this concept.\n"
            "Output: <name>: <description>"
        )
        response = self.local_llm.complete(
            [{"role": "user", "content": prompt}], temperature=0.7
        )
        self.cluster_themes[cluster_id] = response

        return response