import json
import threading
import time
import logging
import os
from pypdf import PdfReader
import re
import sys
from tinydb import TinyDB, where

from agora.agent import Agent, Blackboard, Controller
from agora.llm import LLMProvider, ClaudeLLMProvider, ChatgptLLMProvider, ResultParser

# PAPER_DIR = "./papers/"
PAPER_DIR = "/Users/mrw/Downloads/"
DATABASE = "db/papers.json"

class MetadataExtractor(Agent):
    def can_execute(self, blackboard):
        paper = blackboard.get("paper")
        return paper is not None and blackboard.get("metadata") is None

    def execute(self, blackboard, user_input=None):
        paper = blackboard.get("paper")
        logging.info(f"{self.name} is extracting metadata from the paper.")
        try:
            reader = PdfReader(PAPER_DIR + paper)
            text = ""
            for page in reader.pages[:3]:  # Read first 3 pages for the abstract
                text += page.extract_text() + "\n"
            prompt = f"""
                Extract the title, authors, abstract, and keywords from the following 
                text. If there is no abstract, summarize the text into a concise abstract 
                of no more than 150 words. If the keywords are missing, choose 3 
                descriptive keywords. Also guess the document type (e.g., research 
                paper, book, thesis, news article). Output the metadata in json format 
                with 'title', 'authors', 'abstract', 'keywords', and 'type' fields.
                Text: {text}
            """
            response = self.llm_provider.complete(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
            parser = ResultParser()
            metadata = parser.parse(response)["json"][0]
            metadata["id"] = paper  # Use filename as ID
            metadata["keywords"] = [keyword.lower() for keyword in metadata.get("keywords", [])]
            metadata["num_pages"] = len(reader.pages)
            metadata["doi"] = reader.metadata.get("/doi", "") if reader.metadata else ""
            blackboard.update("metadata", metadata)
        except Exception as e:
            logging.error(f"Error processing {paper}: {e}")

class AbstractAnalyzer(Agent):
    def can_execute(self, blackboard):
        metadata = blackboard.get("metadata")
        return metadata is not None and blackboard.get("abstract-elements") is None

    def execute(self, blackboard, user_input=None):
        metadata = blackboard.get("metadata")
        logging.info(f"{self.name} is analyzing the abstract.")
        prompt = f"""
            Given the abstract of a research paper, identify concise answers to these questions:
            1. Problem: What is the specific gap, limitation, challenge, or unresolved issue in the current state of knowledge or practice that the paper addresses?
            2. Problem owner: Which stakeholders, communities, industries, or fields will benefit from solving this problem, and why does it matter to them?
            3. Approach: What methodology, framework, technique, or research design did the paper employ to investigate the problem and develop its solution?
            4. Solution: What are the key findings, contributions, innovations, or proposed interventions that emerged from the paper?
            5. Consequences: What are the expected outcomes, implications, or impacts of applying the paper's solution, both for the immediate problem and for the broader field?
            Output in json format with 'problem', 'problem owner', 'approach', 'solution' and 'consequences' fields.
            Abstract: {metadata['abstract']}
        """
        response = self.llm_provider.complete(
            [{"role": "user", "content": prompt}], temperature=0.0
        )
        parser = ResultParser()
        abstract_elements = parser.parse(response)["json"][0]
        abstract_elements["id"] = metadata["id"]  # Use paper ID as key
        blackboard.update("abstract-elements", abstract_elements)

def extract_and_analyze(paper):
    # Spin up an agent chain to process the paper and analyze its abstract
    blackboard = Blackboard({
        "paper": None,
        "metadata": None,
        "abstract-elements": None,
    })
    blackboard.update("paper", paper)
    controller = Controller(blackboard, "abstract-elements")
    controller.add_agent(MetadataExtractor("MetadataExtractor", ChatgptLLMProvider(
        system="You are an expert at reading research papers.")))
    controller.add_agent(AbstractAnalyzer("AbstractAnalyzer", ChatgptLLMProvider(
        system="You are an expert at analyzing research paper abstracts.")))
    controller.run(max_iterations=3)
    return blackboard.get("metadata"), blackboard.get("abstract-elements")

def read_year_month():
    if len(sys.argv) > 2:
        try:
            year = int(sys.argv[1])
            month = int(sys.argv[2])
            return year, month
        except ValueError:
            print("Usage: python researcher.py <year> <month>")
            return None, None
    else:
        print("Usage: python researcher.py <year> <month>")
        return None, None

def list_papers_for_date(year, month):
    paper_list = []
    for paper in os.listdir(PAPER_DIR):
        if paper.endswith(".pdf"):
            stats = os.stat(PAPER_DIR + paper)
            mtime = time.localtime(stats.st_mtime)
            if mtime.tm_year == year and mtime.tm_mon == month:
                paper_list.append(paper)
    return paper_list

def main():
    logging.basicConfig(level=logging.ERROR)
    year, month = read_year_month()
    if year is None or month is None:
        return
    paper_list = list_papers_for_date(year, month)
    paper_list.sort()  # makes it easier to track progress, nothing else
    logging.info(f"Found {len(paper_list)} paper(s) in {PAPER_DIR}")

    db = TinyDB(DATABASE)
    num_papers = len(paper_list)
    for i, paper in enumerate(paper_list):
        # skip if we already have metadata for this paper
        metadata_found = db.table("metadata").search(where("id") == paper)
        abstract_found = db.table("abstracts").search(where("id") == paper)
        if not metadata_found or not abstract_found:
            # also reject papers with names starting with a number (ie *PLoP)
            if paper[0].isdigit() or paper[0] == '_':
                continue
            # check if paper name ends with a year
            if not re.search(r"\d{4}\.pdf$", paper):
                continue
            print(f"Processing paper {i+1}/{num_papers}: {paper}")
            metadata, abstract_elements = extract_and_analyze(paper)
            print(f"Adding metadata and abstract elements for {paper}")
            if not metadata_found and metadata:
                db.table("metadata").insert(metadata)
            if not abstract_found and abstract_elements:
                print(f"Abstract elements for {paper}:\n{json.dumps(abstract_elements, indent=2)}")
                db.table("abstracts").insert(abstract_elements)
    db.close()

if __name__ == "__main__":
    main()
