from langchain_community.llms.openai import OpenAIChat
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
import dotenv
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from openai import models

from ldrag.gptconnector import logger

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class HybridGraphRAG:
    """
    Hybrid RAG system combining Cypher query generation with vector similarity search
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._setup_graph()

    def _setup_graph(self):
        """Initialize Neo4j graph connection"""
        from langchain_neo4j import Neo4jGraph
        from langchain_openai import ChatOpenAI

        self.graph = Neo4jGraph(url=self.uri, username=self.user, password=self.password)
        self.graph.refresh_schema()

        # LLMs for different purposes
        self.cypher_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.qa_llm = ChatOpenAI(model="o4-mini-2025-04-16", temperature=1)
        self.routing_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def _classify_query_type(self, user_query: str) -> str:
        """
        Classify whether the query is better suited for Cypher or similarity search
        """
        classification_prompt = f"""
        Analyze the following user query and classify it as either:
        1. "cypher" - if it asks for specific relationships, structured data, counts, aggregations, or graph traversals
        2. "similarity" - if it asks for conceptual matches, semantic similarity, or content-based search
        3. "hybrid" - if it could benefit from both approaches

        Query: "{user_query}"

        Examples:
        - "Which models have the best SHAP values?" → cypher (specific property query)
        - "Find models similar to ResNet" → similarity (semantic search)
        - "What are the relationships between model X and Y?" → cypher (graph traversal)
        - "Find papers about deep learning optimization" → similarity (content search)
        - "Show me models with accuracy > 0.9 that are similar to transformer architectures" → hybrid

        Return only one word: cypher, similarity, or hybrid
        """


        return self.routing_llm.invoke(classification_prompt).content.strip().lower()

    def _cypher_search(self, user_query: str) -> Dict[str, Any]:
        """
        Use Cypher query generation for structured queries
        """
        try:
            from langchain_neo4j import GraphCypherQAChain

            chain = GraphCypherQAChain.from_llm(
                cypher_llm=self.cypher_llm,
                qa_llm=self.qa_llm,
                graph=self.graph,
                top_k=10,
                allow_dangerous_requests=True,
                verbose=True,
                return_direct=False
            )

            result = chain.invoke(user_query)
            return {
                "type": "cypher",
                "result": result["result"],
                "query": getattr(result, 'query', ''),
                "confidence": 0.8  # High confidence for structured queries
            }
        except Exception as e:
            print(f"Cypher search failed: {e}")
            return {"type": "cypher", "result": "", "error": str(e), "confidence": 0.0}

    def _similarity_search(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Use vector similarity search with proper error handling and fallback
        """
        try:
            # First try to use Neo4jVector with the correct index name
            vector_store = Neo4jVector(
                url=self.uri,
                username=self.user,
                password=self.password,
                embedding=self.embeddings,
                index_name="vector",  # Use the index name we created
                embedding_node_property="embedding",
                text_node_property="embedding_text"
            )

            # Perform similarity search
            docs = vector_store.similarity_search(
                query=user_query,
                k=top_k
            )

            # Convert results to structured format
            results = []
            for doc in docs:
                result_item = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', None)
                }
                results.append(result_item)

            confidence = 0.7 if results else 0.0
            if results and len(results) >= 3:
                confidence = 0.8

            return {
                "type": "similarity",
                "results": results,
                "confidence": confidence,
                "query": user_query,
                "total_results": len(results)
            }

        except Exception as e:
            print(f"Neo4jVector similarity search failed: {e}")
            # Try direct vector query
            return self._direct_vector_search(user_query, top_k)

    def _direct_vector_search(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Direct vector search using Neo4j vector procedures
        """
        try:
            query_embedding = self.embeddings.embed_query(user_query)

            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    # Try different vector query methods
                    try:
                        # Try the standard vector query
                        vector_query = """
                        CALL db.index.vector.queryNodes('vector', $k, $embedding)
                        YIELD node, score
                        RETURN node.embedding_text as content, 
                               labels(node)[0] as node_type,
                               elementId(node) as node_id,
                               score
                        ORDER BY score DESC
                        """

                        result = session.run(vector_query, k=top_k, embedding=query_embedding)

                    except Exception as e1:
                        print(f"Vector query failed: {e1}")
                        # Fallback to manual cosine similarity
                        return self._fallback_similarity_search(user_query, top_k)

                    results = []
                    for record in result:
                        if record["content"]:
                            results.append({
                                "content": record["content"],
                                "metadata": {
                                    "node_id": record["node_id"],
                                    "node_type": record["node_type"]
                                },
                                "score": record["score"]
                            })

                    confidence = 0.7 if results else 0.0

                    return {
                        "type": "similarity",
                        "results": results,
                        "confidence": confidence,
                        "query": user_query,
                        "total_results": len(results)
                    }

        except Exception as e:
            print(f"Direct vector search failed: {e}")
            return self._fallback_similarity_search(user_query, top_k)

    def _fallback_similarity_search(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Fallback similarity search using manual cosine similarity computation
        """
        try:
            query_embedding = self.embeddings.embed_query(user_query)

            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    # Get all nodes with embeddings
                    cypher_query = """
                    MATCH (n) 
                    WHERE n.embedding IS NOT NULL 
                    RETURN elementId(n) as node_id, 
                           n.embedding as embedding, 
                           n.embedding_text as text,
                           labels(n)[0] as node_type
                    LIMIT 100
                    """

                    result = session.run(cypher_query)
                    nodes_data = []

                    for record in result:
                        if record["embedding"] and record["text"]:
                            # Calculate cosine similarity
                            node_embedding = np.array(record["embedding"])
                            query_embedding_np = np.array(query_embedding)

                            # Cosine similarity calculation
                            dot_product = np.dot(query_embedding_np, node_embedding)
                            norm_query = np.linalg.norm(query_embedding_np)
                            norm_node = np.linalg.norm(node_embedding)

                            if norm_query > 0 and norm_node > 0:
                                similarity = dot_product / (norm_query * norm_node)

                                nodes_data.append({
                                    "content": record["text"],
                                    "metadata": {
                                        "node_id": record["node_id"],
                                        "node_type": record["node_type"]
                                    },
                                    "score": float(similarity)
                                })

                    # Sort by similarity and take top_k
                    nodes_data.sort(key=lambda x: x["score"], reverse=True)
                    top_results = nodes_data[:top_k]

                    confidence = 0.6 if top_results else 0.0

                    return {
                        "type": "similarity",
                        "results": top_results,
                        "confidence": confidence,
                        "query": user_query,
                        "total_results": len(top_results)
                    }

        except Exception as e:
            print(f"Fallback similarity search failed: {e}")
            return {
                "type": "similarity",
                "results": [],
                "confidence": 0.0,
                "error": str(e),
                "query": user_query,
                "total_results": 0
            }



    def _combine_results(self, cypher_result: Dict, similarity_result: Dict, user_query: str) -> str:
        """
        Combine and synthesize results from both approaches
        """
        combination_prompt = f"""
        Based on the user query: "{user_query}"

        I have results from two different search approaches:

        1. Cypher Query Results:
        {cypher_result.get('result', 'No results')}

        2. Similarity Search Results:
        {json.dumps(similarity_result.get('results', []), indent=2)}

        Please provide a comprehensive answer that:
        - Combines insights from both approaches
        - Prioritizes the most relevant information
        - Maintains accuracy and coherence
        - Clearly indicates if information is missing or uncertain

        Answer:
        """

        response = self.qa_llm.invoke(combination_prompt)
        return response.content

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main query method that routes to appropriate search strategy
        """
        query_type = self._classify_query_type(user_query)

        print(f"Query type classified as: {query_type}")

        if query_type == "cypher":
            result = self._cypher_search(user_query)
            return {
                "approach": "cypher",
                "answer": result["result"],
                "confidence": result["confidence"],
                "details": result
            }

        elif query_type == "similarity":
            result = self._similarity_search(user_query)
            # Generate natural language answer from similarity results
            if result["results"]:
                answer_prompt = f"""
                Based on these similar items found for the query "{user_query}":
                {json.dumps(result["results"], indent=2)}

                Provide a clear, informative answer.
                """
                answer = self.qa_llm.invoke(answer_prompt).content
            else:
                answer = "No similar items found for your query."

            return {
                "approach": "similarity",
                "answer": answer,
                "confidence": result["confidence"],
                "details": result
            }

        else:  # hybrid
            cypher_result = self._cypher_search(user_query)
            similarity_result = self._similarity_search(user_query)

            combined_answer = self._combine_results(cypher_result, similarity_result, user_query)

            return {
                "approach": "hybrid",
                "answer": combined_answer,
                "confidence": max(cypher_result.get("confidence", 0), similarity_result.get("confidence", 0)),
                "details": {
                    "cypher": cypher_result,
                    "similarity": similarity_result
                }
            }


def retrieve_nodes_with_hybrid_approach(uri, user, password, user_query):
    """
    Enhanced version of your original function with hybrid approach
    """
    hybrid_rag = HybridGraphRAG(uri, user, password)
    result = hybrid_rag.query(user_query)
    return result


# Setup vector embeddings in Neo4j (run once)
def setup_vector_embeddings(uri, user, password):
    """
    Setup vector embeddings for existing nodes with proper index creation
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # First, check Neo4j version to determine the correct syntax
            version_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
            version = version_result.single()["version"]
            print(f"Neo4j version: {version}")

            # Get all nodes with their meaningful content
            nodes_query = """
            MATCH (n)
            WITH n, labels(n)[0] as nodeType
            OPTIONAL MATCH (n)-[r]->(m)
            WITH n, nodeType, collect(DISTINCT type(r) + ':' + coalesce(m.node_id, toString(elementId(m)))) as relationships
            RETURN elementId(n) as node_id, nodeType, relationships,
                   CASE 
                     WHEN nodeType = 'Task' THEN coalesce(n.usecase, n.node_id, '')
                     WHEN nodeType = 'Attribute' THEN n.node_id + ' statistics: mean=' + toString(coalesce(n.mean, '')) + ' min=' + toString(coalesce(n.min, '')) + ' max=' + toString(coalesce(n.max, '')) + ' std_dev=' + toString(coalesce(n.std_dev, ''))
                     WHEN nodeType = 'Dataset' THEN n.node_id + ' domain=' + coalesce(n.domain, '') + ' location=' + coalesce(n.locationOfDataRecording, '') + ' date=' + coalesce(n.dateOfRecording, '') + ' rows=' + toString(coalesce(n.amountOfRows, '')) + ' attributes=' + toString(coalesce(n.amountOfAttributes, ''))
                     WHEN nodeType = 'ProcessedAttribute' THEN n.node_id + ' processed attribute'
                     WHEN nodeType = 'SHAPValue' THEN n.node_id + ' SHAP value'
                     WHEN nodeType = 'Model' THEN n.node_id + ' machine learning model'
                     WHEN nodeType = 'Preprocessing' THEN n.node_id + ' preprocessing step'
                     WHEN nodeType IN ['Material', 'Screw', 'Mechanical_Component', 'TestCase', 'Robotarm', 'Gripper'] THEN n.node_id + ' ' + toLower(nodeType)
                     ELSE coalesce(n.node_id, toString(elementId(n)))
                   END as text_content
            """

            result = session.run(nodes_query)
            nodes_data = [(record["node_id"], record["nodeType"], record["relationships"], record["text_content"])
                          for record in result if record["text_content"]]

            # Generate embeddings for meaningful content
            texts = [f"{node[1]}: {node[3]} relationships: {', '.join(node[2]) if node[2] else 'none'}"
                     for node in nodes_data]

            if texts:
                print(f"Generating embeddings for {len(texts)} nodes...")
                text_embeddings = embeddings.embed_documents(texts)

                # Store embeddings back to nodes
                for (node_id, node_type, relationships, text), embedding in zip(nodes_data, text_embeddings):
                    session.run(
                        "MATCH (n) WHERE elementId(n) = $node_id SET n.embedding = $embedding, n.embedding_text = $text",
                        node_id=node_id,
                        embedding=embedding,
                        text=f"{node_type}: {text} relationships: {', '.join(relationships) if relationships else 'none'}"
                    )

                print("Embeddings stored successfully")

                # Drop existing vector index if it exists
                try:
                    session.run("DROP INDEX vector IF EXISTS")
                    print("Dropped existing vector index")
                except:
                    pass

                # Get all unique node labels first
                label_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                available_labels = label_result.single()["labels"]
                print(f"Available node labels: {available_labels}")

                # Create vector index with proper syntax for Neo4j 5.x
                try:
                    # Try with the first available label (Neo4j requires at least one label)
                    if available_labels:
                        first_label = available_labels[0]
                        session.run(f"""
                        CREATE VECTOR INDEX vector IF NOT EXISTS
                        FOR (n:{first_label}) ON (n.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """)
                        print(f"Vector index 'vector' created successfully for label '{first_label}'")
                    else:
                        raise Exception("No node labels found")

                    # Test the index
                    session.run("CALL db.awaitIndexes()")
                    print("Vector index is ready")

                except Exception as e1:
                    print(f"First vector index creation failed: {e1}")
                    try:
                        # Try with multiple specific labels
                        common_labels = [label for label in
                                         ['Task', 'Model', 'Dataset', 'Attribute', 'ProcessedAttribute', 'SHAPValue',
                                          'Preprocessing'] if label in available_labels]
                        if common_labels:
                            # Create index for each label type
                            for label in common_labels:
                                try:
                                    session.run(f"""
                                    CREATE VECTOR INDEX vector_{label.lower()} IF NOT EXISTS
                                    FOR (n:{label}) ON (n.embedding)
                                    OPTIONS {{indexConfig: {{
                                        `vector.dimensions`: 1536,
                                        `vector.similarity_function`: 'cosine'
                                    }}}}
                                    """)
                                    print(f"Vector index created for {label}")
                                except Exception as label_error:
                                    print(f"Failed to create index for {label}: {label_error}")
                        else:
                            raise Exception("No common labels found")
                    except Exception as e2:
                        print(f"Second vector index creation failed: {e2}")
                        try:
                            # Try procedure call method
                            labels_for_procedure = common_labels if 'common_labels' in locals() else available_labels[
                                                                                                     :5]  # Limit to first 5 labels
                            session.run("""
                            CALL db.index.vector.createNodeIndex('vector', $labels, 'embedding', 1536, 'cosine')
                            """, labels=labels_for_procedure)
                            print("Vector index created with procedure call")
                        except Exception as e3:
                            print(f"All vector index creation methods failed: {e1}, {e2}, {e3}")
                            print("Will use fallback similarity search")

                # Verify the index was created
                try:
                    indexes = session.run("SHOW INDEXES").data()
                    vector_indexes = [idx for idx in indexes if 'vector' in idx.get('name', '').lower()]
                    print(f"Available vector indexes: {vector_indexes}")
                except:
                    print("Could not verify indexes")

                print(f"Setup complete for {len(nodes_data)} nodes")




if __name__ == "__main__":
    import os
    #setup_vector_embeddings(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    # Setup embeddings (run once)
    # setup_vector_embeddings(uri, user, password)
    # Test queries tailored to your schema
    test_queries = [
        "Welche Modelle haben die besten SHAP values?",  # Cypher query
        "Was sind die besten Modelle für die screw placement task nach ROC AUC score?"
    ]

    hybrid_rag = HybridGraphRAG(uri, user, password)

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")

        result = hybrid_rag.query(query)
        print(f"Approach: {result['approach']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Answer: {result['answer']}")

        if result['approach'] == 'hybrid':
            print("\nDetailed Results:")
            print(f"Cypher: {result['details']['cypher'].get('result', 'No result')}")
            print(f"Similarity: Found {len(result['details']['similarity'].get('results', []))} similar items")
