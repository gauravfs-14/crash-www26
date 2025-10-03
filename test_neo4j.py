#!/usr/bin/env python3
"""
Test script to verify Neo4j connection and create a sample graph.
"""

import os
import json
from neo4j_graph import Neo4jGraphBuilder
from utils.logger import SimpleLogger

logger = SimpleLogger(log_file="logs/test_neo4j.log")

def test_connection():
    """Test Neo4j connection."""
    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    
    if not uri or not password:
        logger.error("NEO4J_URI and NEO4J_PASSWORD environment variables must be set")
        return False
    
    try:
        builder = Neo4jGraphBuilder(uri, password)
        
        # Test connection
        with builder.driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
            logger.info(f"Neo4j connection test: {message}")
        
        builder.close()
        return True
        
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return False

def create_sample_graph():
    """Create a sample graph with test data."""
    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    
    # Create sample extraction data
    sample_data = {
        "crash_id": "TEST_001",
        "events": [
            {
                "agent_id": "TEST_001_1_1",
                "agent_type": "Driver",
                "agent_description": "Driver of 2020 Red Toyota Camry",
                "action": "disregarded red light",
                "conditions": ["intersection", "alcohol impairment"],
                "outcome": "crashed into Unit 2",
                "causal_confidence": 0.9,
                "mentions_alcohol_or_drugs": True,
                "affected_entities": ["Unit 2"],
                "referenced_units": ["Unit 1", "Unit 2"]
            }
        ]
    }
    
    # Write sample data to file
    with open("test_extractions.jsonl", "w") as f:
        f.write(json.dumps(sample_data) + "\n")
    
    try:
        builder = Neo4jGraphBuilder(uri, password)
        
        # Clear database and build sample graph
        logger.info("Creating sample graph...")
        builder.build_graph_from_jsonl("test_extractions.jsonl", clear_first=True)
        
        # Show statistics
        stats = builder.get_graph_stats()
        logger.info("Sample Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        builder.close()
        logger.info("Sample graph created successfully")
        
        # Clean up
        os.remove("test_extractions.jsonl")
        
    except Exception as e:
        logger.error(f"Error creating sample graph: {e}")

def main():
    """Main test function."""
    logger.info("Testing Neo4j integration...")
    
    # Test connection
    if not test_connection():
        return
    
    # Create sample graph
    create_sample_graph()

if __name__ == "__main__":
    main()
