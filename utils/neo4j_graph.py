import os
import json
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from utils.logger import SimpleLogger

logger = SimpleLogger(log_file="logs/neo4j.log")


class Neo4jGraphBuilder:
    """Builds comprehensive Neo4j causal graph from extracted events for stakeholder analysis."""
    
    def __init__(self, uri: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        self.logger = logger
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def clear_database(self):
        """Delete all nodes and relationships from the database."""
        with self.driver.session() as session:
            result = session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Cleared all nodes and relationships from database")
            return result.consume().counters
    
    def create_person_node(self, session, person_id: str, agent_description: str, 
                          vehicle_details: str = None, vin: str = None, license_plate: str = None,
                          age: int = None, gender: str = None, ethnicity: str = None):
        """Create or update a Person node with demographic and behavioral attributes."""
        query = """
        MERGE (p:Person {id: $person_id})
        SET p.description = $description,
            p.vehicle_details = $vehicle_details,
            p.vin = $vin,
            p.license_plate = $license_plate,
            p.updated_at = datetime()
        RETURN p
        """
        
        # Only set optional fields if they have values
        if age is not None:
            query = query.replace("p.updated_at = datetime()", "p.age = $age, p.updated_at = datetime()")
        if gender is not None:
            query = query.replace("p.updated_at = datetime()", "p.gender = $gender, p.updated_at = datetime()")
        if ethnicity is not None:
            query = query.replace("p.updated_at = datetime()", "p.ethnicity = $ethnicity, p.updated_at = datetime()")
        
        result = session.run(query, {
            "person_id": person_id,
            "description": agent_description,
            "vehicle_details": vehicle_details,
            "vin": vin,
            "license_plate": license_plate,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity
        })
        return result.single()
    
    def create_vehicle_node(self, session, vin: str, vehicle_details: str, 
                          license_plate: str = None, person_id: str = None):
        """Create or update a Vehicle node."""
        if not vin or vin == "N/A":
            return None
            
        query = """
        MERGE (v:Vehicle {vin: $vin})
        SET v.details = $details,
            v.license_plate = $license_plate,
            v.owner_id = $owner_id,
            v.updated_at = datetime()
        RETURN v
        """
        result = session.run(query, {
            "vin": vin,
            "details": vehicle_details,
            "license_plate": license_plate,
            "owner_id": person_id
        })
        return result.single()
    
    def create_causal_event_node(self, session, event_id: str, crash_id: str, action: str, 
                               outcome: str, conditions: List[str], confidence: float,
                               mentions_alcohol: bool = None, event_type: str = "Causal"):
        """Create a CausalEvent node with rich causal attributes."""
        query = """
        MERGE (e:CausalEvent {id: $event_id})
        SET e.crash_id = $crash_id,
            e.action = $action,
            e.outcome = $outcome,
            e.conditions = $conditions,
            e.confidence = $confidence,
            e.mentions_alcohol = $mentions_alcohol,
            e.event_type = $event_type,
            e.severity_impact = $severity_impact,
            e.created_at = datetime()
        RETURN e
        """
        
        # Determine severity impact based on outcome
        severity_impact = self._assess_severity_impact(outcome)
        
        result = session.run(query, {
            "event_id": event_id,
            "crash_id": crash_id,
            "action": action,
            "outcome": outcome,
            "conditions": conditions,
            "confidence": confidence,
            "mentions_alcohol": mentions_alcohol,
            "event_type": event_type,
            "severity_impact": severity_impact
        })
        return result.single()
    
    def _assess_severity_impact(self, outcome: str) -> str:
        """Assess severity impact based on outcome description."""
        outcome_lower = outcome.lower()
        if any(word in outcome_lower for word in ['fatal', 'death', 'killed', 'died']):
            return "Fatal"
        elif any(word in outcome_lower for word in ['injury', 'injured', 'hospital', 'medical']):
            return "Injury"
        elif any(word in outcome_lower for word in ['damage', 'collision', 'struck', 'hit']):
            return "Property_Damage"
        else:
            return "Minor"
    
    def create_condition_nodes(self, session, conditions: List[str], event_id: str):
        """Create Condition nodes for environmental and contextual factors."""
        for condition in conditions:
            if condition and condition.strip():
                # Create condition node
                session.run("""
                    MERGE (c:Condition {name: $condition})
                    SET c.type = $type,
                        c.severity = $severity,
                        c.updated_at = datetime()
                """, {
                    "condition": condition,
                    "type": self._classify_condition_type(condition),
                    "severity": self._assess_condition_severity(condition)
                })
                
                # Link condition to event
                session.run("""
                    MATCH (e:CausalEvent {id: $event_id})
                    MATCH (c:Condition {name: $condition})
                    MERGE (e)-[:OCCURRED_UNDER]->(c)
                """, {"event_id": event_id, "condition": condition})
    
    def _classify_condition_type(self, condition: str) -> str:
        """Classify condition type for better analysis."""
        condition_lower = condition.lower()
        if any(word in condition_lower for word in ['alcohol', 'drunk', 'dwi', 'dui', 'impairment']):
            return "Impairment"
        elif any(word in condition_lower for word in ['wet', 'rain', 'snow', 'ice', 'slippery']):
            return "Weather"
        elif any(word in condition_lower for word in ['dark', 'light', 'dusk', 'dawn', 'night']):
            return "Lighting"
        elif any(word in condition_lower for word in ['speed', 'fast', 'slow', 'unsafe']):
            return "Speed"
        elif any(word in condition_lower for word in ['distraction', 'phone', 'texting', 'attention']):
            return "Distraction"
        else:
            return "Other"
    
    def _assess_condition_severity(self, condition: str) -> str:
        """Assess condition severity level."""
        condition_lower = condition.lower()
        if any(word in condition_lower for word in ['alcohol', 'impairment', 'unsafe speed']):
            return "High"
        elif any(word in condition_lower for word in ['wet', 'dark', 'distraction']):
            return "Medium"
        else:
            return "Low"
    
    def create_causal_relationships(self, session, person_id: str, event_id: str, 
                                  affected_entities: List[str], referenced_units: List[str],
                                  conditions: List[str]):
        """Create comprehensive causal relationships for stakeholder analysis."""
        
        # Person CAUSES Event (primary causal relationship)
        session.run("""
            MATCH (p:Person {id: $person_id})
            MATCH (e:CausalEvent {id: $event_id})
            MERGE (p)-[:CAUSES {relationship_type: 'primary_cause'}]->(e)
        """, {"person_id": person_id, "event_id": event_id})
        
        # Event RESULTS_IN outcomes
        session.run("""
            MATCH (e:CausalEvent {id: $event_id})
            MERGE (o:Outcome {description: e.outcome, crash_id: e.crash_id})
            MERGE (e)-[:RESULTS_IN]->(o)
        """, {"event_id": event_id})
        
        # Create condition nodes and link to event
        self.create_condition_nodes(session, conditions, event_id)
        
        # Event AFFECTS other entities
        for entity in affected_entities:
            if entity and entity != "N/A":
                # Try to match as Vehicle by VIN
                session.run("""
                    MATCH (e:CausalEvent {id: $event_id})
                    MATCH (v:Vehicle {vin: $entity})
                    MERGE (e)-[:AFFECTS {impact_type: 'direct'}]->(v)
                """, {"event_id": event_id, "entity": entity})
                
                # Try to match as Person by ID
                session.run("""
                    MATCH (e:CausalEvent {id: $event_id})
                    MATCH (p:Person {id: $entity})
                    MERGE (e)-[:AFFECTS {impact_type: 'direct'}]->(p)
                """, {"event_id": event_id, "entity": entity})
        
        # Event INVOLVES units (map generic labels like 'Unit 1' to canonical Unit_ID if available)
        for unit in referenced_units:
            if not unit or unit == "N/A":
                continue
            mapped = unit
            # Attempt to parse agent crash id from event id prefix
            crash_id = None
            if event_id and "_" in event_id:
                crash_id = event_id.split("_")[0]
            if crash_id:
                # If the unit label looks like a generic 'Unit X', record as label but avoid proliferating nodes
                # Prefer linking to Person when possible, otherwise create Unit node with normalized name
                pass
            session.run("""
                MATCH (e:CausalEvent {id: $event_id})
                MERGE (u:Unit {name: $name})
                MERGE (e)-[:INVOLVES]->(u)
            """, {"event_id": event_id, "name": mapped})
    
    def create_crash_context(self, session, crash_id: str, extraction_data: Dict[str, Any]):
        """Create crash context nodes for comprehensive analysis."""
        
        # Create Crash node
        session.run("""
            MERGE (cr:Crash {id: $crash_id})
            SET cr.total_events = $total_events,
                cr.severity_level = $severity_level,
                cr.created_at = datetime()
        """, {
            "crash_id": crash_id,
            "total_events": len(extraction_data.get("events", [])),
            "severity_level": self._assess_crash_severity(extraction_data.get("events", []))
        })
        
        # Link all events to crash
        for i, event in enumerate(extraction_data.get("events", [])):
            event_id = f"{crash_id}_{event['agent_id']}_{i}"
            session.run("""
                MATCH (cr:Crash {id: $crash_id})
                MATCH (e:CausalEvent {id: $event_id})
                MERGE (cr)-[:CONTAINS]->(e)
            """, {"crash_id": crash_id, "event_id": event_id})
    
    def _assess_crash_severity(self, events: List[Dict]) -> str:
        """Assess overall crash severity based on events."""
        if not events:
            return "Unknown"
        
        # Check for fatal events
        for event in events:
            outcome = event.get("outcome", "").lower()
            if any(word in outcome for word in ['fatal', 'death', 'killed']):
                return "Fatal"
        
        # Check for injury events
        for event in events:
            outcome = event.get("outcome", "").lower()
            if any(word in outcome for word in ['injury', 'injured', 'hospital']):
                return "Injury"
        
        return "Property_Damage"
    
    def process_extraction(self, session, extraction: Dict[str, Any]):
        """Process a single extraction and create comprehensive causal graph."""
        crash_id = extraction.get("crash_id")
        events = extraction.get("events", [])
        
        if not events:
            return
        
        # Create crash context first
        self.create_crash_context(session, crash_id, extraction)
        
        for i, event in enumerate(events):
            # Create unique event ID
            event_id = f"{crash_id}_{event['agent_id']}_{i}"
            
            # Extract person details
            person_id = event.get("agent_id")
            agent_description = event.get("agent_description", "")
            
            # Extract vehicle details from affected_entities or create from context
            vin = None
            vehicle_details = None
            license_plate = None
            
            # Try to find VIN in affected_entities
            affected_entities = event.get("affected_entities") or []
            for entity in affected_entities:
                if entity and len(entity) == 17 and entity.isalnum():  # VIN format
                    vin = entity
                    break
            
            # Create Person node with demographic data
            self.create_person_node(
                session, person_id, agent_description, 
                vehicle_details, vin, license_plate
            )
            
            # Create Vehicle node if we have VIN
            if vin:
                self.create_vehicle_node(
                    session, vin, vehicle_details or agent_description,
                    license_plate, person_id
                )
            
            # Create CausalEvent node with rich attributes
            self.create_causal_event_node(
                session, event_id, crash_id,
                event.get("action", ""),
                event.get("outcome", ""),
                event.get("conditions", []),
                event.get("causal_confidence", 0.0),
                event.get("mentions_alcohol_or_drugs")
            )
            
            # Normalize referenced units: map 'Unit X' -> f"{crash_id}_{X}" if possible
            referenced_units = event.get("referenced_units") or []
            normalized_units: List[str] = []
            for u in referenced_units:
                if not u:
                    continue
                u_str = str(u)
                # Match patterns like Unit 1, UNIT-01, u2
                import re
                m = re.match(r"(?i)\b(?:unit|u)[-\s#]?0*([1-9]\d*)\b", u_str)
                if m and crash_id:
                    normalized_units.append(f"{crash_id}_{m.group(1)}")
                else:
                    normalized_units.append(u_str)

            # Create comprehensive causal relationships
            self.create_causal_relationships(
                session, person_id, event_id,
                affected_entities,
                normalized_units,
                event.get("conditions") or []
            )
    
    def build_graph_from_jsonl(self, jsonl_file: str, clear_first: bool = False):
        """Build Neo4j graph from JSONL file of extractions."""
        if clear_first:
            self.logger.info("Clearing database before building graph...")
            self.clear_database()
        
        with self.driver.session() as session:
            processed_count = 0
            event_count = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        extraction = json.loads(line.strip())
                        self.process_extraction(session, extraction)
                        processed_count += 1
                        event_count += len(extraction.get("events", []))
                        
                        if processed_count % 10 == 0:
                            self.logger.info(f"Processed {processed_count} extractions, {event_count} events")
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_num}: {e}")
            
            self.logger.info(f"Graph construction complete: {processed_count} extractions, {event_count} events")
    
    def get_graph_stats(self):
        """Get comprehensive statistics about the causal graph."""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
            for record in result:
                stats[f"{record['label']}_count"] = record['count']
            
            # Count relationships by type
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            for record in result:
                stats[f"{record['rel_type']}_count"] = record['count']
            
            # Causal analysis statistics
            stats.update(self._get_causal_analysis_stats(session))
            
            return stats
    
    def _get_causal_analysis_stats(self, session):
        """Get causal analysis statistics for stakeholders."""
        stats = {}
        
        # High-risk causal patterns
        result = session.run("""
            MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)
            WHERE e.mentions_alcohol = true
            RETURN count(DISTINCT p) as alcohol_related_persons
        """)
        stats['alcohol_related_persons'] = result.single()['alcohol_related_persons']
        
        # Severity distribution
        result = session.run("""
            MATCH (e:CausalEvent)
            RETURN e.severity_impact as severity, count(e) as count
        """)
        for record in result:
            stats[f"severity_{record['severity']}_events"] = record['count']
        
        # Condition type distribution
        result = session.run("""
            MATCH (c:Condition)
            RETURN c.type as condition_type, count(c) as count
        """)
        for record in result:
            stats[f"condition_{record['condition_type']}_count"] = record['count']
        
        # Crash severity distribution
        result = session.run("""
            MATCH (cr:Crash)
            RETURN cr.severity_level as severity, count(cr) as count
        """)
        for record in result:
            stats[f"crash_severity_{record['severity']}"] = record['count']
        
        return stats
    
    def get_causal_motifs(self, min_frequency: int = 2):
        """Identify frequent causal patterns (motifs) for stakeholder analysis."""
        with self.driver.session() as session:
            motifs = []
            
            # Pattern 1: Alcohol + Speed + Weather
            result = session.run("""
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)-[:OCCURRED_UNDER]->(c1:Condition {type: 'Impairment'})
                MATCH (e)-[:OCCURRED_UNDER]->(c2:Condition {type: 'Speed'})
                MATCH (e)-[:OCCURRED_UNDER]->(c3:Condition {type: 'Weather'})
                RETURN 'Alcohol_Speed_Weather' as motif, count(DISTINCT e) as frequency
            """)
            for record in result:
                if record['frequency'] >= min_frequency:
                    motifs.append({
                        'pattern': record['motif'],
                        'frequency': record['frequency'],
                        'description': 'Driver under influence + speeding + adverse weather'
                    })
            
            # Pattern 2: Distraction + Intersection
            result = session.run("""
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)-[:OCCURRED_UNDER]->(c1:Condition {type: 'Distraction'})
                MATCH (e)-[:OCCURRED_UNDER]->(c2:Condition)
                WHERE c2.name CONTAINS 'intersection'
                RETURN 'Distraction_Intersection' as motif, count(DISTINCT e) as frequency
            """)
            for record in result:
                if record['frequency'] >= min_frequency:
                    motifs.append({
                        'pattern': record['motif'],
                        'frequency': record['frequency'],
                        'description': 'Distracted driving at intersections'
                    })
            
            return motifs
    
    def get_risk_factors_analysis(self):
        """Analyze risk factors and their associations with crash severity."""
        with self.driver.session() as session:
            analysis = {}
            
            # Alcohol impact on severity
            result = session.run("""
                MATCH (e:CausalEvent)
                WHERE e.mentions_alcohol = true
                RETURN e.severity_impact as severity, count(e) as count
            """)
            analysis['alcohol_severity_impact'] = {record['severity']: record['count'] for record in result}
            
            # Condition severity associations
            result = session.run("""
                MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)
                RETURN c.type as condition_type, e.severity_impact as severity, count(e) as count
            """)
            condition_severity = {}
            for record in result:
                key = f"{record['condition_type']}_{record['severity']}"
                condition_severity[key] = record['count']
            analysis['condition_severity_associations'] = condition_severity
            
            return analysis


def main():
    """Main function to build Neo4j graph from extractions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Neo4j graph from causal event extractions")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="extractions.jsonl",
        help="Path to JSONL file containing extractions"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before building graph (default: append to existing)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics after building"
    )
    
    args = parser.parse_args()
    
    # Get Neo4j connection details from environment
    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    
    if not uri or not password:
        logger.error("NEO4J_URI and NEO4J_PASSWORD environment variables must be set")
        return
    
    logger.info(f"Connecting to Neo4j at {uri}")
    logger.info(f"Mode: {'Clear and rebuild' if args.clear else 'Append to existing'}")
    
    builder = Neo4jGraphBuilder(uri, password)
    
    try:
        # Build the comprehensive causal graph
        builder.build_graph_from_jsonl(args.jsonl, clear_first=args.clear)
        
        if args.stats:
            # Get comprehensive statistics
            stats = builder.get_graph_stats()
            logger.info("=== COMPREHENSIVE CAUSAL GRAPH STATISTICS ===")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            # Run causal motif analysis
            logger.info("\n=== CAUSAL MOTIFS ANALYSIS ===")
            motifs = builder.get_causal_motifs(min_frequency=1)
            for motif in motifs:
                logger.info(f"  Pattern: {motif['pattern']}")
                logger.info(f"  Frequency: {motif['frequency']}")
                logger.info(f"  Description: {motif['description']}")
            
            # Run risk factors analysis
            logger.info("\n=== RISK FACTORS ANALYSIS ===")
            risk_analysis = builder.get_risk_factors_analysis()
            logger.info("Alcohol Severity Impact:")
            for severity, count in risk_analysis['alcohol_severity_impact'].items():
                logger.info(f"  {severity}: {count} events")
            
            logger.info("\nCondition-Severity Associations:")
            for key, count in risk_analysis['condition_severity_associations'].items():
                logger.info(f"  {key}: {count} events")
                
    except Exception as e:
        logger.error(f"Error building graph: {e}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
