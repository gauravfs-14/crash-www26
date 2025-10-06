#!/usr/bin/env python3
"""
Comprehensive Stakeholder Analysis for Crash Causal Networks
Provides detailed insights for different stakeholder groups.
"""

import os
import json
from typing import Dict, List, Any
from neo4j import GraphDatabase
from utils.logger import SimpleLogger

logger = SimpleLogger(log_file="logs/stakeholder_analysis.log")


class StakeholderAnalyzer:
    """Comprehensive causal analysis for different stakeholder groups."""
    
    def __init__(self, uri: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        self.logger = logger
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def analyze_for_traffic_safety_engineers(self):
        """Analysis focused on infrastructure and engineering solutions."""
        with self.driver.session() as session:
            analysis = {}
            
            # High-risk locations and conditions
            result = session.run("""
                MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)
                WHERE c.type IN ['Weather', 'Lighting', 'Speed']
                RETURN c.name as condition, c.type as type, count(e) as frequency
                ORDER BY frequency DESC
                LIMIT 10
            """)
            analysis['high_risk_conditions'] = [
                {
                    'condition': record['condition'],
                    'type': record['type'],
                    'frequency': record['frequency']
                } for record in result
            ]
            
            # Infrastructure-related patterns
            result = session.run("""
                MATCH (e:CausalEvent)
                WHERE e.action CONTAINS 'intersection' OR e.action CONTAINS 'signal'
                RETURN e.action as action, e.severity_impact as severity, count(e) as frequency
                ORDER BY frequency DESC
            """)
            analysis['infrastructure_issues'] = [
                {
                    'action': record['action'],
                    'severity': record['severity'],
                    'frequency': record['frequency']
                } for record in result
            ]
            
            return analysis
    
    def analyze_for_law_enforcement(self):
        """Analysis focused on enforcement priorities and patterns."""
        with self.driver.session() as session:
            analysis = {}
            
            # Alcohol-related incidents
            result = session.run("""
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)
                WHERE e.mentions_alcohol = true
                RETURN p.id as person_id, e.action as action, e.severity_impact as severity
                ORDER BY e.severity_impact DESC
            """)
            analysis['alcohol_incidents'] = [
                {
                    'person_id': record['person_id'],
                    'action': record['action'],
                    'severity': record['severity']
                } for record in result
            ]
            
            # Speeding patterns
            result = session.run("""
                MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition {type: 'Speed'})
                RETURN c.name as speed_condition, e.severity_impact as severity, count(e) as frequency
                ORDER BY frequency DESC
            """)
            analysis['speeding_patterns'] = [
                {
                    'condition': record['speed_condition'],
                    'severity': record['severity'],
                    'frequency': record['frequency']
                } for record in result
            ]
            
            return analysis
    
    def analyze_for_public_health_officials(self):
        """Analysis focused on public health impacts and prevention."""
        with self.driver.session() as session:
            analysis = {}
            
            # Severity distribution by demographics
            result = session.run("""
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)
                WHERE p.age IS NOT NULL AND p.age <> ""
                RETURN p.age as age_group, e.severity_impact as severity, count(e) as frequency
                ORDER BY age_group
            """)
            analysis['age_severity_analysis'] = [
                {
                    'age': record['age_group'],
                    'severity': record['severity'],
                    'frequency': record['frequency']
                } for record in result
            ]
            
            # Risk factor combinations
            result = session.run("""
                MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)
                WITH e, collect(c.type) as condition_types
                WHERE size(condition_types) > 1
                RETURN condition_types, e.severity_impact as severity, count(e) as frequency
                ORDER BY frequency DESC
            """)
            analysis['risk_factor_combinations'] = [
                {
                    'conditions': record['condition_types'],
                    'severity': record['severity'],
                    'frequency': record['frequency']
                } for record in result
            ]
            
            return analysis
    
    def analyze_for_insurance_industry(self):
        """Analysis focused on risk assessment and actuarial insights."""
        with self.driver.session() as session:
            analysis = {}
            
            # High-risk driver profiles
            result = session.run("""
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)
                WHERE e.severity_impact IN ['Fatal', 'Injury']
                RETURN p.id as person_id, 
                       coalesce(p.age, 'Unknown') as age, 
                       coalesce(p.gender, 'Unknown') as gender, 
                       count(e) as high_severity_events
                ORDER BY high_severity_events DESC
            """)
            analysis['high_risk_drivers'] = [
                {
                    'person_id': record['person_id'],
                    'age': record['age'],
                    'gender': record['gender'],
                    'high_severity_events': record['high_severity_events']
                } for record in result
            ]
            
            # Vehicle risk factors
            result = session.run("""
                MATCH (e:CausalEvent)-[:AFFECTS]->(v:Vehicle)
                WHERE e.severity_impact IN ['Fatal', 'Injury']
                RETURN v.vin as vin, 
                       coalesce(v.details, 'Unknown') as make, 
                       coalesce(v.details, 'Unknown') as model,
                       count(e) as high_severity_events
                ORDER BY high_severity_events DESC
            """)
            analysis['high_risk_vehicles'] = [
                {
                    'vin': record['vin'],
                    'make': record['make'],
                    'model': record['model'],
                    'high_severity_events': record['high_severity_events']
                } for record in result
            ]
            
            return analysis

    def analyze_causal_motifs(self, min_frequency: int = 1):
        """Mine common causal motifs similar to builder.get_causal_motifs."""
        with self.driver.session() as session:
            motifs: List[Dict[str, Any]] = []

            # Motif 1: Alcohol + Speed + Weather
            result = session.run(
                """
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)-[:OCCURRED_UNDER]->(c1:Condition {type: 'Impairment'})
                MATCH (e)-[:OCCURRED_UNDER]->(c2:Condition {type: 'Speed'})
                MATCH (e)-[:OCCURRED_UNDER]->(c3:Condition {type: 'Weather'})
                RETURN 'Alcohol_Speed_Weather' as motif, count(DISTINCT e) as frequency
                """
            )
            rec = result.single()
            if rec and rec["frequency"] >= min_frequency:
                motifs.append({
                    "pattern": rec["motif"],
                    "frequency": rec["frequency"],
                    "description": "Driver under influence + speeding + adverse weather"
                })

            # Motif 2: Distraction + Intersection
            result = session.run(
                """
                MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)-[:OCCURRED_UNDER]->(c1:Condition {type: 'Distraction'})
                MATCH (e)-[:OCCURRED_UNDER]->(c2:Condition)
                WHERE c2.name CONTAINS 'intersection'
                RETURN 'Distraction_Intersection' as motif, count(DISTINCT e) as frequency
                """
            )
            rec = result.single()
            if rec and rec["frequency"] >= min_frequency:
                motifs.append({
                    "pattern": rec["motif"],
                    "frequency": rec["frequency"],
                    "description": "Distracted driving at intersections"
                })

            return motifs
    
    def generate_comprehensive_report(self):
        """Generate comprehensive stakeholder analysis report."""
        report = {
            'traffic_safety_engineers': self.analyze_for_traffic_safety_engineers(),
            'law_enforcement': self.analyze_for_law_enforcement(),
            'public_health_officials': self.analyze_for_public_health_officials(),
            'insurance_industry': self.analyze_for_insurance_industry(),
            'causal_motifs': self.analyze_causal_motifs(min_frequency=1)
        }
        
        return report
    
    def export_cypher_queries(self, output_file: str = "stakeholder_queries.cypher"):
        """Export useful Cypher queries for stakeholders."""
        queries = {
            'traffic_safety_engineers': [
                "// High-risk conditions for infrastructure planning",
                "MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)",
                "WHERE c.type IN ['Weather', 'Lighting', 'Speed']",
                "RETURN c.name, c.type, count(e) as frequency",
                "ORDER BY frequency DESC;",
                "",
                "// Infrastructure-related crash patterns",
                "MATCH (e:CausalEvent)",
                "WHERE e.action CONTAINS 'intersection' OR e.action CONTAINS 'signal'",
                "RETURN e.action, e.severity_impact, count(e) as frequency;"
            ],
            'law_enforcement': [
                "// Alcohol-related incidents by severity",
                "MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)",
                "WHERE e.mentions_alcohol = true",
                "RETURN p.id, e.action, e.severity_impact;",
                "",
                "// Speeding patterns and outcomes",
                "MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition {type: 'Speed'})",
                "RETURN c.name, e.severity_impact, count(e) as frequency;"
            ],
            'public_health_officials': [
                "// Age and severity analysis",
                "MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)",
                "WHERE p.age IS NOT NULL AND p.age <> \"\"",
                "RETURN p.age, e.severity_impact, count(e) as frequency;",
                "",
                "// Risk factor combinations",
                "MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)",
                "WITH e, collect(c.type) as condition_types",
                "WHERE size(condition_types) > 1",
                "RETURN condition_types, e.severity_impact, count(e) as frequency;"
            ],
            'insurance_industry': [
                "// High-risk driver profiles",
                "MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)",
                "WHERE e.severity_impact IN ['Fatal', 'Injury']",
                "RETURN p.id, coalesce(p.age, 'Unknown') as age, coalesce(p.gender, 'Unknown') as gender, count(e) as high_severity_events;",
                "",
                "// Vehicle risk assessment",
                "MATCH (e:CausalEvent)-[:AFFECTS]->(v:Vehicle)",
                "WHERE e.severity_impact IN ['Fatal', 'Injury']",
                "RETURN v.vin, coalesce(v.details, 'Unknown') as make, coalesce(v.details, 'Unknown') as model, count(e) as high_severity_events;"
            ]
        }
        
        with open(output_file, 'w') as f:
            f.write("-- Stakeholder-Specific Cypher Queries\n")
            f.write("-- Generated for comprehensive crash causal analysis\n\n")
            
            for stakeholder, query_list in queries.items():
                f.write(f"-- {stakeholder.replace('_', ' ').title()} Queries\n")
                for query in query_list:
                    f.write(f"{query}\n")
                f.write("\n")
        
        self.logger.info(f"Exported stakeholder queries to {output_file}")


def main():
    """Main function for stakeholder analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive stakeholder analysis")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"), help="Output directory")
    parser.add_argument("--output", default=None, help="Output file for analysis (defaults to OUT_DIR/stakeholder_report.json)")
    parser.add_argument("--queries", default=None, help="Output file for Cypher queries (defaults to OUT_DIR/stakeholder_queries.cypher)")
    
    args = parser.parse_args()
    
    # Get connection details from environment
    uri = os.environ.get("NEO4J_URI", args.uri)
    password = os.environ.get("NEO4J_PASSWORD", args.password)
    
    analyzer = StakeholderAnalyzer(uri, password)
    
    try:
        # Generate comprehensive report
        logger.info("Generating comprehensive stakeholder analysis...")
        os.makedirs(args.out_dir, exist_ok=True)
        report_path = args.output or os.path.join(args.out_dir, 'stakeholder_report.json')
        queries_path = args.queries or os.path.join(args.out_dir, 'stakeholder_queries.cypher')

        report = analyzer.generate_comprehensive_report()

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Analysis report saved to {report_path}")

        # Export Cypher queries
        analyzer.export_cypher_queries(queries_path)
        
        # Print summary
        logger.info("\n=== STAKEHOLDER ANALYSIS SUMMARY ===")
        for stakeholder, analysis in report.items():
            logger.info(f"\n{stakeholder.replace('_', ' ').title()}:")
            for key, value in analysis.items():
                if isinstance(value, list):
                    logger.info(f"  {key}: {len(value)} items")
                else:
                    logger.info(f"  {key}: {value}")
                    
    except Exception as e:
        logger.error(f"Error in stakeholder analysis: {e}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
