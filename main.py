import argparse
import json
import os
import signal
import sys
from typing import List, Optional
import getpass
from utils.logger import SimpleLogger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("GOOGLE_API_KEY"):
  # If numbered keys exist, default to the first one without prompting
  numbered_keys = [os.environ.get(f"GOOGLE_API_KEY_{i}") for i in range(1, 11)]
  numbered_keys = [k for k in numbered_keys if k]
  if numbered_keys:
    os.environ["GOOGLE_API_KEY"] = numbered_keys[0]
  else:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("""
  To use Google Gemini, you need to provide an API key.
  If you don't have an API key, you can create one from here: https://aistudio.google.com/app/apikey
  Enter API key for Google Gemini: """)

import pandas as pd
from pydantic import BaseModel, Field

# New modular utilities
from utils.prompting import build_prompt
from utils.gemini_client import RotatingGeminiClient
from utils.preprocess import build_id_maps, replace_mentions

logger = SimpleLogger(log_file="logs/main.log")

# Global variables for progress tracking
total_processed = 0
extractions_path = None

def signal_handler(signum, frame):
    """Handle process termination gracefully"""
    global total_processed, extractions_path
    logger.info(f"Process terminated (signal {signum}). Progress saved: {total_processed} records")
    if extractions_path:
        logger.info(f"Extractions saved to: {extractions_path}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


class CausalEvent(BaseModel):
    """Single causal tuple extracted from a narrative for Neo4j graph construction."""

    # Primary entity identification (use actual IDs from dataset when available)
    agent_id: str = Field(description="Unique identifier for the primary actor (Person_ID, Unit_ID, or descriptive ID)")
    agent_type: str = Field(description="Type of agent: Driver, Pedestrian, Vehicle, or Other")
    agent_description: str = Field(description="Human-readable description of the agent (e.g., 'Driver of 2009 Black Ford Focus')")
    
    # Event details
    action: str = Field(description="Specific action taken by the agent (e.g., 'disregarded red signal', 'hydroplaned')")
    conditions: List[str] = Field(
        description="Environmental or contextual conditions (e.g., 'wet roadway', 'dark conditions', 'alcohol impairment')"
    )
    outcome: str = Field(description="Immediate result of the action (e.g., 'collided with barrier', 'struck Unit 2')")
    
    # Metadata for graph construction
    causal_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence that action caused the outcome"
    )
    mentions_alcohol_or_drugs: Optional[bool] = Field(
        default=None, description="Whether impairment is explicitly mentioned"
    )
    
    # Related entities (for graph edges)
    affected_entities: Optional[List[str]] = Field(
        default=None, description="IDs of other entities affected by this event"
    )
    referenced_units: Optional[List[str]] = Field(
        default=None, description="Unit numbers mentioned in the narrative (e.g., 'Unit 1', 'Unit 2')"
    )


class Extraction(BaseModel):
    """Structured extraction for a single narrative, possibly containing multiple events."""

    crash_id: Optional[str] = Field(default=None, description="Crash identifier if available from input row")
    events: List[CausalEvent]

## Inline prompt builder removed; using utils.prompting.build_prompt


def extract_events_for_rows(df: pd.DataFrame, limit: int) -> List[Extraction]:
    prompt = build_prompt()
    # Use rotating Gemini client for rate-limit and API-key rotation
    client = RotatingGeminiClient()
    logger.info("Building prompt")
    extractions: List[Extraction] = []
    logger.info("Extracting events")
    for _, row in tqdm(df.head(limit).iterrows(), total=limit):
        narrative = str(row.get("Investigator_Narrative", "")).strip()
        crash_id = str(row.get("Crash_ID", "")).strip() or None
        if not narrative: continue

        # Extract entity context for better identification
        person_id = str(row.get("Person_ID", "")).strip() or "N/A"
        unit_id = str(row.get("Unit_ID", "")).strip() or "N/A"
        vin = str(row.get("VIN", "")).strip() or "N/A"
        license_plate = str(row.get("Veh_Lic_Plate_Nbr", "")).strip() or "N/A"
        
        # Build vehicle details string
        vehicle_parts = []
        if str(row.get("Veh_Mod_Year", "")).strip() != "N/A":
            vehicle_parts.append(f"{row.get('Veh_Mod_Year')}")
        if str(row.get("Veh_Color_ID", "")).strip() != "N/A":
            vehicle_parts.append(f"{row.get('Veh_Color_ID')}")
        if str(row.get("Veh_Make_ID", "")).strip() != "N/A":
            vehicle_parts.append(f"{row.get('Veh_Make_ID')}")
        if str(row.get("Veh_Mod_ID", "")).strip() != "N/A":
            vehicle_parts.append(f"{row.get('Veh_Mod_ID')}")
        vehicle_details = " ".join(vehicle_parts) if vehicle_parts else "N/A"

        # Preprocess narrative using CURSOR.md mapping rules
        id_maps = build_id_maps(row)
        safe_narrative = replace_mentions(narrative, id_maps, crash_id or "")

        # Invoke with model and key rotation and retries
        result: Extraction = client.run_structured(prompt, Extraction, {
            "crash_id": crash_id,
            "person_id": person_id,
            "unit_id": unit_id,
            "vin": vin,
            "license_plate": license_plate,
            "vehicle_details": vehicle_details,
            "narrative": safe_narrative
        })
        extractions.append(result)
    logger.info(f"Extracted {len(extractions)} events")
    return extractions


def main():
    logger.info("Starting main function")
    parser = argparse.ArgumentParser(description="Unified pipeline: extract events, build Neo4j graph, and run stakeholder analysis.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "sample_texas_crash_data.csv"),
        help="Path to CSV file containing crash data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to process",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to write outputs (extractions, reports, queries)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (defaults to OUT_DIR/extractions.jsonl)",
    )
    parser.add_argument(
        "--use-existing-extractions",
        type=str,
        default=None,
        help="Skip extraction and use an existing extractions JSONL file",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Build Neo4j graph after extraction",
    )
    parser.add_argument(
        "--clear-neo4j",
        action="store_true",
        help="Clear Neo4j database before building graph (use with --neo4j)",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run stakeholder analysis after building graph",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate visualizations and dashboard from extractions",
    )
    args = parser.parse_args()

    logger.info(f"Reading CSV file: {args.csv}")
    df = pd.read_csv(args.csv)

    # Defensive: normalize column names we depend on
    logger.info("Normalizing column names")
    cols_lower = {c.lower(): c for c in df.columns}
    narrative_col = cols_lower.get("investigator_narrative")
    crash_id_col = cols_lower.get("crash_id")
    if not narrative_col:
        raise ValueError("Investigator_Narrative column not found in CSV")

    # Keep required columns for entity identification
    use_cols = [narrative_col]
    if crash_id_col:
        use_cols.append(crash_id_col)
    
    # Add entity identification columns if available
    entity_cols = ["person_id", "unit_id", "vin", "veh_lic_plate_nbr", "veh_mod_year", "veh_color_id", "veh_make_id", "veh_mod_id"]
    for col in entity_cols:
        if col in cols_lower:
            use_cols.append(cols_lower[col])
    
    logger.info(f"Using columns: {use_cols}")
    df = df[use_cols]

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Decide extraction source
    global extractions_path, total_processed
    extractions_path = args.output or os.path.join(args.out_dir, "extractions.jsonl")
    if args.use_existing_extractions:
        if not os.path.exists(args.use_existing_extractions):
            raise FileNotFoundError(f"Extractions file not found: {args.use_existing_extractions}")
        extractions_path = args.use_existing_extractions
        logger.info(f"Skipping extraction. Using existing extractions at: {extractions_path}")
    else:
        # Process and save incrementally to avoid losing progress
        logger.info(f"Processing {args.limit} records with incremental saving...")
        
        # Check if we have existing progress to resume from
        existing_count = 0
        if os.path.exists(extractions_path):
            with open(extractions_path, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for line in f if line.strip())
            logger.info(f"Found {existing_count} existing extractions, resuming from record {existing_count + 1}")
        
        # Open file in append mode to resume from where we left off
        mode = "a" if existing_count > 0 else "w"
        with open(extractions_path, mode, encoding="utf-8") as f:
            # Skip already processed records
            start_index = existing_count
            total_processed = existing_count
            
            for i, (_, row) in enumerate(tqdm(df.head(args.limit).iterrows(), total=args.limit, desc="Extracting events")):
                # Skip records we've already processed
                if i < start_index:
                    continue
                    
                try:
                    # Extract single record
                    narrative = str(row.get("Investigator_Narrative", "")).strip()
                    crash_id = str(row.get("Crash_ID", "")).strip() or None
                    if not narrative: 
                        continue

                    # Extract entity context for better identification
                    person_id = str(row.get("Person_ID", "")).strip() or "N/A"
                    unit_id = str(row.get("Unit_ID", "")).strip() or "N/A"
                    vin = str(row.get("VIN", "")).strip() or "N/A"
                    license_plate = str(row.get("Veh_Lic_Plate_Nbr", "")).strip() or "N/A"
                    
                    # Build vehicle details string
                    vehicle_parts = []
                    if str(row.get("Veh_Mod_Year", "")).strip() != "N/A":
                        vehicle_parts.append(f"{row.get('Veh_Mod_Year')}")
                    if str(row.get("Veh_Color_ID", "")).strip() != "N/A":
                        vehicle_parts.append(f"{row.get('Veh_Color_ID')}")
                    if str(row.get("Veh_Make_ID", "")).strip() != "N/A":
                        vehicle_parts.append(f"{row.get('Veh_Make_ID')}")
                    if str(row.get("Veh_Mod_ID", "")).strip() != "N/A":
                        vehicle_parts.append(f"{row.get('Veh_Mod_ID')}")
                    vehicle_details = " ".join(vehicle_parts) if vehicle_parts else "N/A"

                    # Preprocess narrative using CURSOR.md mapping rules
                    from utils.preprocess import build_id_maps, replace_mentions
                    id_maps = build_id_maps(row)
                    safe_narrative = replace_mentions(narrative, id_maps, crash_id or "")

                    # Use rotating Gemini client for rate-limit resilience
                    from utils.gemini_client import RotatingGeminiClient
                    from utils.prompting import build_prompt
                    client = RotatingGeminiClient()
                    prompt = build_prompt()
                    structured = client.with_structured_output(Extraction)
                    
                    # Invoke with model and key rotation and retries
                    result: Extraction = client.run_structured(prompt, Extraction, {
                        "crash_id": crash_id,
                        "person_id": person_id,
                        "unit_id": unit_id,
                        "vin": vin,
                        "license_plate": license_plate,
                        "vehicle_details": vehicle_details,
                        "narrative": safe_narrative
                    })
                    
                    # Save immediately to avoid losing progress
                    f.write(result.model_dump_json() + "\n")
                    f.flush()  # Force write to disk
                    os.fsync(f.fileno())  # Force OS to write to disk
                    
                    total_processed += 1
                    
                    # Update database incrementally every 50 records
                    if total_processed % 50 == 0 and args.neo4j:
                        logger.info(f"Updating database with {total_processed} records...")
                        try:
                            from utils.neo4j_graph import Neo4jGraphBuilder
                            uri = os.environ.get("NEO4J_URI")
                            password = os.environ.get("NEO4J_PASSWORD")
                            
                            if uri and password:
                                builder = Neo4jGraphBuilder(uri, password)
                                # Clear database only on first update
                                clear_first = (total_processed == 50)
                                builder.build_graph_from_jsonl(extractions_path, clear_first=clear_first)
                                
                                # Show current database stats
                                stats = builder.get_graph_stats()
                                logger.info(f"Database updated: {stats.get('Crash_count', 0)} crashes, {stats.get('CausalEvent_count', 0)} events")
                                builder.close()
                            else:
                                logger.warning("Neo4j credentials not found, skipping database update")
                        except Exception as e:
                            logger.error(f"Error updating database: {e}")
                    
                    # Log progress more frequently for better monitoring
                    if total_processed % 10 == 0:  # Every 10 records
                        logger.info(f"Processed {total_processed} records, saved to {extractions_path}")
                    elif total_processed % 100 == 0:  # Every 100 records
                        logger.info(f"Milestone: {total_processed} records processed and saved")
                        
                except KeyboardInterrupt:
                    logger.info(f"Process interrupted by user. Progress saved: {total_processed} records")
                    break
                except Exception as e:
                    logger.error(f"Error processing record {i}: {e}")
                    # Continue with next record instead of stopping
                    continue
                    
        logger.info(f"Completed extraction of {total_processed} records, saved to {extractions_path}")
    
    # Build Neo4j graph if requested (final update with any remaining records)
    if args.neo4j:
        logger.info("Performing final Neo4j graph update...")
        try:
            from utils.neo4j_graph import Neo4jGraphBuilder
            
            uri = os.environ.get("NEO4J_URI")
            password = os.environ.get("NEO4J_PASSWORD")
            
            if not uri or not password:
                logger.error("NEO4J_URI and NEO4J_PASSWORD environment variables must be set")
                return
            
            builder = Neo4jGraphBuilder(uri, password)
            # Only clear if explicitly requested, otherwise just add new records
            builder.build_graph_from_jsonl(extractions_path, clear_first=args.clear_neo4j)
            
            # Show graph statistics
            stats = builder.get_graph_stats()
            logger.info("Neo4j Graph Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            # Motif mining summary
            motifs = builder.get_causal_motifs(min_frequency=1)
            if motifs:
                logger.info("Causal Motifs:")
                for m in motifs:
                    logger.info(f"  {m['pattern']}: {m['frequency']} - {m['description']}")
            
            # Optional stakeholder analysis
            if args.analysis:
                try:
                    from utils.stakeholder_analysis import StakeholderAnalyzer
                    analyzer = StakeholderAnalyzer(uri, password)
                    report = analyzer.generate_comprehensive_report()
                    report_path = os.path.join(args.out_dir, "stakeholder_report.json")
                    queries_path = os.path.join(args.out_dir, "stakeholder_queries.cypher")
                    with open(report_path, "w", encoding="utf-8") as rf:
                        json.dump(report, rf, indent=2)
                    analyzer.export_cypher_queries(queries_path)
                    analyzer.close()
                    logger.info(f"Stakeholder analysis complete. Outputs: {report_path}, {queries_path}")
                except Exception as e:
                    logger.error(f"Error running stakeholder analysis: {e}")

            builder.close()
            logger.info("Neo4j graph construction complete")
            
        except Exception as e:
            logger.error(f"Error building Neo4j graph: {e}")

    # Visualizations
    if args.viz:
        try:
            from utils.visualize import generate_visualizations
            viz_dir = os.path.join(args.out_dir, 'viz')
            os.makedirs(viz_dir, exist_ok=True)
            generate_visualizations(extractions_path, viz_dir)
            logger.info(f"Visualization artifacts written to {viz_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    logger.info("Finished main function")

if __name__ == "__main__":
    main()


