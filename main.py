import argparse
import json
import os
from typing import List, Optional
import getpass
from utils.logger import SimpleLogger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("""
  To use Google Gemini, you need to provide an API key.
  If you don't have an API key, you can create one from here: https://aistudio.google.com/app/apikey
  Enter API key for Google Gemini: """)

import pandas as pd
from pydantic import BaseModel, Field

# LangChain - Google GenAI (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logger = SimpleLogger(log_file="logs/main.log")


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


SYSTEM_INSTRUCTIONS = (
    "You are an expert traffic crash analyst extracting events for Neo4j graph construction. "
    "CRITICAL: This narrative describes an entire crash with multiple vehicles/people. "
    "You are extracting events for ONE SPECIFIC person/vehicle (identified by the provided Person_ID, Unit_ID, VIN). "
    "Extract BOTH types of events: "
    "1. CAUSAL events where this person/vehicle is the primary actor (caused something) "
    "2. AFFECTED events where this person/vehicle was impacted by others' actions "
    "Use the provided identifiers as agent_id (Person_ID is preferred). "
    "Include vehicle details in agent_description (year, make, model, color). "
    "Map Texas crash report shorthand: NB/SB/EB/WB=direction; FD=Front Distributed; BD=Back Distributed; "
    "POI=Point of Impact; Unit 1/Unit 2=vehicle references; DUI/DWI=alcohol impairment. "
    "For AFFECTED events, use action like 'was struck by', 'was hit by', 'was affected by'. "
    "Extract ALL relevant events involving this person/vehicle, whether they caused or were affected. "
    "Use consistent, factual language. For Neo4j: ensure agent_id is unique and stable."
)


def build_few_shots() -> List[dict]:
    """Few-shot examples showing how to extract events for a specific person/vehicle from crash narrative."""
    examples = []

    # Example 1: Single vehicle crash - focus on the specific driver
    narrative_1 = (
        "I-10 west at mile marker 20 is a four lane public interstate. Wi-01 stated he was traveling west on I-10 behind unit-01. "
        "Wi-01 stated as they approached I-10 west at mile marker 20, unit-01 for an unknown reason veered from the number one lane "
        "onto the left shoulder colliding into a cement barricade. Unit-01 attempted flee the scene of the accident and continued west "
        "on I-10 until the vehicle became disabled and came to a stop on the right shoulder of I-10 west. Driver of unit-01 was believed "
        "to be under the influence of alcohol and or a controlled substance, drug, dangerous drug Case # 15-213199."
    )
    examples.append(
        {
            "input": f"Extract events for this specific person/vehicle:\nPerson_ID: 14570071_1_1\nUnit_ID: 14570071_1\nVIN: 3FAFP37352R227989\nLicense_Plate: 502545D\nVehicle_Details: 2009 Black Ford Focus\n\nCrash Narrative: {narrative_1}",
            "output": {
                "crash_id": "14570071",
                "events": [
                    {
                        "agent_id": "14570071_1_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2009 Black Ford Focus",
                        "action": "veered from lane onto left shoulder",
                        "conditions": ["unknown reason", "alcohol impairment"],
                        "outcome": "collided with cement barricade",
                        "causal_confidence": 0.9,
                        "mentions_alcohol_or_drugs": True,
                        "affected_entities": ["3FAFP37352R227989"],
                        "referenced_units": ["unit-01"],
                    },
                    {
                        "agent_id": "14570071_1_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2009 Black Ford Focus",
                        "action": "fled scene",
                        "conditions": ["alcohol impairment"],
                        "outcome": "vehicle became disabled on right shoulder",
                        "causal_confidence": 0.8,
                        "mentions_alcohol_or_drugs": True,
                        "affected_entities": ["3FAFP37352R227989"],
                        "referenced_units": ["unit-01"],
                    },
                ],
            },
        }
    )

    # Example 2: Multi-vehicle crash - focus on Unit 1 driver
    narrative_2 = (
        "A witnessed observed Unit 1 traveling in the middle lane, NB along 5200 Sam Houston Parkway, and disregarded a red light "
        "and crashing into Unit 2 traveling in the middle WB along 1700 Genoa Red Bluff Rd. After the initial crash, Unit 2 spun out "
        "causing a secondary crash into Unit 3 traveling in the outside lane WB along 1700 Genoa Red Bluff Rd. See Pasadena PD Case#16-5336 "
        "in reference to the possible contributing factors for this case being Unit 1 driving under the influence."
    )
    examples.append(
        {
            "input": f"Extract events for this specific person/vehicle:\nPerson_ID: 14963691_1_1\nUnit_ID: 14963691_1\nVIN: 1GNEC13T21R116524\nLicense_Plate: FPS2038\nVehicle_Details: 2001 Gray Chevrolet Tahoe\n\nCrash Narrative: {narrative_2}",
            "output": {
                "crash_id": "14963691",
                "events": [
                    {
                        "agent_id": "14963691_1_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2001 Gray Chevrolet Tahoe",
                        "action": "disregarded red light",
                        "conditions": ["intersection", "alcohol impairment"],
                        "outcome": "crashed into Unit 2",
                        "causal_confidence": 0.9,
                        "mentions_alcohol_or_drugs": True,
                        "affected_entities": ["Unit 2"],
                        "referenced_units": ["Unit 1", "Unit 2"],
                    },
                ],
            },
        }
    )

    # Example 3: Person was affected by others' actions (victim)
    narrative_3 = (
        "Unit 1 disregarded red light and crashed into Unit 2. Unit 2 spun out causing secondary crash into Unit 3. "
        "Unit 2 driver was not at fault."
    )
    examples.append(
        {
            "input": f"Extract events for this specific person/vehicle:\nPerson_ID: 14963691_2_1\nUnit_ID: 14963691_2\nVIN: KM8JU3AC4DU771204\nLicense_Plate: CNC0346\nVehicle_Details: 2013 Silver Hyundai Tucson\n\nCrash Narrative: {narrative_3}",
            "output": {
                "crash_id": "14963691",
                "events": [
                    {
                        "agent_id": "14963691_2_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2013 Silver Hyundai Tucson",
                        "action": "was struck by Unit 1",
                        "conditions": ["red light violation by Unit 1"],
                        "outcome": "spun out and caused secondary crash into Unit 3",
                        "causal_confidence": 0.8,
                        "mentions_alcohol_or_drugs": None,
                        "affected_entities": ["Unit 3"],
                        "referenced_units": ["Unit 1", "Unit 2", "Unit 3"],
                    },
                ],
            },
        }
    )

    # Example 4: Person with both causal and affected events
    narrative_4 = (
        "Unit 1 was speeding and lost control, veering into Unit 2. Unit 2 was pushed into the guardrail. "
        "Unit 1 then struck a tree. Both drivers were injured."
    )
    examples.append(
        {
            "input": f"Extract events for this specific person/vehicle:\nPerson_ID: 15515787_1_1\nUnit_ID: 15515787_1\nVIN: 2HN4D18292H509709\nLicense_Plate: HC87348\nVehicle_Details: 2002 Gray Acura MDX\n\nCrash Narrative: {narrative_4}",
            "output": {
                "crash_id": "15515787",
                "events": [
                    {
                        "agent_id": "15515787_1_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2002 Gray Acura MDX",
                        "action": "was speeding and lost control",
                        "conditions": ["excessive speed"],
                        "outcome": "veered into Unit 2",
                        "causal_confidence": 0.9,
                        "mentions_alcohol_or_drugs": None,
                        "affected_entities": ["Unit 2"],
                        "referenced_units": ["Unit 1", "Unit 2"],
                    },
                    {
                        "agent_id": "15515787_1_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2002 Gray Acura MDX",
                        "action": "struck tree",
                        "conditions": ["loss of control"],
                        "outcome": "driver injured",
                        "causal_confidence": 0.8,
                        "mentions_alcohol_or_drugs": None,
                        "affected_entities": ["tree"],
                        "referenced_units": ["Unit 1"],
                    },
                ],
            },
        }
    )

    # Example 5: Clear victim case - Unit 2 was struck by Unit 1
    narrative_5 = (
        "UNIT-2 WAS TRAVELING NORTH ON HORIZON BLVD. UNIT-1 WAS FACING SOUTH ATTEMPTING TO MAKE A U-TURN. "
        "DRIVER OF UNIT-2 WAS STOPPED BEHIND UNIT-1 WAITING WHEN DRIVER OF UNIT-1 CONDUCTED AN UNSAFE BACKING "
        "AND COLLIDED WITH THE FRONT OF UNIT-2 CAUSING DAMAGE. UNIT-1 FLED THE SCENE AND WAS FOLLOWED BY UNIT-2."
    )
    examples.append(
        {
            "input": f"Extract ALL events for this specific person/vehicle (both causal and affected):\nPerson_ID: 14881309_2_1\nUnit_ID: 14881309_2\nVIN: KMHHT6KD5DU083092\nLicense_Plate: FVM1847\nVehicle_Details: 2013 Blue Hyundai Genesis\n\nCrash Narrative: {narrative_5}",
            "output": {
                "crash_id": "14881309",
                "events": [
                    {
                        "agent_id": "14881309_2_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2013 Blue Hyundai Genesis",
                        "action": "was traveling north and stopped behind Unit-1",
                        "conditions": ["waiting for Unit-1 to correct"],
                        "outcome": "was struck by Unit-1's unsafe backing",
                        "causal_confidence": 0.9,
                        "mentions_alcohol_or_drugs": None,
                        "affected_entities": ["Unit-1"],
                        "referenced_units": ["UNIT-2", "UNIT-1"],
                    },
                    {
                        "agent_id": "14881309_2_1",
                        "agent_type": "Driver",
                        "agent_description": "Driver of 2013 Blue Hyundai Genesis",
                        "action": "followed Unit-1",
                        "conditions": ["Unit-1 fled the scene"],
                        "outcome": "gathered information for accident report",
                        "causal_confidence": 0.8,
                        "mentions_alcohol_or_drugs": None,
                        "affected_entities": ["Unit-1"],
                        "referenced_units": ["UNIT-2", "UNIT-1"],
                    },
                ],
            },
        }
    )

    return examples


def build_prompt() -> ChatPromptTemplate:
    examples = build_few_shots()

    messages = [
        ("system", SYSTEM_INSTRUCTIONS),
        (
            "system",
            "You will be provided a police crash narrative. Extract one or more causal events. "
            "Conform exactly to the provided schema. If nothing causal is present, return an empty list for events.",
        ),
    ]

    # Add few-shots (escape braces in JSON so template engine doesn't treat them as vars)
    for ex in examples:
        messages.append(("human", ex["input"]))
        ai_json = json.dumps(ex["output"], ensure_ascii=False)
        ai_json_escaped = ai_json.replace("{", "{{").replace("}", "}}")
        messages.append(("ai", ai_json_escaped))

    # Runtime input with entity context - focus on specific person/vehicle
    messages.append(("human", "Extract ALL events for this specific person/vehicle (both causal and affected):\nPerson_ID: {person_id}\nUnit_ID: {unit_id}\nVIN: {vin}\nLicense_Plate: {license_plate}\nVehicle_Details: {vehicle_details}\n\nCrash Narrative: {narrative}"))

    return ChatPromptTemplate.from_messages(messages)


def get_llm(model: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass(
            "Enter GOOGLE_API_KEY (create at https://aistudio.google.com/app/apikey): "
        ).strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required to run extraction.")
        os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model, temperature=0.2)


def extract_events_for_rows(df: pd.DataFrame, limit: int) -> List[Extraction]:
    prompt = build_prompt()
    llm = get_llm()
    structured = llm.with_structured_output(Extraction)
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

        chain = prompt | structured
        result: Extraction = chain.invoke({
            "crash_id": crash_id,
            "person_id": person_id,
            "unit_id": unit_id,
            "vin": vin,
            "license_plate": license_plate,
            "vehicle_details": vehicle_details,
            "narrative": narrative
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
    extractions_path = args.output or os.path.join(args.out_dir, "extractions.jsonl")
    if args.use_existing_extractions:
        if not os.path.exists(args.use_existing_extractions):
            raise FileNotFoundError(f"Extractions file not found: {args.use_existing_extractions}")
        extractions_path = args.use_existing_extractions
        logger.info(f"Skipping extraction. Using existing extractions at: {extractions_path}")
    else:
        results = extract_events_for_rows(df, args.limit)
        with open(extractions_path, "w", encoding="utf-8") as f:
            for item in tqdm(results, desc="Writing results", total=len(results)):
                f.write(item.model_dump_json() + "\n")
        logger.info(f"Wrote {len(results)} extractions to {extractions_path}")
    
    # Build Neo4j graph if requested
    if args.neo4j:
        logger.info("Building Neo4j graph...")
        try:
            from utils.neo4j_graph import Neo4jGraphBuilder
            
            uri = os.environ.get("NEO4J_URI")
            password = os.environ.get("NEO4J_PASSWORD")
            
            if not uri or not password:
                logger.error("NEO4J_URI and NEO4J_PASSWORD environment variables must be set")
                return
            
            builder = Neo4jGraphBuilder(uri, password)
            builder.build_graph_from_jsonl(extractions_path, clear_first=args.clear_neo4j)
            
            # Show graph statistics
            stats = builder.get_graph_stats()
            logger.info("Neo4j Graph Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
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


