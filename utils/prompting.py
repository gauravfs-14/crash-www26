from typing import List
import json
from langchain_core.prompts import ChatPromptTemplate

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
    examples = []

    narrative_1 = (
        "I-10 west at mile marker 20 is a four lane public interstate. Wi-01 stated he was traveling west on I-10 behind unit-01. "
        "Wi-01 stated as they approached I-10 west at mile marker 20, unit-01 for an unknown reason veered from the number one lane "
        "onto the left shoulder colliding into a cement barricade. Unit-01 attempted flee the scene of the accident and continued west "
        "on I-10 until the vehicle became disabled and came to a stop on the right shoulder of I-10 west. Driver of unit-01 was believed "
        "to be under the influence of alcohol and or a controlled substance, drug, dangerous drug Case # 15-213199."
    )
    examples.append(
        {
            "input": (
                "Extract events for this specific person/vehicle:\n"
                "Person_ID: 14570071_1_1\nUnit_ID: 14570071_1\nVIN: 3FAFP37352R227989\n"
                "License_Plate: 502545D\nVehicle_Details: 2009 Black Ford Focus\n\nCrash Narrative: "
                f"{narrative_1}"
            ),
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

    narrative_2 = (
        "A witnessed observed Unit 1 traveling in the middle lane, NB along 5200 Sam Houston Parkway, and disregarded a red light "
        "and crashing into Unit 2 traveling in the middle WB along 1700 Genoa Red Bluff Rd. After the initial crash, Unit 2 spun out "
        "causing a secondary crash into Unit 3 traveling in the outside lane WB along 1700 Genoa Red Bluff Rd. See Pasadena PD Case#16-5336 "
        "in reference to the possible contributing factors for this case being Unit 1 driving under the influence."
    )
    examples.append(
        {
            "input": (
                "Extract events for this specific person/vehicle:\n"
                "Person_ID: 14963691_1_1\nUnit_ID: 14963691_1\nVIN: 1GNEC13T21R116524\n"
                "License_Plate: FPS2038\nVehicle_Details: 2001 Gray Chevrolet Tahoe\n\nCrash Narrative: "
                f"{narrative_2}"
            ),
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
                    }
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

    for ex in examples:
        messages.append(("human", ex["input"]))
        ai_json = json.dumps(ex["output"], ensure_ascii=False)
        ai_json_escaped = ai_json.replace("{", "{{").replace("}", "}}")
        messages.append(("ai", ai_json_escaped))

    messages.append((
        "human",
        "Extract ALL events for this specific person/vehicle (both causal and affected):\n"
        "Person_ID: {person_id}\nUnit_ID: {unit_id}\nVIN: {vin}\n"
        "License_Plate: {license_plate}\nVehicle_Details: {vehicle_details}\n\nCrash Narrative: {narrative}"
    ))

    return ChatPromptTemplate.from_messages(messages)


def get_system_instructions() -> str:
    return SYSTEM_INSTRUCTIONS
