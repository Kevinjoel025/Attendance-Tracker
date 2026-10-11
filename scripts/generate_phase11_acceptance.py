"""
scripts/generate_phase11_acceptance.py
──────────────────────────────────────
Generates the comprehensive Phase 11 acceptance dataset (tests/data/phase11_acceptance.json)
for the current semester at VNR VJIET.

Covers all 8 intents, 9 canonical subjects, aliases, natural phrasings, word counts,
target resolution rules, noisy language, and out-of-domain fallback.
"""

import json
from pathlib import Path

ACCEPTANCE_DATA = [
    # ── 1. Overall Attendance (GET_ATTENDANCE) ───────────────────────────────
    {"query": "What is my attendance?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "What is my overall attendance?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "What's my attendance percentage?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "How much attendance do I have overall?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "Show my attendance", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "Tell me my current attendance", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "what's my attendance percentage", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "can you show me my total attendance so far?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "give me my overall attendance summary", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "what is my overall percentage across all subjects", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "check my overall attendance", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "how much is my total attendance", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "what is my attendance percentage overall?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "attendance summary pls", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "hey assistant how is my attendance overall?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "WHAT IS MY TOTAL ATTENDANCE?", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},
    {"query": "show overall attendance status", "expected_intent": "GET_ATTENDANCE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "overall_attendance"},

    # ── 2. Subject Attendance (GET_SUBJECT_ATTENDANCE) ────────────────────────
    # GS (Gender Sensitization)
    {"query": "What is my GS attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in Gender Sensitization?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Show me my attendance for gs", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "tell me my gender sensitization attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "what's my percentage in gender", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # OE-1 (Storytelling for Effective Communication)
    {"query": "What is my attendance in OE-1?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in Storytelling for Effective Communication?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my storytelling attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What's my attendance in OE1?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "show attendance in OE", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # DE (Data Engineering)
    {"query": "Tell me my DE attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in data engineering?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What's my attendance in DE?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Show Data Engineering attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "how much have i attended in data eng", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # ACD (Automata and Compiler Design)
    {"query": "How much have I attended ACD?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What is my attendance in Automata and Compiler Design?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my compiler design attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "what is my acd attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "show attendance for automata", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # IoTAP (IoT Architecture and Protocols)
    {"query": "Show IoT attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What is my IoT Architecture and Protocols attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in iotap?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my IoT attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "what's my percentage in iot architecture", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # OS (Operating Systems)
    {"query": "What is my OS attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in operating systems?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What's my attendance in OS?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my OS attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "how much have i attended in operating system", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # DE LAB (Data Engineering Laboratory)
    {"query": "What is my DE LAB attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in Data Engineering Laboratory?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my data engineering lab attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What's my attendance in DE Lab?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "show attendance in data eng lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # IoTAP LAB (IoT Architecture and Protocols Laboratory)
    {"query": "What is my IoT LAB attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in IoT Architecture and Protocols Laboratory?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my iotap lab attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What's my attendance in IoT Lab?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "show attendance in iot architecture lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    
    # WP (Web Programming)
    {"query": "What's my Web Programming attendance?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "How much attendance do I have in WP?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "Tell me my web programming attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "What is my attendance in web prog?", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},
    {"query": "show attendance for WP", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_attendance"},

    # ── 3. Attendance Projection (ATTENDANCE_PROJECTION) ───────────────────────
    {"query": "What happens if I attend the next 5 classes of OS?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "OS", "expected_target": None, "expected_count": 5, "category": "attendance_projection"},
    {"query": "If I attend 3 more classes in DE, what will my attendance be?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "DE", "expected_target": None, "expected_count": 3, "category": "attendance_projection"},
    {"query": "What will my attendance become after attending 5 classes in IoT?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "IoTAP", "expected_target": None, "expected_count": 5, "category": "attendance_projection"},
    {"query": "If I attend the next 2 classes of ACD", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "ACD", "expected_target": None, "expected_count": 2, "category": "attendance_projection"},
    {"query": "How much will my attendance increase if I attend 4 OS classes?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "OS", "expected_target": None, "expected_count": 4, "category": "attendance_projection"},
    {"query": "what happens if I attend 10 more classes in DE LAB", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "DE LAB", "expected_target": None, "expected_count": 10, "category": "attendance_projection"},
    {"query": "What happens if I attend next three classes in Operating Systems?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "OS", "expected_target": None, "expected_count": 3, "category": "attendance_projection"},
    {"query": "What will my attendance be if I attend next five classes in WP?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "WP", "expected_target": None, "expected_count": 5, "category": "attendance_projection"},
    {"query": "attend two more classes in GS what will be my percentage", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "GS", "expected_target": None, "expected_count": 2, "category": "attendance_projection"},
    {"query": "What happens if I miss the next 2 classes in OS?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "OS", "expected_target": None, "expected_count": 2, "category": "attendance_projection"},
    {"query": "If I skip 3 classes in Data Engineering, what is my projected attendance?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "DE", "expected_target": None, "expected_count": 3, "category": "attendance_projection"},
    {"query": "How will my attendance suffer if I absent myself from 1 class of ACD?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "ACD", "expected_target": None, "expected_count": 1, "category": "attendance_projection"},
    {"query": "if I attend 4 classes in Storytelling what will my attendance become?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "OE-1", "expected_target": None, "expected_count": 4, "category": "attendance_projection"},
    {"query": "What happens if I attend 6 classes in IoT Lab?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": 6, "category": "attendance_projection"},
    {"query": "What happens if I attend the next 2 classes?", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": None, "expected_target": None, "expected_count": 2, "category": "attendance_projection"},

    # ── 4. Classes Required for Target (CLASSES_REQUIRED_FOR_TARGET) ───────────
    {"query": "How many classes do I need to attend in DE to reach 75%?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "DE", "expected_target": 75.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many OS classes do I need for 80%?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "OS", "expected_target": 80.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes do I need in IoT to reach 85%?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "IoTAP", "expected_target": 85.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many more ACD classes do I need to reach my target?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "ACD", "expected_target": 70.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes do I need for my target in DE?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "DE", "expected_target": 70.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes do I need to attend in WP to reach 75 percent?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "WP", "expected_target": 75.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many more sessions of GS must I attend to hit 80%?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "GS", "expected_target": 80.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes required to reach 75% in OE-1?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "OE-1", "expected_target": 75.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes do I need in DE LAB to reach my target?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "DE LAB", "expected_target": 70.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes must I attend in IoTAP LAB for college requirement?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "IoTAP LAB", "expected_target": 75.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes do I need for my goal in ACD?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "ACD", "expected_target": 70.0, "expected_count": None, "category": "classes_required"},
    {"query": "How many classes are required in Operating Systems for 90%?", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "OS", "expected_target": 90.0, "expected_count": None, "category": "classes_required"},

    # ── 5. Maximum Classes Can Miss (MAX_CLASSES_CAN_MISS) ────────────────────
    {"query": "How many classes can I miss in OS?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "OS", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many can I bunk in DE?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "DE", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I skip in ACD?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "ACD", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "What's my bunk margin in IoT?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "IoTAP", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many more classes can I miss in WP?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "WP", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I forgo in GS without falling below 75%?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "GS", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I miss in OE-1?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "OE-1", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I miss in DE LAB?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "DE LAB", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "What is the maximum number of classes I can miss in IoT Lab?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "IoTAP LAB", "expected_target": 75.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I miss in ACD to stay above my target?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "ACD", "expected_target": 70.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I miss in OS to maintain 70%?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "OS", "expected_target": 70.0, "expected_count": None, "category": "max_can_miss"},
    {"query": "How many classes can I skip in WP without falling under 80%?", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "WP", "expected_target": 80.0, "expected_count": None, "category": "max_can_miss"},

    # ── 6. Check Attendance Update (CHECK_ATTENDANCE_UPDATE) ───────────────────
    {"query": "Did the attendance update recently?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Has my attendance been updated today?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Check if attendance was updated for OS", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Was attendance updated yesterday for DE?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Did the teacher mark attendance for ACD on 19th September?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Check attendance updates for IoT", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Has new attendance been posted for WP?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Did my attendance change recently?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Any new attendance updates?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "change_update"},
    {"query": "Was attendance updated for OE-1?", "expected_intent": "CHECK_ATTENDANCE_UPDATE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "change_update"},

    # ── 7. Help & Information (HELP) ─────────────────────────────────────────
    {"query": "Help", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "How can you help me?", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "What can you do?", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "Show available commands", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "How do I use this attendance tracker?", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "What queries are supported?", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "List features of this assistant", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},
    {"query": "What can I ask you?", "expected_intent": "HELP", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "help"},

    # ── 8. Unknown / Out-of-Domain (UNKNOWN) ──────────────────────────────────
    {"query": "Tell me a joke", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "What is the weather?", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "Who is Virat Kohli?", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "Explain quantum mechanics", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "Play music", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "asdfgh", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "hello", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "write a python script to reverse a string", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "what is the capital of France?", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},
    {"query": "how to make pasta", "expected_intent": "UNKNOWN", "expected_subject": None, "expected_target": None, "expected_count": None, "category": "unknown"},

    # ── 9. Noisy Student Language & Realistic Phrasings ───────────────────────
    {"query": "hey can you tell me my os attendance pls", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "bro how much attendance do i have in de", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "can i bunk some os classes", "expected_intent": "MAX_CLASSES_CAN_MISS", "expected_subject": "OS", "expected_target": 75.0, "expected_count": None, "category": "noisy_student_language"},
    {"query": "if i go to the next 5 iot classes what happens", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "IoTAP", "expected_target": None, "expected_count": 5, "category": "noisy_student_language"},
    {"query": "how many more de classes should i attend to get 75", "expected_intent": "CLASSES_REQUIRED_FOR_TARGET", "expected_subject": "DE", "expected_target": 75.0, "expected_count": None, "category": "noisy_student_language"},
    {"query": "what's my current attendance in operating systems??", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "pls tell me my attendance for web", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "attendance for operating system", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "OPS attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "tell me compiler attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "storytelling comms attendance", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "attendance in iotap lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "noisy_student_language"},
    {"query": "what happens if i go to 3 more data engineering lab classes", "expected_intent": "ATTENDANCE_PROJECTION", "expected_subject": "DE LAB", "expected_target": None, "expected_count": 3, "category": "noisy_student_language"},

    # ── 10. Subject Resolution Edge Cases ─────────────────────────────────────
    {"query": "operating system", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "Operating Systems", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "OS", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "os", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "data engineering", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "data eng", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "DE", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "de", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "compiler", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "compiler design", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "ACD", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "ACD", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "iot", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "IoT architecture", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "iotap", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "web", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "web programming", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "web prog", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "WP", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "gender", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "gender sensitization", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "GS", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "storytelling", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "OE", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "OE-1", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "OE-1", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "data engineering lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "DE lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "DE LAB", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "iot lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
    {"query": "IoTAP lab", "expected_intent": "GET_SUBJECT_ATTENDANCE", "expected_subject": "IoTAP LAB", "expected_target": None, "expected_count": None, "category": "subject_edge_cases"},
]


def main():
    target_path = Path("tests/data/phase11_acceptance.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(ACCEPTANCE_DATA, f, indent=2)
    print(f"Generated {len(ACCEPTANCE_DATA)} acceptance examples in {target_path}")


if __name__ == "__main__":
    main()
