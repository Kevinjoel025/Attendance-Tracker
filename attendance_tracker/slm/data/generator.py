"""
slm/data/generator.py
─────────────────────
Deterministic synthetic NLU training data generator for College Attendance Tracker SLM.

RULES:
- Deterministic output with configurable random seed (default: 42).
- Covers all 8 canonical MVP intents:
    1. GET_ATTENDANCE
    2. GET_SUBJECT_ATTENDANCE
    3. ATTENDANCE_PROJECTION
    4. CLASSES_REQUIRED_FOR_TARGET
    5. MAX_CLASSES_CAN_MISS
    6. CHECK_ATTENDANCE_UPDATE
    7. HELP
    8. UNKNOWN
- Covers all 9 canonical subjects + canonical aliases.
- Distinguishes college threshold (75%) vs personal target (70%) vs explicit targets.
- Uses template-family splitting to guarantee ZERO leakage between Train (80%), Val (10%), and Test (10%).
- Outputs pipe-delimited structured labels: INTENT|SUBJECT|TARGET|COUNT|DATE.
- NO arithmetic performed in labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)


# ── Canonical Constants ───────────────────────────────────────────────────────

CANONICAL_INTENTS = [
    "GET_ATTENDANCE",
    "GET_SUBJECT_ATTENDANCE",
    "ATTENDANCE_PROJECTION",
    "CLASSES_REQUIRED_FOR_TARGET",
    "MAX_CLASSES_CAN_MISS",
    "CHECK_ATTENDANCE_UPDATE",
    "HELP",
    "UNKNOWN",
]

# Canonical short names
SUBJECT_CANONICAL = [
    "GS",
    "OE-1",
    "DE",
    "ACD",
    "IoTAP",
    "OS",
    "DE LAB",
    "IoTAP LAB",
    "WP",
]

# Subject phrasing mappings (canonical short name -> list of aliases & surface forms)
SUBJECT_PHRASINGS: dict[str, list[str]] = {
    "GS": [
        "GS", "gs", "Gender", "gender", "Gender Sensitization", "gender sensitization",
        "Gender Sensitisation", "gender sensitisation", "GS class"
    ],
    "OE-1": [
        "OE-1", "oe-1", "OE1", "oe1", "OE", "oe", "Storytelling", "storytelling",
        "Storytelling for Effective Communication", "storytelling for effective communication",
        "storytelling communication", "Storytelling Communication"
    ],
    "DE": [
        "DE", "de", "Data", "data", "Data Engineering", "data engineering",
        "Data Eng", "data eng", "DE subject"
    ],
    "ACD": [
        "ACD", "acd", "Automata", "automata", "Automata and Compiler Design",
        "automata and compiler design", "Compiler Design", "compiler design",
        "Compiler", "compiler"
    ],
    "IoTAP": [
        "IoTAP", "iotap", "IoT", "iot", "IoT Architecture", "iot architecture",
        "IoT Architecture and Protocols", "iot architecture and protocols",
        "IOT", "IOTAP"
    ],
    "OS": [
        "OS", "os", "Operating System", "operating system",
        "Operating Systems", "operating systems", "OS class"
    ],
    "DE LAB": [
        "DE LAB", "de lab", "Data Engineering Lab", "data engineering lab",
        "Data Engineering Laboratory", "data engineering laboratory", "DE practical"
    ],
    "IoTAP LAB": [
        "IoTAP LAB", "iotap lab", "IoT Lab", "iot lab", "IoT Architecture Lab",
        "iot architecture lab", "IoT Architecture and Protocols Lab",
        "iot architecture and protocols lab", "IOT LAB"
    ],
    "WP": [
        "WP", "wp", "Web", "web", "Web Programming", "web programming",
        "Web Prog", "web prog", "Web Development", "web dev"
    ],
}

NUMBER_WORDS = {
    "1": ["1", "one", "a single"],
    "2": ["2", "two", "couple of"],
    "3": ["3", "three"],
    "4": ["4", "four"],
    "5": ["5", "five"],
    "6": ["6", "six"],
    "7": ["7", "seven"],
    "8": ["8", "eight"],
    "9": ["9", "nine"],
    "10": ["10", "ten"],
    "12": ["12", "twelve"],
    "15": ["15", "fifteen"],
    "20": ["20", "twenty"],
}

# Target percentage phrasing (canonical target -> list of surface phrasings)
TARGET_PHRASINGS: dict[str, list[str]] = {
    "70": [
        "70%", "70 percent", "70 percentage", "seventy percent", "70",
        "my target", "my goal", "my personal target", "my attendance target",
        "personal target of 70", "my 70% target", "target attendance"
    ],
    "75": [
        "75%", "75 percent", "seventy five percent", "75 percentage", "75",
        "college requirement", "college threshold", "official requirement",
        "college criteria", "mandatory 75", "college minimum", "required percentage",
        "75% attendance"
    ],
    "80": ["80%", "80 percent", "eighty percent", "80 percentage", "80"],
    "85": ["85%", "85 percent", "eighty five percent", "85 percentage", "85"],
    "90": ["90%", "90 percent", "ninety percent", "90 percentage", "90"],
    "95": ["95%", "95 percent", "ninety five percent", "95 percentage", "95"],
    "100": ["100%", "100 percent", "hundred percent", "full attendance", "100"],
}

DATE_PHRASINGS: dict[str, list[str]] = {
    "TODAY": [
        "today", "today's", "this morning", "today's class", "today's lecture",
        "for today", "after today's class", "marked today"
    ],
    "YESTERDAY": [
        "yesterday", "yesterday's", "last class", "the previous class",
        "for yesterday", "marked yesterday"
    ],
    "2026-09-19": [
        "on 2026-09-19", "on 19-09-2026", "on 19th September", "19 September", "19th Sep"
    ],
}

CONVERSATIONAL_PREFIXES = [
    "", "", "", "", "",  # Direct queries
    "hey ", "hi ", "hello ", "please ", "can you ", "could you ",
    "can you please ", "could you please ", "tell me ", "i want to know ",
    "quick question ", "assistant ", "just checking ", "do you know ",
    "would you ", "i'd like to know ", "kindly tell me ", "may i know ",
    "can someone tell me ", "bot ", "yo ", "hey assistant "
]

CONVERSATIONAL_SUFFIXES = [
    "", "", "", "", "?", "?", "?", ".",
    " please", " please?", " thanks", " right now", " for me?",
    " if possible", " quickly", " today", " asap", " currently"
]


# ── Data Record Dataclass ─────────────────────────────────────────────────────

@dataclass
class NLURecord:
    text: str
    intent: str
    subject: Optional[str]
    target: Optional[str]
    count: Optional[str]
    date: Optional[str]
    output: str
    template_family: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "subject": self.subject,
            "target": self.target,
            "count": self.count,
            "date": self.date,
            "output": self.output,
            "template_family": self.template_family,
            "split": self.split,
        }


# ── Template Families by Intent and Split ─────────────────────────────────────
# Structure: INTENT -> { "train": [...], "val": [...], "test": [...] }
# Each split has completely distinct template families to guarantee zero leakage!

TEMPLATE_DEFINITIONS: dict[str, dict[str, list[dict[str, Any]]]] = {

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GET_ATTENDANCE
    # ─────────────────────────────────────────────────────────────────────────
    "GET_ATTENDANCE": {
        "train": [
            {"family": "get_att_direct", "patterns": [
                "what is my attendance", "what's my attendance", "show my attendance",
                "check my attendance", "my attendance", "attendance status", "get attendance"
            ]},
            {"family": "get_att_percentage", "patterns": [
                "what is my overall attendance percentage", "show overall attendance",
                "what's my total attendance percentage", "give me my attendance percentage",
                "how much is my overall attendance percentage"
            ]},
            {"family": "get_att_how_much", "patterns": [
                "how much attendance do i have", "how much is my total attendance",
                "how much attendance is marked", "how much attendance have i got"
            ]},
            {"family": "get_att_standing", "patterns": [
                "where does my attendance stand", "what is my current attendance standing",
                "tell me my current attendance", "check my overall attendance status"
            ]},
            {"family": "get_att_short", "patterns": [
                "attendance", "overall attendance", "total attendance", "show attendance stats",
                "attendance summary", "view attendance"
            ]},
        ],
        "val": [
            {"family": "get_att_val_overview", "patterns": [
                "can you pull up my attendance summary", "display my overall attendance records",
                "what is my cumulative attendance across all subjects", "show my overall class attendance"
            ]},
            {"family": "get_att_val_standing", "patterns": [
                "how is my attendance looking overall", "give me a breakdown of my current attendance",
                "check where my attendance stands in total"
            ]},
        ],
        "test": [
            {"family": "get_att_test_novel", "patterns": [
                "could you fetch my complete college attendance record",
                "where do i stand with my overall college attendance",
                "let me see my total attended classes percentage",
                "i need to check my overall attendance status right now",
                "how much total attendance have i accumulated so far"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 2. GET_SUBJECT_ATTENDANCE
    # ─────────────────────────────────────────────────────────────────────────
    "GET_SUBJECT_ATTENDANCE": {
        "train": [
            {"family": "get_subj_what_is", "patterns": [
                "what is my {subject} attendance", "what's my {subject} attendance",
                "what is the attendance in {subject}", "what is my attendance in {subject}"
            ]},
            {"family": "get_subj_show_check", "patterns": [
                "show {subject} attendance", "check {subject} attendance",
                "show attendance for {subject}", "check attendance for {subject}",
                "tell me my {subject} attendance"
            ]},
            {"family": "get_subj_how_much", "patterns": [
                "how much attendance do i have in {subject}", "how much {subject} attendance do i have",
                "how much is my {subject} attendance", "how much percentage do i have in {subject}"
            ]},
            {"family": "get_subj_short", "patterns": [
                "{subject} attendance", "attendance in {subject}", "{subject} percentage",
                "attendance for {subject}", "check {subject}"
            ]},
            {"family": "get_subj_percentage", "patterns": [
                "what is my {subject} percentage", "what percentage do i have in {subject}",
                "tell me the attendance percentage of {subject}", "show my {subject} class percentage"
            ]},
        ],
        "val": [
            {"family": "get_subj_val_standing", "patterns": [
                "how am i doing in {subject} attendance", "where does my {subject} attendance currently stand",
                "pull up my attendance record for {subject}", "give me the stats for {subject} attendance"
            ]},
        ],
        "test": [
            {"family": "get_subj_test_novel", "patterns": [
                "could you inspect my attendance percentage for {subject}",
                "how many classes did i attend in {subject}",
                "i want to see how much attendance i have maintained in {subject}",
                "let me know my current standing in {subject} classes",
                "can you display the attendance figures for {subject}"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ATTENDANCE_PROJECTION
    # ─────────────────────────────────────────────────────────────────────────
    "ATTENDANCE_PROJECTION": {
        "train": [
            {"family": "proj_if_i_attend", "patterns": [
                "if i attend {count} more classes what will my attendance be",
                "if i attend {count} more {subject} classes what will my attendance be",
                "what will my {subject} attendance be if i attend {count} classes",
                "what happens if i attend {count} more {subject} classes"
            ]},
            {"family": "proj_if_i_miss", "patterns": [
                "if i skip {count} {subject} classes what will my attendance be",
                "if i miss {count} more {subject} classes what will my attendance become",
                "what will my attendance be if i bunk {count} {subject} classes",
                "what happens if i miss {count} classes in {subject}"
            ]},
            {"family": "proj_after_attending", "patterns": [
                "what will my attendance become after attending {count} {subject} classes",
                "project my attendance after attending {count} classes in {subject}",
                "what is my projected {subject} attendance after {count} classes"
            ]},
            {"family": "proj_after_missing", "patterns": [
                "what will my {subject} percentage be after missing {count} classes",
                "how much will my attendance drop if i miss {count} {subject} classes",
                "if i take leave for {count} {subject} classes what will my percentage be"
            ]},
        ],
        "val": [
            {"family": "proj_val_scenarios", "patterns": [
                "calculate my projected percentage if i attend {count} upcoming {subject} lectures",
                "suppose i skip {count} classes of {subject} what is the resulting percentage",
                "what does my {subject} attendance look like if i attend {count} more sessions"
            ]},
        ],
        "test": [
            {"family": "proj_test_novel", "patterns": [
                "assuming i present myself for {count} further {subject} classes what will the new percentage be",
                "how badly will my {subject} attendance suffer if i absent myself from {count} classes",
                "forecast my {subject} attendance if i manage to attend {count} consecutive classes",
                "estimate my attendance in {subject} after skipping {count} periods",
                "if i attend {count} additional {subject} periods what will my ratio become"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CLASSES_REQUIRED_FOR_TARGET
    # ─────────────────────────────────────────────────────────────────────────
    "CLASSES_REQUIRED_FOR_TARGET": {
        "train": [
            {"family": "req_how_many_to_reach", "patterns": [
                "how many classes do i need to reach {target}",
                "how many classes do i need to get to {target}",
                "how many {subject} classes do i need to reach {target}",
                "how many {subject} classes do i need to attend to reach {target}"
            ]},
            {"family": "req_how_many_more", "patterns": [
                "how many more classes to reach {target} in {subject}",
                "how many more {subject} classes to get {target}",
                "how many classes should i attend in {subject} to hit {target}",
                "how many classes must i attend to reach {target} in {subject}"
            ]},
            {"family": "req_target_reach", "patterns": [
                "to reach {target} in {subject} how many classes do i need",
                "how many classes needed for {target} in {subject}",
                "classes required for {target} in {subject}",
                "classes needed to reach {target} in {subject}"
            ]},
            {"family": "req_implicit_target", "patterns": [
                "how many classes do i need to reach college requirement in {subject}",
                "how many {subject} classes to reach the college threshold",
                "how many {subject} classes needed for my target",
                "how many classes do i need to reach my goal in {subject}"
            ]},
        ],
        "val": [
            {"family": "req_val_targets", "patterns": [
                "tell me the number of {subject} classes i must attend to attain {target}",
                "how many consecutive {subject} lectures do i need to achieve {target}",
                "what is the class count required to bring my {subject} attendance to {target}"
            ]},
        ],
        "test": [
            {"family": "req_test_novel", "patterns": [
                "how many more times do i have to sit in {subject} to hit {target}",
                "what is the minimum number of {subject} classes required to reach {target}",
                "how many future sessions of {subject} must i attend to satisfy {target}",
                "calculate how many {subject} classes are needed to push my percentage to {target}",
                "to fulfill the {target} criteria in {subject} how many periods must i attend"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MAX_CLASSES_CAN_MISS
    # ─────────────────────────────────────────────────────────────────────────
    "MAX_CLASSES_CAN_MISS": {
        "train": [
            {"family": "miss_how_many_can_i", "patterns": [
                "how many classes can i miss", "how many classes can i skip",
                "how many {subject} classes can i miss", "how many {subject} classes can i skip",
                "how many {subject} classes can i bunk"
            ]},
            {"family": "miss_stay_above_target", "patterns": [
                "how many classes can i miss and stay above {target}",
                "how many {subject} classes can i miss and stay above {target}",
                "how many {subject} classes can i skip to remain above {target}",
                "how many {subject} classes can i bunk without falling below {target}"
            ]},
            {"family": "miss_max_classes", "patterns": [
                "maximum classes i can miss in {subject}", "max classes i can skip in {subject}",
                "how many bunks do i have in {subject}", "how many bunks left in {subject}",
                "how many leaves can i take in {subject}"
            ]},
            {"family": "miss_safe_skips", "patterns": [
                "how many safe skips do i have in {subject}",
                "how many classes can i miss in {subject} for {target}",
                "can i skip {subject} and stay above {target}"
            ]},
        ],
        "val": [
            {"family": "miss_val_margins", "patterns": [
                "what is the maximum margin of {subject} classes i can afford to miss for {target}",
                "how many {subject} lectures can i skip before my attendance drops below {target}",
                "calculate the allowed absences for {subject} maintaining {target}"
            ]},
        ],
        "test": [
            {"family": "miss_test_novel", "patterns": [
                "how many days can i safely absent myself from {subject} without breaching {target}",
                "what is my bunk limit in {subject} while keeping above {target}",
                "how many {subject} hours can i cut and still preserve {target}",
                "how many periods of {subject} can i afford to sacrifice for {target}",
                "determine how many {subject} classes i can forgo without falling under {target}"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 6. CHECK_ATTENDANCE_UPDATE
    # ─────────────────────────────────────────────────────────────────────────
    "CHECK_ATTENDANCE_UPDATE": {
        "train": [
            {"family": "update_did_get_marked", "patterns": [
                "did today's attendance get marked", "did {date} attendance get marked",
                "did today's {subject} attendance get marked", "did {date} {subject} attendance get marked",
                "was {subject} attendance marked {date}"
            ]},
            {"family": "update_has_it_updated", "patterns": [
                "has my attendance updated", "has {subject} attendance updated",
                "has {subject} attendance been updated {date}", "has the portal updated attendance"
            ]},
            {"family": "update_is_it_marked", "patterns": [
                "is {subject} attendance marked for {date}", "is today's attendance posted",
                "check if {subject} attendance was updated", "check if {date} attendance is registered"
            ]},
            {"family": "update_status_check", "patterns": [
                "check attendance update for {subject}", "did the faculty mark {subject} attendance",
                "was attendance registered for {subject} {date}"
            ]},
        ],
        "val": [
            {"family": "update_val_portal", "patterns": [
                "verify whether {subject} attendance has been uploaded to the portal for {date}",
                "has today's {subject} lecture been entered into the attendance system",
                "confirm if {date} attendance records for {subject} are live"
            ]},
        ],
        "test": [
            {"family": "update_test_novel", "patterns": [
                "has the teacher registered our presence in {subject} {date}",
                "did the portal reflect the attendance for {date} {subject} session",
                "can you see if today's attendance has posted on vnrvjiet portal for {subject}",
                "was today's class participation logged on the server for {subject}",
                "is there any new attendance entry recorded for {subject} {date}"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 7. HELP
    # ─────────────────────────────────────────────────────────────────────────
    "HELP": {
        "train": [
            {"family": "help_direct", "patterns": [
                "help", "help me", "how does this work", "what can you do",
                "what commands are available", "show commands", "usage", "give me help",
                "i need assistance", "command list", "list of actions", "support"
            ]},
            {"family": "help_capabilities", "patterns": [
                "what questions can i ask", "how can you assist me", "show help menu",
                "list your features", "what can this attendance assistant do",
                "what are your abilities", "tell me what you can do for me",
                "show me the available tools", "what functionalities do you support"
            ]},
            {"family": "help_guide", "patterns": [
                "how do i use this tool", "give me a guide on what to ask",
                "show example queries", "what can i ask about my attendance",
                "how to check my attendance with you", "how should i ask questions",
                "give me some sample attendance queries", "instruction manual"
            ]},
            {"family": "help_faq", "patterns": [
                "what can i query here", "what are the options in this assistant",
                "how do i calculate my required classes", "how to check attendance status",
                "getting started guide", "manual for attendance tracker"
            ]},
        ],
        "val": [
            {"family": "help_val_features", "patterns": [
                "explain your capabilities for attendance tracking",
                "provide instructions on how to query my attendance",
                "what options do i have to interact with this bot",
                "display user guide for attendance assistance",
                "how can a student make use of this tool"
            ]},
        ],
        "test": [
            {"family": "help_test_novel", "patterns": [
                "what are the supported operations of this attendance tracker",
                "could you walk me through the features available in this assistant",
                "show me the prompt syntax and supported queries for attendance",
                "i am new here what kind of requests can you understand",
                "give me a brief overview of what this attendance system does",
                "summarize all available attendance queries supported by this model"
            ]},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 8. UNKNOWN (Out-of-Domain & Hard Negatives)
    # ─────────────────────────────────────────────────────────────────────────
    "UNKNOWN": {
        "train": [
            {"family": "unk_weather", "patterns": [
                "what is the weather today", "will it rain in hyderabad", "is it sunny outside",
                "what is the temperature", "weather forecast for tomorrow"
            ]},
            {"family": "unk_coding", "patterns": [
                "write a python program to sort a list", "how to implement binary search",
                "explain recursion in c", "write sql query for joins", "debug this javascript error"
            ]},
            {"family": "unk_hard_subject_non_att", "patterns": [
                "explain what operating systems do", "what is data engineering",
                "how does compiler design work", "what are iot architecture and protocols",
                "define automata and formal languages", "tell me about gender sensitization topic",
                "what is the syllabus of web programming", "notes for operating systems",
                "who is the author of data engineering textbook", "recommend a book for automata"
            ]},
            {"family": "unk_general_knowledge", "patterns": [
                "who is the prime minister of india", "who won the cricket match",
                "tell me a joke", "what is the capital of france", "how far is the moon"
            ]},
            {"family": "unk_commands", "patterns": [
                "open youtube", "play music", "set an alarm for 7am", "turn off lights",
                "order pizza", "search google for movies"
            ]},
        ],
        "val": [
            {"family": "unk_val_academic_chat", "patterns": [
                "give me study tips for semester exams", "how to prepare a resume for placements",
                "explain the difference between process and thread in os",
                "what is the time complexity of quicksort"
            ]},
        ],
        "test": [
            {"family": "unk_test_novel", "patterns": [
                "can you write an essay on artificial intelligence in education",
                "explain how a four stroke internal combustion engine functions",
                "what is the best framework for building mobile applications",
                "give me a summary of the latest space exploration mission",
                "how do i connect my wireless headphones to windows laptop"
            ]},
        ],
    },
}


# ── Generator Class ───────────────────────────────────────────────────────────

class NLUDataGenerator:
    """
    Deterministic synthetic training data generator.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def _format_output(
        self,
        intent: str,
        subject: Optional[str] = None,
        target: Optional[str] = None,
        count: Optional[str] = None,
        date_val: Optional[str] = None,
    ) -> str:
        """Format 5-position pipe-delimited output string."""
        s = subject if subject else "NONE"
        t = target if target else "NONE"
        c = count if count else "NONE"
        d = date_val if date_val else "NONE"
        return f"{intent}|{s}|{t}|{c}|{d}"

    def _render_pattern(
        self,
        pattern: str,
        subject_canonical: Optional[str] = None,
        target_canonical: Optional[str] = None,
        count_val: Optional[str] = None,
        date_canonical: Optional[str] = None,
        split: str = "train",
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Fill placeholders in a pattern with realistic surface variations.
        """
        text = pattern
        actual_subj = subject_canonical
        actual_target = target_canonical
        actual_count = count_val
        actual_date = date_canonical

        # 1. Subject placeholder
        if "{subject}" in text:
            if actual_subj is None:
                actual_subj = self.rng.choice(SUBJECT_CANONICAL)
            surface_options = SUBJECT_PHRASINGS[actual_subj]
            surface_subj = self.rng.choice(surface_options)
            text = text.replace("{subject}", surface_subj)

        # 2. Target placeholder
        if "{target}" in text:
            if actual_target is None:
                # Weighted choice: 75% (college requirement) and 70% (personal target) are most frequent
                actual_target = self.rng.choices(
                    ["75", "70", "80", "85", "90", "65", "60", "100"],
                    weights=[40, 30, 10, 10, 5, 2, 2, 1],
                    k=1,
                )[0]

            if actual_target in TARGET_PHRASINGS:
                surface_target = self.rng.choice(TARGET_PHRASINGS[actual_target])
            else:
                surface_target = f"{actual_target}%"
            text = text.replace("{target}", surface_target)

        # 3. Count placeholder
        if "{count}" in text:
            if actual_count is None:
                actual_count = self.rng.choice(list(NUMBER_WORDS.keys()))
            surface_count = self.rng.choice(NUMBER_WORDS[actual_count])
            text = text.replace("{count}", surface_count)

        # 4. Date placeholder
        if "{date}" in text:
            if actual_date is None:
                actual_date = self.rng.choices(["TODAY", "YESTERDAY", "2026-09-19"], weights=[70, 20, 10], k=1)[0]
            surface_date = self.rng.choice(DATE_PHRASINGS[actual_date])
            text = text.replace("{date}", surface_date)

        # Add conversational decorations with some probability
        if self.rng.random() < 0.4:
            prefix = self.rng.choice(CONVERSATIONAL_PREFIXES)
            suffix = self.rng.choice(CONVERSATIONAL_SUFFIXES)
            text = f"{prefix}{text}{suffix}".strip()
        else:
            # Capitalize first letter and occasionally add question mark
            if text and text[0].islower() and self.rng.random() < 0.5:
                text = text[0].upper() + text[1:]
            if not text.endswith(("?", ".", "!")) and self.rng.random() < 0.5:
                text += "?"

        return text, actual_subj, actual_target, actual_count, actual_date

    def generate_dataset(
        self,
        total_count: int = 30000,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ) -> dict[str, list[NLURecord]]:
        """
        Generate a full synthetic dataset partitioned into train, val, and test splits.
        """
        # Reset RNG with seed for strict reproducibility
        self.rng = random.Random(self.seed)

        target_train = int(total_count * train_ratio)
        target_val = int(total_count * val_ratio)
        target_test = total_count - target_train - target_val

        splits_targets = {
            "train": target_train,
            "val": target_val,
            "test": target_test,
        }

        generated_records: dict[str, list[NLURecord]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Track global texts to guarantee uniqueness across all splits
        seen_texts: set[str] = set()

        for split_name, count_needed in splits_targets.items():
            records_for_split: list[NLURecord] = []

            # Determine items per intent for this split (roughly balanced)
            base_per_intent = count_needed // len(CANONICAL_INTENTS)
            remainder = count_needed % len(CANONICAL_INTENTS)

            for i_idx, intent in enumerate(CANONICAL_INTENTS):
                intent_target = base_per_intent + (1 if i_idx < remainder else 0)
                family_list = TEMPLATE_DEFINITIONS[intent].get(split_name, [])

                if not family_list:
                    # Fallback to general patterns if split is empty
                    family_list = TEMPLATE_DEFINITIONS[intent]["train"]

                generated_for_intent = 0
                max_attempts = intent_target * 50
                attempts = 0

                while generated_for_intent < intent_target and attempts < max_attempts:
                    attempts += 1
                    family_obj = self.rng.choice(family_list)
                    family_name = family_obj["family"]
                    pattern = self.rng.choice(family_obj["patterns"])

                    # Determine appropriate slot values based on intent
                    subj_val: Optional[str] = None
                    target_val: Optional[str] = None
                    count_val: Optional[str] = None
                    date_val: Optional[str] = None

                    if intent == "GET_SUBJECT_ATTENDANCE":
                        subj_val = self.rng.choice(SUBJECT_CANONICAL)
                    elif intent == "ATTENDANCE_PROJECTION":
                        subj_val = self.rng.choice(SUBJECT_CANONICAL) if self.rng.random() < 0.85 else None
                        count_val = self.rng.choice(list(NUMBER_WORDS.keys()))
                    elif intent == "CLASSES_REQUIRED_FOR_TARGET":
                        subj_val = self.rng.choice(SUBJECT_CANONICAL) if self.rng.random() < 0.85 else None
                        target_val = self.rng.choice(list(TARGET_PHRASINGS.keys()))
                    elif intent == "MAX_CLASSES_CAN_MISS":
                        subj_val = self.rng.choice(SUBJECT_CANONICAL) if self.rng.random() < 0.85 else None
                        target_val = self.rng.choice(list(TARGET_PHRASINGS.keys()))
                    elif intent == "CHECK_ATTENDANCE_UPDATE":
                        subj_val = self.rng.choice(SUBJECT_CANONICAL) if self.rng.random() < 0.80 else None
                        date_val = self.rng.choice(["TODAY", "YESTERDAY", "2026-09-19"])

                    rendered_text, s, t, c, d = self._render_pattern(
                        pattern=pattern,
                        subject_canonical=subj_val,
                        target_canonical=target_val,
                        count_val=count_val,
                        date_canonical=date_val,
                        split=split_name,
                    )

                    # Ensure text normalization for deduplication
                    norm_text = re.sub(r"\s+", " ", rendered_text.strip().lower())
                    if norm_text in seen_texts:
                        continue

                    seen_texts.add(norm_text)

                    # Output label construction
                    output_str = self._format_output(
                        intent=intent,
                        subject=s,
                        target=t,
                        count=c,
                        date_val=d,
                    )

                    record = NLURecord(
                        text=rendered_text,
                        intent=intent,
                        subject=s,
                        target=t,
                        count=c,
                        date=d,
                        output=output_str,
                        template_family=family_name,
                        split=split_name,
                    )
                    records_for_split.append(record)
                    generated_for_intent += 1

            # Shuffle split records deterministically
            self.rng.shuffle(records_for_split)
            generated_records[split_name] = records_for_split

        return generated_records

    def save_splits(
        self,
        splits: dict[str, list[NLURecord]],
        output_dir: Path | str | None = None,
    ) -> dict[str, Path]:
        """
        Save splits as JSONL files.
        """
        if output_dir is None:
            output_dir = Path("data/nlu")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        for split_name, records in splits.items():
            filepath = out_path / f"{split_name}.jsonl"
            with filepath.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            saved_files[split_name] = filepath
            logger.info(f"Saved {len(records)} records to {filepath}")

        return saved_files


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NLU training data.")
    parser.add_argument("--count", type=int, default=30000, help="Total number of examples to generate (default: 30000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data/nlu", help="Output directory for JSONL files")
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic NLU examples (seed={args.seed})...")
    generator = NLUDataGenerator(seed=args.seed)
    splits = generator.generate_dataset(total_count=args.count)

    saved_files = generator.save_splits(splits, output_dir=args.output_dir)
    print(f"Generation complete. Saved files:")
    for split_name, filepath in saved_files.items():
        print(f"  - {split_name}: {len(splits[split_name])} examples -> {filepath}")


if __name__ == "__main__":
    main()
