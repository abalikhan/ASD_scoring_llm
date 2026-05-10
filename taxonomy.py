# ─────────────────────────────────────────────────────────────────────────────
# OFFICIAL ENGLISH ITEM NAMES (WPS ADOS-2 Manual terminology)
# These are hardcoded because clinical terms must be exact
# ─────────────────────────────────────────────────────────────────────────────

ITEM_NAMES_EN = {
    "A1": "Overall Level of Non-Echoed Language",
    "A2": "Frequency of Spontaneous Vocalization Directed to Others",
    "A3": "Intonation of Vocalizations or Verbalizations",
    "A4": "Immediate Echolalia",
    "A5": "Stereotyped/Idiosyncratic Use of Words or Phrases",
    "A6": "Use of Another's Body to Communicate",
    "A7": "Pointing",
    "A8": "Gestures",
    "B1": "Unusual Eye Contact",
    "B2": "Responsive Social Smile",
    "B3": "Facial Expressions Directed to Others",
    "B4": "Integration of Gaze and Other Behaviors During Social Overtures",
    "B5": "Shared Enjoyment in Interaction",
    "B6": "Response to Name",
    "B7": "Requesting",
    "B8": "Giving",
    "B9": "Showing",
    "B9a": "Amount of Social Overtures/Maintenance of Attention: EXAMINER",
    "B9b": "Amount of Social Overtures/Maintenance of Attention: PARENT",
    "B10": "Spontaneous Initiation of Joint Attention",
    "B11": "Response to Joint Attention",
    "B12": "Quality of Social Overtures",
    "B13a": "Amount of Social Overtures/Maintenance of Attention: EXAMINER",
    "B13b": "Amount of Social Overtures/Maintenance of Attention: PARENT",
    "B14": "Quality of Social Response",
    "B15": "Level of Engagement",
    "B16": "Overall Quality of Rapport",
    "C1": "Functional Play with Objects",
    "C2": "Imagination/Creativity",
    "D1": "Unusual Sensory Interest in Play Material/Person",
    "D2": "Hand and Finger and Other Complex Mannerisms",
    "D3": "Self-Injurious Behavior",
    "D4": "Unusually Repetitive Interests or Stereotyped Behaviors",
    "E1": "Overactivity/Agitation",
    "E2": "Tantrums, Aggression, Negative or Disruptive Behavior",
    "E3": "Anxiety",
}

# M2-specific overrides (some items have different names in M2)
ITEM_NAMES_EN_M2 = {
    "A2": "Speech Abnormalities Associated with Autism",
    "A4": "Stereotyped/Idiosyncratic Use of Words or Phrases",
    "A5": "Conversation",
    "A6": "Pointing",
    "A7": "Descriptive, Conventional, Instrumental, or Informational Gestures",
    "B2": "Facial Expressions Directed to Others",
    "B3": "Shared Enjoyment in Interaction",
    "B5": "Showing",
    "B6": "Spontaneous Initiation of Joint Attention",
    "B8": "Quality of Social Overtures",
    "B10": "Quality of Social Response",
    "B11": "Amount of Reciprocal Social Communication",
    "B12": "Overall Quality of Rapport",
}

DOMAIN_META = {
    "A": {"name_fr": "Langage et Communication",
          "name_en": "Language and Communication"},
    "B": {"name_fr": "Interaction Sociale Réciproque",
          "name_en": "Reciprocal Social Interaction"},
    "C": {"name_fr": "Jeu",
          "name_en": "Play"},
    "D": {"name_fr": "Comportements Stéréotypés et Intérêts Restreints",
          "name_en": "Stereotyped Behaviors and Restricted Interests"},
    "E": {"name_fr": "Autres Comportements Anormaux",
          "name_en": "Other Abnormal Behaviors"},
}

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_M1 = {
    "SA_items":  ["A2", "A7", "A8", "B1", "B3", "B4", "B5", "B9", "B10", "B11", "B12"],
    "CRR_items": ["A3", "A5", "D1", "D2", "D4"],
    "score_conversion": "Raw 3 → 2; Raw 7/8/9 → 0; Raw 0/1/2 unchanged",
    "cutoffs": {
        "few_or_no_words": {"autism": 16, "spectrum": 11},
        "some_words":      {"autism": 12, "spectrum": 8},
    },
}

ALGORITHM_M2 = {
    "SA_items":  ["A6", "A7", "B1", "B2", "B3", "B5", "B6", "B8", "B11", "B12"],
    "CRR_items": ["A4", "D1", "D2", "D4"],
    "score_conversion": "Raw 3 → 2; Raw 7/8/9 → 0; Raw 0/1/2 unchanged",
    "cutoffs": {
        "under_5_years":       {"autism": 10, "spectrum": 7},
        "5_years_and_older":   {"autism": 9,  "spectrum": 8},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITIES
# ─────────────────────────────────────────────────────────────────────────────

ACTIVITIES_M1 = [
    {"id": 1,  "name_en": "Free Play"},
    {"id": 2,  "name_en": "Response to Name"},
    {"id": 3,  "name_en": "Response to Joint Attention"},
    {"id": 4,  "name_en": "Bubble Play"},
    {"id": 5,  "name_en": "Anticipation of a Routine with Objects"},
    {"id": 6,  "name_en": "Responsive Social Smile"},
    {"id": 7,  "name_en": "Anticipation of a Social Routine"},
    {"id": 8,  "name_en": "Functional and Symbolic Imitation"},
    {"id": 9,  "name_en": "Birthday Party"},
    {"id": 10, "name_en": "Snack"},
]

ACTIVITIES_M2 = [
    {"id": 1,  "name_en": "Construction Task"},
    {"id": 2,  "name_en": "Response to Name"},
    {"id": 3,  "name_en": "Make-Believe Play"},
    {"id": 4,  "name_en": "Joint Interactive Play"},
    {"id": 5,  "name_en": "Conversation"},
    {"id": 6,  "name_en": "Response to Joint Attention"},
    {"id": 7,  "name_en": "Demonstration Task"},
    {"id": 8,  "name_en": "Description of a Picture"},
    {"id": 9,  "name_en": "Telling a Story from a Book"},
    {"id": 10, "name_en": "Free Play"},
    {"id": 11, "name_en": "Birthday Party"},
    {"id": 12, "name_en": "Snack"},
    {"id": 13, "name_en": "Anticipation of a Routine with Objects"},
]

# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT — BEHAVIORAL INDICATORS PER ITEM
# These are simplified observable behaviors the scoring LLM can match against
# ─────────────────────────────────────────────────────────────────────────────

BEHAVIORAL_INDICATORS = {
    "A2": {
        "observable_behaviors": [
            "vocalizations directed to examiner or parent",
            "variety of communicative contexts (requesting, commenting, protesting)",
            "frequency and consistency of directed vocalizations",
        ],
        "red_flags": ["no directed vocalizations", "vocalizations only self-directed"],
    },
    "A3": {
        "observable_behaviors": [
            "pitch variation in speech/vocalizations",
            "rhythm and rate of speech",
            "volume modulation",
        ],
        "red_flags": ["flat/monotone intonation", "mechanical speech", "unusual pitch patterns"],
    },
    "A5": {
        "observable_behaviors": [
            "repetitive phrases or word patterns",
            "echolalia (delayed)",
            "idiosyncratic word usage",
            "neologisms or pronoun reversal",
        ],
        "red_flags": ["frequent scripted language", "referring to self by name instead of 'I'"],
    },
    "A7": {
        "observable_behaviors": [
            "index finger pointing to distant objects",
            "pointing coordinated with gaze to person",
            "pointing to share interest vs only to request",
        ],
        "red_flags": ["no pointing", "pointing without gaze coordination", "only contact pointing"],
    },
    "A8": {
        "observable_behaviors": [
            "descriptive gestures (e.g. waving, nodding, shaking head)",
            "conventional gestures (thumbs up, shrugging)",
            "instrumental gestures (reaching, giving)",
            "variety and spontaneity of gestures",
        ],
        "red_flags": ["no spontaneous gestures", "only reaching/grabbing"],
    },
    "B1": {
        "observable_behaviors": [
            "eye contact during social interaction",
            "gaze modulation (varying eye contact based on context)",
            "eye contact to initiate, regulate, or end interaction",
        ],
        "red_flags": ["absent/fleeting eye contact", "staring", "gaze avoidance"],
    },
    "B3": {
        "observable_behaviors": [
            "directing facial expressions toward examiner/parent",
            "range of facial expressions (joy, surprise, frustration, curiosity)",
            "facial expressions to communicate affect",
        ],
        "red_flags": ["flat affect", "expressions only toward objects", "no directed expressions"],
    },
    "B4": {
        "observable_behaviors": [
            "combining eye contact with vocalizations during social bids",
            "combining eye contact with gestures",
            "coordinated multimodal communication",
        ],
        "red_flags": ["gaze and gestures used independently", "no integration of communication modes"],
    },
    "B5": {
        "observable_behaviors": [
            "smiling/laughing directed at examiner during shared activity",
            "pleasure shown during non-physical interactions",
            "enjoyment communicated to another person",
        ],
        "red_flags": ["pleasure only in own activities", "no directed enjoyment"],
    },
    "B9": {
        "observable_behaviors": [
            "holding up objects for examiner/parent to see",
            "placing objects in view of others",
            "showing accompanied by eye contact",
        ],
        "red_flags": ["no showing behavior", "shows without gaze coordination"],
    },
    "B10": {
        "observable_behaviors": [
            "spontaneous gaze alternation (object → person → object)",
            "three-point gaze to share interest in distant object",
            "pointing + looking at person to direct attention",
        ],
        "red_flags": ["no spontaneous joint attention initiation", "only looks at objects"],
    },
    "B11": {
        "observable_behaviors": [
            "follows examiner's point to look at target",
            "follows examiner's gaze direction alone",
            "orients to distal target when directed",
        ],
        "red_flags": ["ignores pointing", "only looks when object makes noise"],
    },
    "B12": {
        "observable_behaviors": [
            "clear social overtures to examiner/parent",
            "appropriate form and context of social bids",
            "variety of social overture types",
        ],
        "red_flags": ["overtures only for requests", "bizarre/inappropriate overtures", "no overtures"],
    },
    "D1": {
        "observable_behaviors": [
            "sniffing objects or people",
            "visual inspection of objects at unusual angles",
            "repetitive touching of textures",
            "licking or mouthing objects beyond developmental age",
        ],
        "red_flags": ["prolonged unusual sensory exploration", "sensory seeking interfering with tasks"],
    },
    "D2": {
        "observable_behaviors": [
            "hand flapping",
            "finger flicking or twisting",
            "complex body mannerisms",
            "unusual hand/finger postures",
        ],
        "red_flags": ["frequent stereotyped motor movements"],
    },
    "D4": {
        "observable_behaviors": [
            "spinning wheels or parts of objects",
            "lining up objects",
            "insistence on specific routines",
            "preoccupation with unusual objects or topics",
            "ritualistic behaviors",
        ],
        "red_flags": ["resistance to redirection", "distress when routine disrupted"],
    },
}
