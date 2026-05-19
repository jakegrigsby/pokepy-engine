"""
Move effects data for Pokemon battle simulation.

This module contains the effect data for all moves, including:
- Status application
- Stat changes
- Secondary effects (flinch, burn chance, etc.)
- Special move categories (hazards, weather, protect, recovery, etc.)
"""

import numpy as np
from typing import Dict, Tuple

# =============================================================================
# Effect Type Constants
# =============================================================================

# Primary effect types
EFFECT_NONE = 0
EFFECT_DAMAGE = 1
EFFECT_STATUS = 2
EFFECT_STAT_CHANGE = 3
EFFECT_HAZARD = 4
EFFECT_WEATHER = 5
EFFECT_TERRAIN = 6
EFFECT_PROTECT = 7
EFFECT_RECOVERY = 8
EFFECT_SWITCH = 9  # U-turn, Volt Switch, etc.
EFFECT_RECOIL = 10
EFFECT_DRAIN = 11
EFFECT_MULTI_HIT = 12
EFFECT_OHKO = 13  # Fissure, Sheer Cold, etc.
EFFECT_TRICK_ROOM = 14  # Toggle Trick Room (reverses speed order)
EFFECT_KNOCK_OFF = 15   # Remove target's item, +50% damage if they have one
EFFECT_RAPID_SPIN = 16  # Remove hazards from user's side
EFFECT_DEFOG = 17       # Remove hazards from both sides
EFFECT_LEECH_SEED = 18  # Seed target, drain 1/8 max HP per turn
EFFECT_SUBSTITUTE = 19  # Create substitute (1/4 max HP)
EFFECT_DISABLE = 20     # Disable target's last used move
EFFECT_HAZE = 21        # Reset all stat changes (both sides)
EFFECT_CLEAR_SMOG = 22  # Reset target's stat changes + deal damage
EFFECT_PSYCH_UP = 23    # Copy target's stat changes
EFFECT_SCREEN_BREAK = 24 # Removes target's screens after dealing damage

# Status constants (match multi_format_fast_env.py)
STATUS_NONE = 0
STATUS_BURN = 1
STATUS_PARALYSIS = 2
STATUS_SLEEP = 3
STATUS_FREEZE = 4
STATUS_POISON = 5
STATUS_TOXIC = 6

# Volatile status (applied separately, can stack)
VOLATILE_CONFUSION = 1
VOLATILE_FLINCH = 2
VOLATILE_TAUNT = 3
VOLATILE_ENCORE = 4
VOLATILE_LEECH_SEED = 5
VOLATILE_PROTECT = 6
VOLATILE_FOCUS_ENERGY = 7
VOLATILE_TORMENT = 8
VOLATILE_ATTRACT = 9
VOLATILE_YAWN = 10
VOLATILE_PERISH_SONG = 11
VOLATILE_DESTINY_BOND = 12
VOLATILE_EMBARGO = 13
VOLATILE_HEAL_BLOCK = 14
VOLATILE_IMPRISON = 15
VOLATILE_INGRAIN = 16
VOLATILE_AQUA_RING = 17
VOLATILE_CURSE = 18
VOLATILE_MEAN_LOOK = 19  # Trapping
VOLATILE_NIGHTMARE = 20
VOLATILE_LOCK_ON = 21
VOLATILE_PARTIAL_TRAP = 22  # Binding moves (Bind, Wrap, Fire Spin, etc.)
VOLATILE_SALT_CURE = 23     # Salt Cure effect
VOLATILE_FORESIGHT = 24     # Foresight/Odor Sleuth - removes Ghost immunity to Normal/Fighting

# Weather constants
WEATHER_NONE = 0
WEATHER_SUN = 1
WEATHER_RAIN = 2
WEATHER_SAND = 3
WEATHER_SNOW = 4  # Gen 9 changed Hail to Snow

# Terrain constants
TERRAIN_NONE = 0
TERRAIN_ELECTRIC = 1
TERRAIN_GRASSY = 2
TERRAIN_PSYCHIC = 3
TERRAIN_MISTY = 4

# Hazard constants
HAZARD_STEALTH_ROCK = 1
HAZARD_SPIKES = 2
HAZARD_TOXIC_SPIKES = 3
HAZARD_STICKY_WEB = 4

# Stat indices for stat changes
STAT_ATK = 0
STAT_DEF = 1
STAT_SPA = 2
STAT_SPD = 3
STAT_SPE = 4
STAT_ACC = 5
STAT_EVA = 6

# =============================================================================
# Move Effect Data
# =============================================================================

# Move effects dictionary: move_name -> effect_data
# Effect data format: {
#     'effect': primary effect type,
#     'status': status to apply (if any),
#     'status_chance': chance to apply status (100 = guaranteed),
#     'stat_changes': [(stat, stages, target), ...],  # target: 0=self, 1=opponent
#     'stat_chance': chance for stat change (100 = guaranteed),
#     'volatile': volatile status to apply,
#     'volatile_chance': chance for volatile,
#     'recoil': recoil percentage (negative = drain/heal),
#     'hits': (min_hits, max_hits) for multi-hit moves,
#     'weather': weather to set,
#     'terrain': terrain to set,
#     'hazard': hazard to set,
#     'heal': heal percentage of max HP,
#     'priority': priority bracket (overrides move data if specified),
#     'flags': special flags
# }

MOVE_EFFECTS: Dict[str, dict] = {
    # ==========================================================================
    # Status Moves
    # ==========================================================================
    'willowisp': {'effect': EFFECT_STATUS, 'status': STATUS_BURN, 'status_chance': 100},
    'thunderwave': {'effect': EFFECT_STATUS, 'status': STATUS_PARALYSIS, 'status_chance': 100},
    'toxic': {'effect': EFFECT_STATUS, 'status': STATUS_TOXIC, 'status_chance': 100},
    'poisonpowder': {'effect': EFFECT_STATUS, 'status': STATUS_POISON, 'status_chance': 100},
    'sleeppowder': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'spore': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'hypnosis': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'sing': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'lovelykiss': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'darkvoid': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'stunspore': {'effect': EFFECT_STATUS, 'status': STATUS_PARALYSIS, 'status_chance': 100},
    'glare': {'effect': EFFECT_STATUS, 'status': STATUS_PARALYSIS, 'status_chance': 100},
    'nuzzle': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 100},

    # ==========================================================================
    # Stat-Changing Moves (Self)
    # ==========================================================================
    'swordsdance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 0)]},
    'nastyplot': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 2, 0)]},
    'calmmind': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 1, 0), (STAT_SPD, 1, 0)]},
    'dragondance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPE, 1, 0)]},
    'quiverdance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)]},
    'shellsmash': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 0), (STAT_SPA, 2, 0), (STAT_SPE, 2, 0), (STAT_DEF, -1, 0), (STAT_SPD, -1, 0)]},
    'bulkup': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0)]},
    'irondefense': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 2, 0)]},
    'amnesia': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, 2, 0)]},
    'agility': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, 2, 0)]},
    'rockpolish': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, 2, 0)]},
    'autotomize': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, 2, 0)]},
    'coil': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_ACC, 1, 0)]},
    'workup': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPA, 1, 0)]},
    'howl': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0)]},
    'growth': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPA, 1, 0)]},
    'tailglow': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 3, 0)]},
    'bellydrum': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 12, 0)], 'recoil': 50},  # Costs 50% HP, +12 to guarantee max
    'curse': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPE, -1, 0)]},  # Non-ghost
    'minimize': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_EVA, 2, 0)]},
    'doubleteam': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_EVA, 1, 0)]},
    'harden': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0)]},
    'withdraw': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0)]},
    'barrier': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 2, 0)]},
    'acidarmor': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 2, 0)]},
    'meditate': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0)]},
    'sharpen': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0)]},
    'defensecurl': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0)]},
    'filletaway': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 0), (STAT_SPA, 2, 0), (STAT_SPE, 2, 0)], 'recoil': 50},  # Costs 50% HP
    'honeclaws': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_ACC, 1, 0)]},
    'noretreat': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)]},
    'shelter': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 2, 0)]},
    'shiftgear': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, 2, 0), (STAT_ATK, 1, 0)]},
    'victorydance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPE, 1, 0)]},
    'geomancy': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 2, 0), (STAT_SPD, 2, 0), (STAT_SPE, 2, 0)], 'flags': ['charge']},  # Needs charge turn
    'stockpile': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0), (STAT_SPD, 1, 0)]},
    'cosmicpower': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0), (STAT_SPD, 1, 0)]},
    'cottonguard': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 3, 0)]},
    'charge': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, 1, 0)]},  # Also doubles next Electric move
    'clangoroussoul': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)], 'recoil': 33},  # Costs 1/3 HP
    'defendorder': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 1, 0), (STAT_SPD, 1, 0)]},
    'attackorder': {'effect': EFFECT_DAMAGE},  # High crit rate
    'healorder': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'lusterpurge': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 50},
    'mistball': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 50},

    # ==========================================================================
    # Stat-Lowering Moves (Opponent)
    # ==========================================================================
    'screech': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -2, 1)]},
    'faketears': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, -2, 1)]},
    'metalsound': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, -2, 1)]},
    'charm': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -2, 1)]},
    'featherdance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -2, 1)]},
    'growl': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1)]},
    'leer': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -1, 1)]},
    'tailwhip': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -1, 1)]},
    'stringshot': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -2, 1)]},
    'cottonspore': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -2, 1)]},
    'scaryface': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -2, 1)]},
    'sweetscent': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_EVA, -2, 1)]},
    'flash': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ACC, -1, 1)]},
    'sandattack': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ACC, -1, 1)]},
    'smokescreen': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ACC, -1, 1)]},
    'kinesis': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ACC, -1, 1)]},
    'eerieimpulse': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, -2, 1)]},
    'nobleroar': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'playnice': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1)]},
    'tarshot': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -1, 1)]},
    'tearfullook': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'tickle': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_DEF, -1, 1)]},
    'toxicthread': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'status': STATUS_POISON, 'status_chance': 100},
    'spicyextract': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 1), (STAT_DEF, -2, 1)]},
    'babydolleyes': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'priority': 1},
    'confide': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, -1, 1)]},
    'captivate': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, -2, 1)]},
    'partingshot': {'effect': EFFECT_SWITCH, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'venomdrench': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1), (STAT_SPE, -1, 1)]},

    # ==========================================================================
    # Damaging Moves with Stat Changes
    # ==========================================================================
    'closecombat': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 0), (STAT_SPD, -1, 0)], 'stat_chance': 100},
    'superpower': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 0), (STAT_DEF, -1, 0)], 'stat_chance': 100},
    'overheat': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -2, 0)], 'stat_chance': 100},
    'armorcannon': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 0), (STAT_SPD, -1, 0)], 'stat_chance': 100},
    'brickbreak': {'effect': EFFECT_SCREEN_BREAK},
    'psychicfangs': {'effect': EFFECT_SCREEN_BREAK},
    'ragingbull': {'effect': EFFECT_SCREEN_BREAK},
    'makeitrain': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 0)], 'stat_chance': 100},
    'headlongrush': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 0), (STAT_SPD, -1, 0)], 'stat_chance': 100},
    'icespinner': {'effect': EFFECT_DAMAGE, 'flags': ['removes_terrain']},
    'dracometeor': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -2, 0)], 'stat_chance': 100},
    'leafstorm': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -2, 0)], 'stat_chance': 100},
    'psychoboost': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -2, 0)], 'stat_chance': 100},
    'hammerarm': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 0)], 'stat_chance': 100},
    'vcreate': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 0), (STAT_SPD, -1, 0), (STAT_SPE, -1, 0)], 'stat_chance': 100},
    'fleurcannon': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -2, 0)], 'stat_chance': 100},
    'wavecrash': {'effect': EFFECT_DAMAGE, 'recoil': 33},
    'supercellslam': {'effect': EFFECT_DAMAGE, 'recoil': 50},  # 50% recoil on miss (crash)
    'highjumpkick': {'effect': EFFECT_DAMAGE, 'recoil': 50},   # 50% crash on miss
    # Self-boosting damaging moves (100% secondary)
    'flamecharge': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'aquastep': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'esperwing': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'rapidspin': {'effect': EFFECT_RAPID_SPIN, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'poweruppunch': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0)], 'stat_chance': 100},
    'chargebeam': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 70},
    'fierywrath': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    # Showdown models Alluring Voice as a guaranteed secondary callback that
    # only confuses targets whose stats were raised earlier in the same turn.
    # The turn-local gate is handled in the engine, but the move still always
    # spends the secondaryRoll PRNG frame on every successful hit.
    'alluringvoice': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'steelwing': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, 1, 0)], 'stat_chance': 10},
    'metalclaw': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0)], 'stat_chance': 10},
    # Meteor Mash: 20% +1 Atk to user (Showdown moves.ts:meteormash secondary).
    'meteormash': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0)], 'stat_chance': 20},
    'ancientpower': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)], 'stat_chance': 10},
    'silverwind': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)], 'stat_chance': 10},
    'ominouswind': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_SPA, 1, 0), (STAT_SPD, 1, 0), (STAT_SPE, 1, 0)], 'stat_chance': 10},

    # Moves that lower opponent stats
    'psychic': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'shadowball': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 20},
    'energyball': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'earthpower': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'flashcannon': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'focusblast': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'crunch': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 20},
    'acidspray': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -2, 1)], 'stat_chance': 100},
    'rocksmash': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 50},
    # 100% stat drop damaging moves
    'appleacid': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 100},
    'bittermalice': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    'breakingswipe': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    'bulldoze': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'chillingwater': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    'drumbeating': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'firelash': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 100},
    'gravapple': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 100},
    'lowsweep': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'luminacrash': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -2, 1)], 'stat_chance': 100},
    'lunge': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    'mudslap': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ACC, -1, 1)], 'stat_chance': 100},
    'octazooka': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ACC, -1, 1)], 'stat_chance': 50},
    'mysticalfire': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},
    'strugglebug': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},
    'snarl': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},
    'rocktomb': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'icywind': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'electroweb': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'glaciate': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'mirrorshot': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ACC, -1, 1)], 'stat_chance': 30},
    'muddywater': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ACC, -1, 1)], 'stat_chance': 30},
    'nightdaze': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ACC, -1, 1)], 'stat_chance': 40},
    'moonblast': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 30},
    'bugbuzz': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -1, 1)], 'stat_chance': 10},
    'playrough': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 10},
    'spiritbreak': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},
    'throatchop': {'effect': EFFECT_DAMAGE, 'flags': ['throat_chop']},  # Blocks sound moves
    'seedflare': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -2, 1)], 'stat_chance': 40},
    'crushclaw': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 50},
    'razorshell': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 50},
    'shadowbone': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 20},
    'liquidation': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 20},
    'skittersmack': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},
    'tropkick': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    # Damaging confusion secondaries from Showdown's `secondary.volatileStatus`.
    'confusion': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 10},
    'psybeam': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 10},
    'signalbeam': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 10},
    'waterpulse': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 20},
    'dizzypunch': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 20},
    'rockclimb': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 20},
    'strangesteam': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 20},
    'twinbeam': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'ironhead': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'airslash': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'chatter': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'hurricane': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 30},
    'darkpulse': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'zenheadbutt': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'rockslide': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'waterfall': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'fakeout': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 100, 'priority': 3},

    # ==========================================================================
    # Damaging Moves with Status Effects
    # ==========================================================================
    'flamethrower': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'fireblast': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'flareblitz': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10, 'recoil': 33},
    'lavaplume': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30},
    'scald': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30},
    'scorchingsands': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30},
    'steameruption': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30},
    'sacredfire': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 50},
    'blueflare': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 20},
    'heatwave': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'blazekick': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'firefang': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'icefang': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'thunderfang': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 10, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'firepunch': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'ember': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'flamewheel': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},
    'pyroball': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 10},

    'thunderbolt': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 10},
    'thunder': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'discharge': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'thunderpunch': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 10},
    'spark': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'thundershock': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 10},
    'boltstrike': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 20},
    'bodyslam': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'lick': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'forcepalm': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'zapcannon': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 100},  # 100% paralysis on hit

    'icebeam': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10},
    'blizzard': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10},
    'icepunch': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10},
    'freezedry': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10},
    'freezingglare': {'effect': EFFECT_DAMAGE, 'status': STATUS_FREEZE, 'status_chance': 10},
    'iciclespear': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},

    'poisonjab': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 30},
    'sludgebomb': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 30},
    'sludgewave': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 10},
    'gunkshot': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 30},
    'crosspoison': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 10},
    'poisonsting': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 30},
    'sludge': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 30},
    'malignantchain': {'effect': EFFECT_DAMAGE, 'status': STATUS_TOXIC, 'status_chance': 50},

    # ==========================================================================
    # Entry Hazards
    # ==========================================================================
    'stealthrock': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_STEALTH_ROCK},
    'spikes': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_SPIKES},
    'toxicspikes': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_TOXIC_SPIKES},
    'stickyweb': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_STICKY_WEB},

    # Hazard removal
    'rapidspin': {'effect': EFFECT_RAPID_SPIN, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'defog': {'effect': EFFECT_DEFOG, 'stat_changes': [(STAT_EVA, -1, 1)]},
    'mortalspin': {'effect': EFFECT_RAPID_SPIN, 'status': STATUS_POISON, 'status_chance': 100},
    'tidyup': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPE, 1, 0)], 'flags': ['removes_hazards', 'removes_substitute']},

    # ==========================================================================
    # Weather
    # ==========================================================================
    'sunnyday': {'effect': EFFECT_WEATHER, 'weather': WEATHER_SUN},
    'raindance': {'effect': EFFECT_WEATHER, 'weather': WEATHER_RAIN},
    'sandstorm': {'effect': EFFECT_WEATHER, 'weather': WEATHER_SAND},
    'snowscape': {'effect': EFFECT_WEATHER, 'weather': WEATHER_SNOW},
    'hail': {'effect': EFFECT_WEATHER, 'weather': WEATHER_SNOW},  # Gen 9 hail = snow

    # ==========================================================================
    # Terrain
    # ==========================================================================
    'electricterrain': {'effect': EFFECT_TERRAIN, 'terrain': TERRAIN_ELECTRIC},
    'grassyterrain': {'effect': EFFECT_TERRAIN, 'terrain': TERRAIN_GRASSY},
    'psychicterrain': {'effect': EFFECT_TERRAIN, 'terrain': TERRAIN_PSYCHIC},
    'mistyterrain': {'effect': EFFECT_TERRAIN, 'terrain': TERRAIN_MISTY},

    # ==========================================================================
    # Protection Moves
    # ==========================================================================
    'protect': {'effect': EFFECT_PROTECT, 'priority': 4},
    'detect': {'effect': EFFECT_PROTECT, 'priority': 4},
    'kingsshield': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_penalty']},
    'spikyshield': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_damage']},
    'banefulbunker': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_poison']},
    'silktrap': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_speed_drop']},
    'obstruct': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_def_drop']},
    'burningbulwark': {'effect': EFFECT_PROTECT, 'priority': 4, 'flags': ['contact_burn']},
    'endure': {'effect': EFFECT_PROTECT, 'priority': 4},
    'wideguard': {'effect': EFFECT_PROTECT, 'priority': 3, 'flags': ['wide_guard']},
    'quickguard': {'effect': EFFECT_PROTECT, 'priority': 3, 'flags': ['quick_guard']},

    # ==========================================================================
    # Recovery Moves
    # ==========================================================================
    'recover': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'softboiled': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'milkdrink': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'slackoff': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'roost': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['removes_flying']},
    'moonlight': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['weather_dependent']},
    'morningsun': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['weather_dependent']},
    'synthesis': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['weather_dependent']},
    'shoreup': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['sand_boost']},
    'rest': {'effect': EFFECT_RECOVERY, 'heal': 100, 'status': STATUS_SLEEP, 'status_chance': 100},  # Self-inflicts sleep
    'wish': {'effect': EFFECT_RECOVERY, 'heal': 50, 'flags': ['delayed']},
    'healbell': {'effect': EFFECT_NONE, 'flags': ['team_heal_status']},
    'aromatherapy': {'effect': EFFECT_NONE, 'flags': ['team_heal_status']},

    # Draining moves
    'drainingkiss': {'effect': EFFECT_DRAIN, 'recoil': -75},  # Negative = heal
    'gigadrain': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'bitterblade': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'drainingkiss': {'effect': EFFECT_DRAIN, 'recoil': -75},
    'matchagotcha': {'effect': EFFECT_DRAIN, 'recoil': -50, 'status': STATUS_BURN, 'status_chance': 20},
    'drainpunch': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'hornleech': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'absorb': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'megadrain': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'leechlife': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'oblivionwing': {'effect': EFFECT_DRAIN, 'recoil': -75},
    'paraboliccharge': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'dreameater': {'effect': EFFECT_DRAIN, 'recoil': -50, 'flags': ['dream_eater']},  # Only works on sleeping target

    # ==========================================================================
    # Recoil Moves
    # ==========================================================================
    'doubleedge': {'effect': EFFECT_RECOIL, 'recoil': 33},
    'bravebird': {'effect': EFFECT_RECOIL, 'recoil': 33},
    'woodhammer': {'effect': EFFECT_RECOIL, 'recoil': 33},
    'headsmash': {'effect': EFFECT_RECOIL, 'recoil': 50},
    'wildcharge': {'effect': EFFECT_RECOIL, 'recoil': 25},
    'volttackle': {'effect': EFFECT_RECOIL, 'recoil': 33, 'status': STATUS_PARALYSIS, 'status_chance': 10},
    'takedown': {'effect': EFFECT_RECOIL, 'recoil': 25},
    'submission': {'effect': EFFECT_RECOIL, 'recoil': 25},
    'headcharge': {'effect': EFFECT_RECOIL, 'recoil': 25},
    'lightofruin': {'effect': EFFECT_RECOIL, 'recoil': 50},
    'struggle': {'effect': EFFECT_RECOIL, 'recoil': 25},

    # ==========================================================================
    # Multi-Hit Moves
    # ==========================================================================
    'bulletseed': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'rockblast': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'pinmissile': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'tailslap': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'scaleshot': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5), 'stat_changes': [(STAT_SPE, 1, 0), (STAT_DEF, -1, 0)], 'stat_chance': 100},
    'bonerush': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'armthrust': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'barrage': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'cometpunch': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'furyattack': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'furyswipes': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'doubleslap': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'spikecannon': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'doublehit': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'doublekick': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'twineedle': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2), 'status': STATUS_POISON, 'status_chance': 20},
    'bonemerang': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'dualwingbeat': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'doubleironbash': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2), 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'geargrind': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'dragondarts': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'surgingstrikes': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3), 'flags': ['always_crit']},
    'tripleaxel': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3), 'flags': ['increasing_power']},
    'triplekick': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3), 'flags': ['increasing_power']},
    'tripledive': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3)},
    'populationbomb': {'effect': EFFECT_MULTI_HIT, 'hits': (1, 10)},
    # Stone Axe (id 830) and Ceaseless Edge (id 845): single-hit damaging
    # moves (BP 65) that ALSO set a hazard on the foe's side via onAfterHit.
    # Showdown source: data/moves.ts:stoneaxe / ceaselessedge — neither is
    # a multi-hit move. Pokepy used to mark them with hits 2-3 which made
    # them deal absurdly high damage. Effect type stays HAZARD so the
    # hazard setter still fires.
    'ceaselessedge': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_SPIKES},
    'stoneaxe': {'effect': EFFECT_HAZARD, 'hazard': HAZARD_STEALTH_ROCK},

    # ==========================================================================
    # Switching Moves
    # ==========================================================================
    'uturn': {'effect': EFFECT_SWITCH},
    'voltswitch': {'effect': EFFECT_SWITCH},
    'flipturn': {'effect': EFFECT_SWITCH},
    'partingshot': {'effect': EFFECT_SWITCH, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'batonpass': {'effect': EFFECT_SWITCH, 'flags': ['passes_boosts']},
    'teleport': {'effect': EFFECT_SWITCH, 'priority': -6},
    'chillyreception': {'effect': EFFECT_SWITCH, 'weather': WEATHER_SNOW},
    'shedtail': {'effect': EFFECT_SWITCH, 'flags': ['substitute']},

    # ==========================================================================
    # Priority Moves
    # ==========================================================================
    'extremespeed': {'effect': EFFECT_DAMAGE, 'priority': 2},
    'aquajet': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'bulletpunch': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'machpunch': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'iceshard': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'shadowsneak': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'quickattack': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'suckerpunch': {'effect': EFFECT_DAMAGE, 'priority': 1, 'flags': ['sucker_punch']},
    'vacuumwave': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'watershuriken': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5), 'priority': 1},
    'accelerock': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'jetpunch': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'firstimpression': {'effect': EFFECT_DAMAGE, 'priority': 2, 'flags': ['first_turn_only']},
    'grassyglide': {'effect': EFFECT_DAMAGE, 'flags': ['terrain_priority']},  # +1 priority on Grassy Terrain

    # Negative priority
    'trickroom': {'effect': EFFECT_TRICK_ROOM, 'priority': -7},
    'roar': {'effect': EFFECT_NONE, 'priority': -6, 'flags': ['phaze']},
    'whirlwind': {'effect': EFFECT_NONE, 'priority': -6, 'flags': ['phaze']},
    'dragontail': {'effect': EFFECT_DAMAGE, 'priority': -6, 'flags': ['phaze']},
    'circlethrow': {'effect': EFFECT_DAMAGE, 'priority': -6, 'flags': ['phaze']},
    'counter': {'effect': EFFECT_DAMAGE, 'priority': -5, 'flags': ['counter']},
    'mirrorcoat': {'effect': EFFECT_DAMAGE, 'priority': -5, 'flags': ['mirror_coat']},
    'metalburst': {'effect': EFFECT_DAMAGE, 'priority': 0, 'flags': ['metal_burst']},
    'avalanche': {'effect': EFFECT_DAMAGE, 'priority': -4, 'flags': ['revenge']},
    'revenge': {'effect': EFFECT_DAMAGE, 'priority': -4, 'flags': ['revenge']},
    'focuspunch': {'effect': EFFECT_DAMAGE, 'priority': -3, 'flags': ['focus_punch']},

    # ==========================================================================
    # Confusion
    # ==========================================================================
    'confuseray': {'effect': EFFECT_NONE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'sweetkiss': {'effect': EFFECT_NONE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'supersonic': {'effect': EFFECT_NONE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'teeterdance': {'effect': EFFECT_NONE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'flatter': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 1, 1)], 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'swagger': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 1)], 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'dynamicpunch': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'outrage': {'effect': EFFECT_DAMAGE, 'flags': ['locked', 'self_confuse']},
    'petaldance': {'effect': EFFECT_DAMAGE, 'flags': ['locked', 'self_confuse']},
    'thrash': {'effect': EFFECT_DAMAGE, 'flags': ['locked', 'self_confuse']},

    # ==========================================================================
    # Stat Reset Moves
    # ==========================================================================
    'haze': {'effect': EFFECT_HAZE},
    'clearsmog': {'effect': EFFECT_CLEAR_SMOG},  # Also deals damage
    'psychup': {'effect': EFFECT_PSYCH_UP},

    # ==========================================================================
    # 100% Secondary Effect Damaging Moves
    # ==========================================================================
    'pounce': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'trailblaze': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'torchsong': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 100},
    'thunderouskick': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 100},
    'mysticalpower': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 100},
    'fierydance': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 50},
    'chargebeam': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 70},
    'flamecharge': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, 1, 0)], 'stat_chance': 100},
    'acidspray': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPD, -2, 1)], 'stat_chance': 100},
    'lowsweep': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'bulldoze': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'glaciate': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'drumbeating': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'breakingswipe': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100},
    'spiritbreak': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, -1, 1)], 'stat_chance': 100},

    # ==========================================================================
    # Other Notable Moves
    # ==========================================================================
    'earthquake': {'effect': EFFECT_DAMAGE},
    'closecombat': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 0), (STAT_SPD, -1, 0)], 'stat_chance': 100},
    'knockoff': {'effect': EFFECT_KNOCK_OFF},
    'trick': {'effect': EFFECT_NONE, 'flags': ['swap_items']},
    'switcheroo': {'effect': EFFECT_NONE, 'flags': ['swap_items']},
    'taunt': {'effect': EFFECT_NONE, 'volatile': VOLATILE_TAUNT, 'volatile_chance': 100},
    'encore': {'effect': EFFECT_NONE, 'volatile': VOLATILE_ENCORE, 'volatile_chance': 100},
    'disable': {'effect': EFFECT_NONE, 'flags': ['disable']},
    'leechseed': {'effect': EFFECT_NONE, 'volatile': VOLATILE_LEECH_SEED, 'volatile_chance': 100},
    'substitute': {'effect': EFFECT_SUBSTITUTE},
    'destinybond': {'effect': EFFECT_NONE, 'volatile': VOLATILE_DESTINY_BOND, 'volatile_chance': 100},
    'perishsong': {'effect': EFFECT_NONE, 'volatile': VOLATILE_PERISH_SONG, 'volatile_chance': 100},
    # Additional volatile moves
    'focusenergy': {'effect': EFFECT_NONE, 'volatile': VOLATILE_FOCUS_ENERGY, 'volatile_chance': 100},
    'torment': {'effect': EFFECT_NONE, 'volatile': VOLATILE_TORMENT, 'volatile_chance': 100},
    'attract': {'effect': EFFECT_NONE, 'volatile': VOLATILE_ATTRACT, 'volatile_chance': 100},
    'yawn': {'effect': EFFECT_NONE, 'volatile': VOLATILE_YAWN, 'volatile_chance': 100},
    'embargo': {'effect': EFFECT_NONE, 'volatile': VOLATILE_EMBARGO, 'volatile_chance': 100},
    'healblock': {'effect': EFFECT_NONE, 'volatile': VOLATILE_HEAL_BLOCK, 'volatile_chance': 100},
    'imprison': {'effect': EFFECT_NONE, 'volatile': VOLATILE_IMPRISON, 'volatile_chance': 100},
    'ingrain': {'effect': EFFECT_NONE, 'volatile': VOLATILE_INGRAIN, 'volatile_chance': 100},
    'aquaring': {'effect': EFFECT_NONE, 'volatile': VOLATILE_AQUA_RING, 'volatile_chance': 100},
    'nightmare': {'effect': EFFECT_NONE, 'volatile': VOLATILE_NIGHTMARE, 'volatile_chance': 100},
    'lockon': {'effect': EFFECT_NONE, 'volatile': VOLATILE_LOCK_ON, 'volatile_chance': 100},
    'mindreader': {'effect': EFFECT_NONE, 'volatile': VOLATILE_LOCK_ON, 'volatile_chance': 100},
    'foresight': {'effect': EFFECT_NONE, 'volatile': VOLATILE_FORESIGHT, 'volatile_chance': 100},
    'odorsleuth': {'effect': EFFECT_NONE, 'volatile': VOLATILE_FORESIGHT, 'volatile_chance': 100},
    'meanlook': {'effect': EFFECT_NONE, 'volatile': VOLATILE_MEAN_LOOK, 'volatile_chance': 100},
    'block': {'effect': EFFECT_NONE, 'volatile': VOLATILE_MEAN_LOOK, 'volatile_chance': 100},
    'spiderweb': {'effect': EFFECT_NONE, 'volatile': VOLATILE_MEAN_LOOK, 'volatile_chance': 100},
    # Note: Ghost-type Curse is handled specially in the environment based on user type
    # Non-ghost Curse is defined above with stat changes
    'painsplit': {'effect': EFFECT_NONE, 'flags': ['pain_split']},
    'endeavor': {'effect': EFFECT_DAMAGE, 'flags': ['endeavor']},
    'superfang': {'effect': EFFECT_DAMAGE, 'flags': ['super_fang']},
    'ruination': {'effect': EFFECT_DAMAGE, 'flags': ['super_fang']},
    'seismictoss': {'effect': EFFECT_DAMAGE, 'flags': ['level_damage']},
    'nightshade': {'effect': EFFECT_DAMAGE, 'flags': ['level_damage']},
    'finalgambit': {'effect': EFFECT_DAMAGE, 'flags': ['final_gambit']},
    'explosion': {'effect': EFFECT_DAMAGE, 'flags': ['self_destruct']},
    'selfdestruct': {'effect': EFFECT_DAMAGE, 'flags': ['self_destruct']},
    'mistyexplosion': {'effect': EFFECT_DAMAGE, 'flags': ['self_destruct']},
    'memento': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -2, 1), (STAT_SPA, -2, 1)], 'flags': ['self_destruct']},
    'healingwish': {'effect': EFFECT_NONE, 'flags': ['self_destruct', 'healing_wish']},
    'lunardance': {'effect': EFFECT_NONE, 'flags': ['self_destruct', 'healing_wish']},

    # ==========================================================================
    # Binding/Trapping Moves (damage over time + prevent switching)
    # ==========================================================================
    'bind': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'wrap': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'firespin': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'whirlpool': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'sandtomb': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'magmastorm': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'infestation': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'thundercage': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'clamp': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    'snaptrap': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},

    # ==========================================================================
    # Additional Move Effects
    # ==========================================================================
    'psychicnoise': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_HEAL_BLOCK, 'volatile_chance': 100},
    'saltcure': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_SALT_CURE, 'volatile_chance': 100},
    'astonish': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'extrasensory': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'bite': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'headbutt': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    # 'darkcutter': dead key, no Showdown equivalent. Removed.
    'stomp': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'rollingkick': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'needlearm': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'heartstamp': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'snore': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'twister': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'skyattack': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30, 'flags': ['charge']},
    'hyperfang': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'boneclub': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 10},
    'dragonrush': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'fierywrath': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 20},
    'floatyfall': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'iciclecrash': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'mountaingale': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    # 'stealthystrikes': dead key, no Showdown equivalent. Removed.
    # icehammer (Crabominable signature): self -1 spe 100% via `self.boosts`.
    # Was previously typo'd as 'icyhammer' which never matched the move-id map.
    'icehammer': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 0)], 'stat_chance': 100},

    # ==========================================================================
    # Let's Go Pikachu/Eevee Exclusive Moves
    # ==========================================================================
    'buzzybuzz': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 100},
    'sizzlyslide': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 100},
    'zippyzap': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_EVA, 1, 0)], 'stat_chance': 100, 'flags': ['always_crit']},  # +1 evasion, always crits
    'splishysplash': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'bouncybubble': {'effect': EFFECT_DRAIN, 'recoil': -50},
    'freezyfrost': {'effect': EFFECT_HAZE},  # Resets all stat changes

    # Ghost moves that deal damage (ensuring they work)
    'shadowclaw': {'effect': EFFECT_DAMAGE},
    'shadowsneak': {'effect': EFFECT_DAMAGE, 'priority': 1},
    'phantomforce': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'shadowforce': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'astralbarrage': {'effect': EFFECT_DAMAGE},
    'glaciallance': {'effect': EFFECT_DAMAGE},
    'poltergeist': {'effect': EFFECT_DAMAGE},
    'hex': {'effect': EFFECT_DAMAGE},
    'lick': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'spiritshackle': {'effect': EFFECT_DAMAGE, 'flags': ['trapping']},
    'lastrespects': {'effect': EFFECT_DAMAGE},
    'bloodmoon': {'effect': EFFECT_DAMAGE},
    'infernalparade': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30},
    'behemothbash': {'effect': EFFECT_DAMAGE},

    # ==========================================================================
    # Two-Turn Moves (charge or semi-invulnerable)
    # ==========================================================================
    'dig': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'dive': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'fly': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'bounce': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable'], 'status': STATUS_PARALYSIS, 'status_chance': 30},
    'skydrop': {'effect': EFFECT_DAMAGE, 'flags': ['charge', 'semi_invulnerable']},
    'solarbeam': {'effect': EFFECT_DAMAGE, 'flags': ['charge']},
    'solarblade': {'effect': EFFECT_DAMAGE, 'flags': ['charge']},
    'meteorbeam': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 100, 'flags': ['charge']},  # +1 SpA on charge
    'skullbash': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, 1, 0)], 'stat_chance': 100, 'flags': ['charge']},  # +1 Def on charge
    'razorwind': {'effect': EFFECT_DAMAGE, 'flags': ['charge']},
    'freezeshock': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 30, 'flags': ['charge']},
    'iceburn': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 30, 'flags': ['charge']},
    'electroshot': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 100, 'flags': ['charge']},  # +1 SpA on charge

    # ==========================================================================
    # Variable-Power Moves (flags are documentation only; logic is in the engine)
    # ==========================================================================
    'gyroball':     {'effect': EFFECT_DAMAGE, 'flags': ['speed_based_power']},
    'bodypress':    {'effect': EFFECT_DAMAGE, 'flags': ['defense_based_damage']},
    'heavyslam':    {'effect': EFFECT_DAMAGE, 'flags': ['weight_based_power']},
    'heatcrash':    {'effect': EFFECT_DAMAGE, 'flags': ['weight_based_power']},
    'lowkick':      {'effect': EFFECT_DAMAGE, 'flags': ['weight_based_power']},
    'grassknot':    {'effect': EFFECT_DAMAGE, 'flags': ['weight_based_power']},
    'electroball':  {'effect': EFFECT_DAMAGE, 'flags': ['speed_based_power']},
    'weatherball':  {'effect': EFFECT_DAMAGE, 'flags': ['weather_based']},
    'terrainpulse': {'effect': EFFECT_DAMAGE, 'flags': ['terrain_based']},
    'waterspout':   {'effect': EFFECT_DAMAGE, 'flags': ['hp_scaled_power']},
    'eruption':     {'effect': EFFECT_DAMAGE, 'flags': ['hp_scaled_power']},
    'dragonenergy': {'effect': EFFECT_DAMAGE, 'flags': ['hp_scaled_power']},
    'frostbreath':  {'effect': EFFECT_DAMAGE, 'flags': ['always_crit']},
    'stormthrow':   {'effect': EFFECT_DAMAGE, 'flags': ['always_crit']},

    # Batch 4: More competitive move effects
    'spinout': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -2, 0)], 'stat_chance': 100},
    'wickedblow': {'effect': EFFECT_DAMAGE, 'flags': ['always_crit']},
    # Surging Strikes (Urshifu-RS): 3-hit always-crit (Showdown moves.ts: multihit:3, willCrit:true).
    # First entry above (line 482) is the canonical multihit one — keep flags merged.
    'surgingstrikes': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3), 'flags': ['always_crit']},
    # Batch 5: More moves
    'collisioncourse': {'effect': EFFECT_DAMAGE, 'flags': ['se_boost']},  # 1.33x when SE
    'electrodrift': {'effect': EFFECT_DAMAGE, 'flags': ['se_boost']},     # 1.33x when SE
    'lastrespects': {'effect': EFFECT_DAMAGE, 'flags': ['fainted_scaling']},  # +50 BP per fainted ally
    'ragefist': {'effect': EFFECT_DAMAGE, 'flags': ['hit_scaling']},      # +50 BP per time hit
    'lashout': {'effect': EFFECT_DAMAGE, 'flags': ['stat_lowered_boost']},  # 2x if stats lowered
    'burningjealousy': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 100, 'flags': ['conditional_burn']},
    'strengthsap': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1)], 'stat_chance': 100, 'flags': ['heal_by_target_atk']},
    'courtchange': {'effect': EFFECT_NONE, 'flags': ['swap_hazards']},
    'healingwish': {'effect': EFFECT_NONE, 'flags': ['self_destruct', 'healing_wish']},
    'lunardance': {'effect': EFFECT_NONE, 'flags': ['self_destruct', 'healing_wish']},
    # Throat Chop: Showdown moves.ts:throatchop applies a 2-turn 'throatchop'
    # condition that disables sound moves. pokepy doesn't track a sound-disable
    # volatile, and it does NOT apply heal block (the previous mapping was
    # wrong). Leave as plain damage; sound-disable is rare in OU and the
    # heal-block side effect would be strictly incorrect.
    'throatchop': {'effect': EFFECT_DAMAGE, 'flags': ['throat_chop']},
    'jawlock': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_PARTIAL_TRAP, 'volatile_chance': 100},
    # Batch 6: Dynamic BP + more moves
    'boltbeak': {'effect': EFFECT_DAMAGE, 'flags': ['doubles_if_first']},
    'fishiousrend': {'effect': EFFECT_DAMAGE, 'flags': ['doubles_if_first']},
    'assurance': {'effect': EFFECT_DAMAGE, 'flags': ['doubles_if_hit']},
    'stompingtantrum': {'effect': EFFECT_DAMAGE, 'flags': ['doubles_if_failed']},
    'punishment': {'effect': EFFECT_DAMAGE, 'flags': ['target_boost_scaling']},
    'flail': {'effect': EFFECT_DAMAGE, 'flags': ['hp_inverse_scaling']},
    'reversal': {'effect': EFFECT_DAMAGE, 'flags': ['hp_inverse_scaling']},
    'crushgrip': {'effect': EFFECT_DAMAGE, 'flags': ['target_hp_scaling']},
    'wringout': {'effect': EFFECT_DAMAGE, 'flags': ['target_hp_scaling']},
    'trumpcard': {'effect': EFFECT_DAMAGE, 'flags': ['pp_scaling']},
    # More status moves
    'spore': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'sleeppowder': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'grasswhistle': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'sing': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'hypnosis': {'effect': EFFECT_STATUS, 'status': STATUS_SLEEP, 'status_chance': 100},
    'yawn': {'effect': EFFECT_STATUS, 'volatile': VOLATILE_YAWN, 'volatile_chance': 100},
    'confuseray': {'effect': EFFECT_STATUS, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'sweetkiss': {'effect': EFFECT_STATUS, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'swagger': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 2, 1)], 'stat_chance': 100, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    'flatter': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 1, 1)], 'stat_chance': 100, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 100},
    # More recovery moves
    'slackoff': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'softboiled': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'milkdrink': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'shoreup': {'effect': EFFECT_RECOVERY, 'heal': 50},
    'rest': {'effect': EFFECT_RECOVERY, 'heal': 100, 'status': STATUS_SLEEP, 'status_chance': 100},
    # Self-boost moves
    'coil': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_DEF, 1, 0), (STAT_ACC, 1, 0)]},
    'shiftgear': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPE, 2, 0)]},
    'honeclaws': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_ACC, 1, 0)]},
    'workup': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, 1, 0), (STAT_SPA, 1, 0)]},
    # Growth already defined above (line 168)
    'rockpolish': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, 2, 0)]},
    'tailglow': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPA, 3, 0)]},
    'cottonguard': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, 3, 0)]},
    'minimize': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_EVA, 2, 0)]},
    'doubleteam': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_EVA, 1, 0)]},
    # Opponent stat drops
    'charm': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -2, 1)]},
    'featherdance': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -2, 1)]},
    'screech': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -2, 1)]},
    'metalsound': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, -2, 1)]},
    'faketears': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPD, -2, 1)]},
    'growl': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1)]},
    'leer': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -1, 1)]},
    'tailwhip': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_DEF, -1, 1)]},
    'stringshot': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -2, 1)]},
    'scaryface': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_SPE, -2, 1)]},
    'nobleroar': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'tearfullook': {'effect': EFFECT_STAT_CHANGE, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'partingshot': {'effect': EFFECT_SWITCH, 'stat_changes': [(STAT_ATK, -1, 1), (STAT_SPA, -1, 1)]},
    'populationbomb': {'effect': EFFECT_MULTI_HIT, 'hits': (1, 10)},  # 1-10 hits
    'tripledive': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3)},  # Always 3 hits
    'triplekick': {'effect': EFFECT_MULTI_HIT, 'hits': (3, 3)},  # 3 hits, escalating BP
    'watershuriken': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},  # 2-5 hits, special priority
    'bonerush': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'tailslap': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'cometpunch': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'furyswipes': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 5)},
    'doublekick': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'dualchop': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'doubleironbash': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2), 'volatile': VOLATILE_FLINCH, 'volatile_chance': 30},
    'dragondarts': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'geargrind': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'twinbeam': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'twineedle': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2), 'status': STATUS_POISON, 'status_chance': 20},
    # Batch 7: More competitive moves from deep audit
    'venoshock': {'effect': EFFECT_DAMAGE, 'flags': ['doubles_vs_poison']},
    'shellsidearm': {'effect': EFFECT_DAMAGE, 'status': STATUS_POISON, 'status_chance': 20},
    'photongeyser': {'effect': EFFECT_DAMAGE, 'flags': ['uses_higher_atk']},
    'gravapple': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_DEF, -1, 1)], 'stat_chance': 100},
    'psyblade': {'effect': EFFECT_DAMAGE, 'flags': ['terrain_boost_electric']},
    # Electro Shot: 2-turn charge move (skips charge in rain). Showdown moves.ts:electroshot.
    # Earlier duplicate (line 709) had the 'charge' flag — preserve it here so the engine
    # picks up the charge-turn behavior even though the second entry wins dict insertion.
    'electroshot': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPA, 1, 0)], 'stat_chance': 100, 'flags': ['charge']},
    'bleakwindstorm': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 30},
    'wildboltstorm': {'effect': EFFECT_DAMAGE, 'status': STATUS_PARALYSIS, 'status_chance': 20},
    'sandsearstorm': {'effect': EFFECT_DAMAGE, 'status': STATUS_BURN, 'status_chance': 20},
    'bloodmoon': {'effect': EFFECT_DAMAGE},
    'syrupbomb': {'effect': EFFECT_DAMAGE, 'stat_changes': [(STAT_SPE, -1, 1)], 'stat_chance': 100},
    'supercellslam': {'effect': EFFECT_DAMAGE, 'recoil': 50},
    'upperhand': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_FLINCH, 'volatile_chance': 100},
    'ficklebeam': {'effect': EFFECT_DAMAGE},
    # Showdown moves.ts: Tachyon Cutter is multihit:2, accuracy:true.
    'tachyoncutter': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    # Thunderclap (Raging Bolt signature): 70 BP Electric special, +1 priority,
    # fails if target isn't about to use an attacking move. The "only if target
    # attacks" gating is NOT modeled here (complex onTry logic); pokepy just
    # treats it as a +1 priority special move, which still covers the common
    # revenge-kill usage pattern. REPORT: full gating requires engine support.
    'thunderclap': {'effect': EFFECT_DAMAGE, 'priority': 1},
    # Triple Arrows (Decidueye-H signature): 90 BP Fighting, +2 crit ratio
    # (baked into move_crit_ratio.npy), 50% -1 Def + 30% flinch.
    'triplearrows': {
        'effect': EFFECT_DAMAGE,
        'stat_changes': [(STAT_DEF, -1, 1)],
        'stat_chance': 50,
        'volatile': VOLATILE_FLINCH,
        'volatile_chance': 30,
    },
    # Mighty Cleave (Kleavor signature): 95 BP Rock physical, bypasses protect
    # (no 'protect' flag in Showdown flags). REPORT: pokepy only exempts Feint
    # from protect in battle_gen9.py; Mighty Cleave needs to be added there
    # (battle_gen9.py lines ~691/738, or better via a data-driven flag).
    # Axe Kick: 30% confusion + 50% crash on miss
    'axekick': {'effect': EFFECT_DAMAGE, 'volatile': VOLATILE_CONFUSION, 'volatile_chance': 30, 'recoil': 50},
    # Life Dew (Indeedee/etc): heal 1/4 max HP
    'lifedew': {'effect': EFFECT_RECOVERY, 'heal': 25},
    # Jungle Healing (Zarude): heal 1/4 max HP + cure status
    'junglehealing': {'effect': EFFECT_RECOVERY, 'heal': 25, 'flags': ['cure_status']},
    # Heal Order
    'healorder': {'effect': EFFECT_RECOVERY, 'heal': 50},
    # Shore Up (Hippo): heal 1/2 max HP, 2/3 in sand
    'shoreup': {'effect': EFFECT_RECOVERY, 'heal': 50},
    # Bitter Malice: 75 BP Ghost special, 100% -1 Atk. Already defined above
    # with the correct effect (line ~278); do NOT override it here with burn.
    # (Showdown data/moves.ts:bittermalice — secondary: { chance: 100, boosts: { atk: -1 } })
    # Dragon Darts: 2 guaranteed hits
    'dragondarts': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    # Bonemerang: 2 guaranteed hits
    'bonemerang': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    # Double Hit / Double Kick / Twin Beam / Tera Starstorm aren't OU but list anyway
    'doublehit': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'doublekick': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    'twinbeam': {'effect': EFFECT_MULTI_HIT, 'hits': (2, 2)},
    # Surging Strikes / Triple Kick / Population Bomb already have entries
}

def get_move_effect(move_name: str) -> dict:
    """Get the effect data for a move by name."""
    return MOVE_EFFECTS.get(move_name.lower().replace(' ', '').replace('-', ''), {'effect': EFFECT_DAMAGE})

def create_move_effect_arrays(move_to_idx: Dict[str, int], num_moves: int) -> Tuple[np.ndarray, ...]:
    """Create numpy arrays for move effects indexed by move ID.

    Returns:
        move_effect_type: [num_moves] - Primary effect type
        move_status: [num_moves] - Status to apply
        move_status_chance: [num_moves] - Chance to apply status (0-100)
        move_stat_target: [num_moves] - 0=self, 1=opponent for stat changes
        move_stat_changes: [num_moves, 7] - Stat stages change for each stat
        move_stat_chance: [num_moves] - Chance to apply stat changes (0-100)
        move_volatile: [num_moves] - Volatile status to apply
        move_volatile_chance: [num_moves] - Chance for volatile
        move_recoil: [num_moves] - Recoil percentage (negative = drain)
        move_heal: [num_moves] - Heal percentage
        move_hazard: [num_moves] - Hazard type to set
        move_weather: [num_moves] - Weather to set
        move_terrain: [num_moves] - Terrain to set
    """
    move_effect_type = np.zeros(num_moves, dtype=np.int8)
    move_status = np.zeros(num_moves, dtype=np.int8)
    move_status_chance = np.zeros(num_moves, dtype=np.int8)
    move_stat_target = np.zeros(num_moves, dtype=np.int8)
    move_stat_changes = np.zeros((num_moves, 7), dtype=np.int8)  # 7 stats
    move_stat_chance = np.zeros(num_moves, dtype=np.int8)
    move_volatile = np.zeros(num_moves, dtype=np.int8)
    move_volatile_chance = np.zeros(num_moves, dtype=np.int8)
    move_recoil = np.zeros(num_moves, dtype=np.int8)
    move_heal = np.zeros(num_moves, dtype=np.int8)
    move_hazard = np.zeros(num_moves, dtype=np.int8)
    move_weather = np.zeros(num_moves, dtype=np.int8)
    move_terrain = np.zeros(num_moves, dtype=np.int8)
    move_hits_min = np.ones(num_moves, dtype=np.int8)  # Default 1 hit
    move_hits_max = np.ones(num_moves, dtype=np.int8)  # Default 1 hit

    for move_name, effect_data in MOVE_EFFECTS.items():
        if move_name in move_to_idx:
            idx = move_to_idx[move_name]

            move_effect_type[idx] = effect_data.get('effect', EFFECT_DAMAGE)
            move_status[idx] = effect_data.get('status', STATUS_NONE)
            move_status_chance[idx] = effect_data.get('status_chance', 0)
            move_volatile[idx] = effect_data.get('volatile', 0)
            move_volatile_chance[idx] = effect_data.get('volatile_chance', 0)
            move_recoil[idx] = effect_data.get('recoil', 0)
            move_heal[idx] = effect_data.get('heal', 0)
            move_hazard[idx] = effect_data.get('hazard', 0)
            move_weather[idx] = effect_data.get('weather', WEATHER_NONE)
            move_terrain[idx] = effect_data.get('terrain', TERRAIN_NONE)

            # Multi-hit moves
            hits = effect_data.get('hits', (1, 1))
            move_hits_min[idx] = hits[0]
            move_hits_max[idx] = hits[1]

            stat_changes = effect_data.get('stat_changes', [])
            if stat_changes:
                move_stat_chance[idx] = effect_data.get('stat_chance', 100)
                for stat_idx, stages, target in stat_changes:
                    move_stat_changes[idx, stat_idx] = stages
                    move_stat_target[idx] = target

    return (
        move_effect_type, move_status, move_status_chance,
        move_stat_target, move_stat_changes, move_stat_chance,
        move_volatile, move_volatile_chance,
        move_recoil, move_heal, move_hazard, move_weather, move_terrain,
        move_hits_min, move_hits_max
    )
