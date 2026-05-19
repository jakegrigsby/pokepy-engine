"""Bit-packing helpers for the flat int16 battle state buffer.

 The
upstream uses jnp.where over batched arrays; pokepy's port operates on Python
ints / numpy scalars (single-battle scalar engine). The numpy-batched variant
in phase 10 will reintroduce vectorized versions sharing this layout.

All functions accept and return Python ints (or numpy int16/int32) and are
side-effect free.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Boost packing (4 bits per stat, stored as 0-15 where 6 = neutral)
# -----------------------------------------------------------------------------
#
# pokemon[13]: atk(4) | def(4)<<4 | spa(4)<<8 | spd(4)<<12
# pokemon[14]: spe(4) | acc(4)<<4 | eva(4)<<8 | tera_type(4)<<12

def extract_boost(boosts_packed: int, shift: int) -> int:
    """Extract a 4-bit signed boost value (-6..+6) from a packed int16."""
    raw = (int(boosts_packed) >> shift) & 0xF
    return raw - 6

def apply_boost_to_packed(boosts_packed: int, shift: int, delta: int) -> int:
    """Apply a stat boost delta and return the new packed int16."""
    current = ((int(boosts_packed) >> shift) & 0xF) - 6
    new_boost = max(-6, min(6, current + int(delta)))
    new_raw = new_boost + 6
    mask = ~(0xF << shift) & 0xFFFF
    val = (int(boosts_packed) & 0xFFFF & mask) | (new_raw << shift)
    if val >= 0x8000:
        val -= 0x10000
    return val

# -----------------------------------------------------------------------------
# Status packing (status: low byte, status_turns: high byte)
# -----------------------------------------------------------------------------

def get_status(pokemon_status_field: int) -> int:
    return int(pokemon_status_field) & 0xFF

def get_status_turns(pokemon_status_field: int) -> int:
    return (int(pokemon_status_field) >> 8) & 0xFF

def set_status(status: int, turns: int = 0) -> int:
    val = (int(status) & 0xFF) | ((int(turns) & 0xFF) << 8)
    if val >= 0x8000:
        val -= 0x10000
    return val

# -----------------------------------------------------------------------------
# Hazards (per-side packed int16)
# -----------------------------------------------------------------------------
#
# bits 0-1: spikes layers (0-3)
# bit 2:    stealth rock (0/1)
# bits 3-4: toxic spikes layers (0-2)
# bit 5:    sticky web (0/1)

def get_spikes_layers(hazards: int) -> int:
    return int(hazards) & 0x3

def get_stealth_rock(hazards: int) -> int:
    return (int(hazards) >> 2) & 0x1

def get_toxic_spikes_layers(hazards: int) -> int:
    return (int(hazards) >> 3) & 0x3

def get_sticky_web(hazards: int) -> int:
    return (int(hazards) >> 5) & 0x1

def set_spikes(hazards: int, layers: int) -> int:
    layers = min(int(layers), 3)
    return (int(hazards) & ~0x3) | layers

def set_stealth_rock(hazards: int) -> int:
    return int(hazards) | 0x4

def set_toxic_spikes(hazards: int, layers: int) -> int:
    layers = min(int(layers), 2)
    return (int(hazards) & ~0x18) | (layers << 3)

def set_sticky_web(hazards: int) -> int:
    return int(hazards) | 0x20

def clear_hazards(hazards: int) -> int:
    return 0

# -----------------------------------------------------------------------------
# Protect state
# -----------------------------------------------------------------------------
#
# bit 0:    protect active this turn
# bits 1-3: consecutive uses (0-7)
# bits 4-7: protect type (0-15)

def get_protect_active(protect_state: int) -> int:
    return int(protect_state) & 0x1

def get_protect_consecutive(protect_state: int) -> int:
    return (int(protect_state) >> 1) & 0x7

def get_protect_type(protect_state: int) -> int:
    return (int(protect_state) >> 4) & 0xF

def set_protect_active(protect_state: int, active: bool) -> int:
    if active:
        return int(protect_state) | 0x1
    return int(protect_state) & ~0x1

def set_protect_consecutive(protect_state: int, count: int) -> int:
    count = min(int(count), 7)
    return (int(protect_state) & 0xF1) | (count << 1)

def set_protect_type(protect_state: int, protect_type: int) -> int:
    protect_type = min(int(protect_type), 15)
    return (int(protect_state) & 0x0F) | (protect_type << 4)

def increment_protect_consecutive(protect_state: int) -> int:
    return set_protect_consecutive(protect_state, get_protect_consecutive(protect_state) + 1)

def reset_protect_state(protect_state: int) -> int:
    return 0

def clear_protect_active(protect_state: int) -> int:
    return int(protect_state) & ~0x1

# -----------------------------------------------------------------------------
# Per-turn volatile field
# -----------------------------------------------------------------------------
#
# bit  0:    flinched
# bits 1-3:  confusion turns
# bits 4-6:  taunt turns
# bits 7-9:  encore turns
# bit  10:   confusion "newly applied this turn" — when set, the next call to
#            check_confusion_self_hit returns False without consuming a PRNG
#            frame and clears the bit. Used so that Showdown's "first inline
#            confusion check" (which fires inside the same turn the volatile
#            was applied via secondary) lines up with pokepy's frame schedule
#            without double-counting. See pokepy/engine/battle_gen9.py
#            `_preroll_move_secondaries` confusion-duration path.
# bits 11-13: Heal Block turns
# bits 14-15: Throat Chop turns

def get_flinched(volatile: int) -> bool:
    return bool(int(volatile) & 0x1)

def set_flinched(volatile: int, flinched: bool) -> int:
    if flinched:
        return int(volatile) | 0x1
    return int(volatile) & ~0x1

def get_confusion_turns(volatile: int) -> int:
    return (int(volatile) >> 1) & 0x7

def set_confusion_turns(volatile: int, turns: int) -> int:
    turns = min(int(turns), 7)
    return (int(volatile) & ~0xE) | (turns << 1)

# Bit 10: "confusion newly applied this turn" — see header comment.
def get_confusion_newly_applied(volatile: int) -> bool:
    return bool(int(volatile) & 0x400)

def set_confusion_newly_applied(volatile: int, flag: bool) -> int:
    if flag:
        return int(volatile) | 0x400
    return int(volatile) & ~0x400

def get_taunt_turns(volatile: int) -> int:
    return (int(volatile) >> 4) & 0x7

def set_taunt_turns(volatile: int, turns: int) -> int:
    turns = min(int(turns), 7)
    return (int(volatile) & ~0x70) | (turns << 4)

def get_encore_turns(volatile: int) -> int:
    return (int(volatile) >> 7) & 0x7

def set_encore_turns(volatile: int, turns: int) -> int:
    turns = min(int(turns), 7)
    return (int(volatile) & ~0x380) | (turns << 7)

def get_heal_block_turns(volatile: int) -> int:
    return (int(volatile) >> 11) & 0x7

def set_heal_block_turns(volatile: int, turns: int) -> int:
    turns = min(int(turns), 7)
    return (int(volatile) & ~(0x7 << 11)) | (turns << 11)

def get_throat_chop_turns(volatile: int) -> int:
    return (int(volatile) >> 14) & 0x3

def set_throat_chop_turns(volatile: int, turns: int) -> int:
    turns = min(int(turns), 3)
    return (int(volatile) & ~(0x3 << 14)) | (turns << 14)

def clear_volatile_turn_effects(volatile: int) -> int:
    """Clear per-turn volatile effects (flinch) at end of turn."""
    return int(volatile) & ~0x1

# -----------------------------------------------------------------------------
# Stockpile state (stored in OFF_MOVES)
# -----------------------------------------------------------------------------
#
# bits 0-1: stockpile layers (0-3)
# bits 2-4: number of successful Def raises granted by Stockpile
# bits 5-7: number of successful SpD raises granted by Stockpile

def get_stockpile_layers(stockpile_state: int) -> int:
    return int(stockpile_state) & 0x3

def set_stockpile_layers(stockpile_state: int, layers: int) -> int:
    layers = min(max(int(layers), 0), 0x3)
    return (int(stockpile_state) & ~0x3) | layers

def get_stockpile_def_count(stockpile_state: int) -> int:
    return (int(stockpile_state) >> 2) & 0x7

def set_stockpile_def_count(stockpile_state: int, count: int) -> int:
    count = min(max(int(count), 0), 0x7)
    return (int(stockpile_state) & ~(0x7 << 2)) | (count << 2)

def get_stockpile_spd_count(stockpile_state: int) -> int:
    return (int(stockpile_state) >> 5) & 0x7

def set_stockpile_spd_count(stockpile_state: int, count: int) -> int:
    count = min(max(int(count), 0), 0x7)
    return (int(stockpile_state) & ~(0x7 << 5)) | (count << 5)
