"""Pokemon: per-mon battle state + stat/type queries.

Port of the slice-relevant surface of sim/pokemon.ts. Stats are computed with
Showdown's exact integer formulas (Battle.statModify / calculateStat / getStat
/ getActionSpeed). Boosts, status, volatiles, item, ability follow the same
shapes the dispatch and pipeline read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pokepy.showdown.dex import to_id
from pokepy.showdown.util import trunc

if TYPE_CHECKING:
    from pokepy.showdown.battle import Battle
    from pokepy.showdown.side import Side

_STAT_TABLE = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
_BOOST_TABLE = [1, 1.5, 2, 2.5, 3, 3.5, 4]
_STAT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")


class Pokemon:
    def __init__(self, battle: "Battle", side: "Side", pset: Dict[str, Any], position: int):
        self.battle = battle
        self.side = side
        self.set = pset
        self.position = position

        self.name = pset.get("name") or pset.get("species") or ""
        self.species = battle.dex.species.get(pset.get("species") or pset.get("name"))
        self.base_species = self.species
        self.level = int(pset.get("level", 100))
        self.gender = pset.get("gender", "")
        self.happiness = int(pset.get("happiness", 255))

        # Types (live; terastallized handled in get_types).
        self.types: List[str] = list(self.species.types or ["Normal"])
        self.added_type: str = ""
        self.terastallized: str = ""
        self.tera_type: str = pset.get("teraType") or (self.types[0] if self.types else "Normal")
        self.known_type = True

        # Ability / item.
        self.ability = to_id(pset.get("ability") or self._default_ability())
        self.base_ability = self.ability
        self.ability_state: Dict[str, Any] = {"id": self.ability, "target": self}
        self.item = to_id(pset.get("item", ""))
        self.item_state: Dict[str, Any] = {"id": self.item, "target": self}
        self.last_item = ""
        self.used_item_this_turn = False

        # Status / volatiles.
        self.status = ""
        self.status_state: Dict[str, Any] = {"id": "", "target": self}
        self.volatiles: Dict[str, Dict[str, Any]] = {}
        self.species_state: Dict[str, Any] = {"id": self.species.id}

        # Boosts.
        self.boosts: Dict[str, int] = {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0,
        }

        # Stats.
        self.base_stored_stats: Dict[str, int] = battle.spread_modify(self.species.baseStats, pset)
        self.stored_stats: Dict[str, int] = {
            k: v for k, v in self.base_stored_stats.items() if k != "hp"
        }
        self.maxhp = self.base_stored_stats["hp"]
        self.baseMaxhp = self.maxhp  # Showdown name; used by status residuals etc.
        self.hp = self.maxhp
        self.speed = 0

        # Move slots.
        self.move_slots: List[Dict[str, Any]] = []
        self.base_move_slots: List[Dict[str, Any]] = []
        for mv in pset.get("moves", []):
            move = battle.dex.moves.get(mv)
            pp = move.pp if move.noPPBoosts or not move.pp else int(move.pp * 8 / 5)
            slot = {
                "move": move.name, "id": move.id, "pp": pp, "maxpp": pp,
                "target": move.target, "disabled": False, "used": False,
            }
            self.move_slots.append(slot)
            self.base_move_slots.append(dict(slot))

        # Bookkeeping flags read by the pipeline.
        self.fainted = False
        self.faint_queued = False
        self.active = False
        self.is_active = False
        self.is_started = False
        self.transformed = False
        self.illusion = None
        self.last_move = None
        self.last_move_used = None
        self.move_this_turn: Any = ""
        self.move_this_turn_result: Any = None
        self.last_damage = 0
        self.attacked_by: List[Dict[str, Any]] = []
        self.hurt_this_turn = 0
        self.new_ly = 0
        self.move_hit_data: Dict[str, Dict[str, Any]] = {}
        self.active_move_actions = 0
        self.stats: Dict[str, int] = dict(self.stored_stats)

    # ------------------------------------------------------------------ #
    @property
    def effect_type(self) -> str:
        return "Pokemon"

    def _default_ability(self) -> str:
        ab = self.species.abilities or {}
        return ab.get("0", "")

    def __str__(self) -> str:
        side_id = getattr(self.side, "id", "p?")
        return f"{side_id}a: {self.name}"

    # -- stats ---------------------------------------------------------- #
    def calculate_stat(self, stat_name: str, boost: int, modifier: int = 1, stat_user: "Pokemon" = None) -> int:
        stat_name = to_id(stat_name)
        stat = self.stored_stats[stat_name]
        # ModifyBoost event (abilities like Simple/Unaware). Data-driven path: no-op.
        b = boost
        if b > 6:
            b = 6
        if b < -6:
            b = -6
        if b >= 0:
            stat = int(stat * _BOOST_TABLE[b])
        else:
            stat = int(stat / _BOOST_TABLE[-b])
        return self.battle.modify(stat, modifier or 1)

    def get_stat(self, stat_name: str, unboosted: bool = False, unmodified: bool = False) -> int:
        stat_name = to_id(stat_name)
        stat = self.stored_stats[stat_name]
        if not unboosted:
            boost = self.boosts[stat_name]
            if boost > 6:
                boost = 6
            if boost < -6:
                boost = -6
            if boost >= 0:
                stat = int(stat * _BOOST_TABLE[boost])
            else:
                stat = int(stat / _BOOST_TABLE[-boost])
        if self.battle.gen <= 2:
            if not unmodified:
                if self.status == "par" and stat_name == "spe":
                    stat = stat // 4
                if self.status == "brn" and stat_name == "atk":
                    stat = stat // 2
            return self.battle.clamp_int_range(stat, 1, 999)
        if not unmodified:
            stat = self.battle.run_event("Modify" + _STAT_TABLE[stat_name], self, None, None, stat)
        if stat_name == "spe" and stat > 10000:
            stat = 10000
        return stat

    def get_action_speed(self) -> int:
        speed = self.get_stat("spe", False, False)
        if self.battle.field.get_pseudo_weather("trickroom"):
            speed = 10000 - speed
        return trunc(speed, 13)

    def get_weight(self) -> int:
        # Showdown stores weighthg (hectograms); JSON dump exposes weightkg.
        return max(1, trunc((self.species.weightkg or 0) * 10))

    # -- types / effectiveness ------------------------------------------ #
    def get_types(self, exclude_added: bool = False, preterastallized: bool = False) -> List[str]:
        if not preterastallized and self.terastallized and self.terastallized != "Stellar":
            return [self.terastallized]
        types = list(self.types)
        if not types:
            types = ["Normal" if self.battle.gen >= 5 else "???"]
        if not exclude_added and self.added_type:
            return types + [self.added_type]
        return types

    def has_type(self, type_) -> bool:
        my_types = self.get_types()
        if isinstance(type_, (list, tuple)):
            return any(t in my_types for t in type_)
        return type_ in my_types

    def run_effectiveness(self, move) -> int:
        total = 0
        if self.terastallized and move.type == "Stellar":
            return 1
        for t in self.get_types():
            total += self.battle.dex.get_effectiveness(move.type, t)
        return total

    def run_immunity(self, move, message=False) -> bool:
        if not move:
            return True
        mtype = move.type if not isinstance(move, str) else move
        if not isinstance(move, str):
            ii = move.ignoreImmunity
            if ii and (ii is True or (isinstance(ii, dict) and ii.get(mtype))):
                return True
        if not mtype or mtype == "???":
            return True
        if mtype == "Ground":
            return self.is_grounded() is not False
        return self.battle.dex.get_immunity(mtype, self.get_types())

    def is_grounded(self, negate_immunity: bool = False):
        if "gravity" in self.battle.field.pseudo_weather:
            return True
        if "ingrain" in self.volatiles and self.battle.gen >= 4:
            return True
        if "smackdown" in self.volatiles:
            return True
        item = "" if self.ignoring_item() else self.item
        if item == "ironball":
            return True
        if not negate_immunity and self.has_type("Flying"):
            return False
        if self.has_ability("levitate") and not self.battle.suppressing_ability(self):
            return None
        if "magnetrise" in self.volatiles:
            return False
        if "telekinesis" in self.volatiles:
            return False
        return item != "airballoon"

    # -- ability / item ------------------------------------------------- #
    def get_ability(self):
        return self.battle.dex.abilities.get(self.ability)

    def get_item(self):
        return self.battle.dex.items.get(self.item)

    def has_ability(self, ability) -> bool:
        if self.ignoring_ability():
            return False
        if isinstance(ability, (list, tuple)):
            return any(self.ability == to_id(a) for a in ability)
        return self.ability == to_id(ability)

    def has_item(self, item) -> bool:
        if self.ignoring_item():
            return False
        if isinstance(item, (list, tuple)):
            return any(self.item == to_id(i) for i in item)
        return self.item == to_id(item)

    def ignoring_item(self) -> bool:
        return bool(
            self.item == ""
            or (self.has_ability("klutz") and not (self.get_item().get("ignoreKlutz") if self.get_item() else False))
            or "embargo" in self.volatiles
            or "magicroom" in self.battle.field.pseudo_weather
        )

    def ignoring_ability(self) -> bool:
        return bool("gastroacid" in self.volatiles)

    def get_status(self):
        return self.battle.dex.conditions.get(self.status)

    # -- moves ---------------------------------------------------------- #
    def get_move_data(self, move) -> Optional[Dict[str, Any]]:
        mid = to_id(move if isinstance(move, str) else getattr(move, "id", move))
        for slot in self.move_slots:
            if slot["id"] == mid:
                return slot
        return None

    def get_move_hit_data(self, move) -> Dict[str, Any]:
        mid = move.id
        data = self.move_hit_data.get(mid)
        if data is None:
            data = {"crit": False, "typeMod": 0, "zBrokeProtect": False}
            self.move_hit_data[mid] = data
        return data

    def deduct_pp(self, move, amount=None, target=None) -> int:
        mid = to_id(getattr(move, "id", move))
        slot = self.get_move_data(mid)
        if not slot:
            return 0
        if amount is None:
            amount = 1
        had = slot["pp"]
        slot["pp"] -= amount
        if slot["pp"] < 0:
            slot["pp"] = 0
        slot["used"] = True
        return had - slot["pp"]

    def move_used(self, move, target_loc=None):
        self.last_move = move
        if self.battle.gen >= 3:
            self.last_move_used = move
        self.move_this_turn = move.id

    # -- boosts --------------------------------------------------------- #
    def boost_by(self, boosts: Dict[str, int]) -> int:
        delta = 0
        for stat, amt in boosts.items():
            before = self.boosts.get(stat, 0)
            self.boosts[stat] = max(-6, min(6, before + amt))
            delta = self.boosts[stat] - before
        return delta

    def clear_boosts(self):
        for k in self.boosts:
            self.boosts[k] = 0

    # -- hp ------------------------------------------------------------- #
    def damage(self, amount: int, source=None, effect=None) -> int:
        if not self.hp or amount <= 0:
            return 0
        amount = trunc(amount)
        amount = min(amount, self.hp)
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
        self.last_damage = amount
        return amount

    def heal(self, amount: int, source=None, effect=None) -> int:
        if not self.hp:
            return 0
        amount = trunc(amount)
        if amount <= 0:
            return 0
        if self.hp >= self.maxhp:
            return 0
        self.hp += amount
        if self.hp > self.maxhp:
            amount -= self.hp - self.maxhp
            self.hp = self.maxhp
        return amount

    def faint(self, source=None, effect=None):
        if self.fainted or self.faint_queued:
            return 0
        d = self.hp
        self.hp = 0
        self.faint_queued = True
        return d

    def clear_status(self, *args, **kwargs):
        self.status = ""
        self.status_state = {"id": "", "target": self}
        return True

    clearStatus = clear_status

    def cure_status(self, *args, **kwargs):
        if not self.hp or not self.status:
            return False
        self.clear_status()
        return True

    cureStatus = cure_status

    def set_status(self, status, source=None, source_effect=None, ignore_immunities=False) -> bool:
        if not self.hp:
            return False
        status_obj = self.battle.dex.conditions.get(status)
        if not status_obj or not status_obj.id:
            self.clear_status()
            return True
        status_id = status_obj.id
        if self.status == status_id:
            return False

        prev_status = self.status
        prev_state = dict(self.status_state) if self.status_state else {"id": "", "target": self}

        self.status = status_id
        self.status_state = self.battle.init_effect_state({"id": status_id, "target": self})
        if source is not None:
            self.status_state["source"] = source

        started = self.battle.single_event(
            "Start", status_obj, self.status_state, self, source, source_effect
        )
        if started is False:
            self.status = prev_status
            self.status_state = prev_state
            return False
        return True

    setStatus = set_status

    # -- positional helpers used by dispatch ---------------------------- #
    def alliesAndSelf(self) -> List["Pokemon"]:  # noqa: N802 (matches Showdown name surface)
        return [p for p in self.side.active if p and not p.fainted]

    def allies(self) -> List["Pokemon"]:
        return [p for p in self.side.active if p and p is not self and not p.fainted]

    def foes(self) -> List["Pokemon"]:
        foe = self.side.foe
        return [p for p in foe.active if p and not p.fainted] if foe else []

    def get_field_position_value(self) -> int:
        return self.side.n * self.battle.active_per_half + self.position
