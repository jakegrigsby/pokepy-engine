"""BattleActions: the move pipeline, ported from sim/battle-actions.ts.

Slice scope (gen9, singles, single-target): the verbatim spine
runMove -> useMove -> useMoveInner -> trySpreadMoveHit -> hitStep* ->
spreadMoveHit -> getSpreadDamage -> getDamage -> modifyDamage -> runMoveEffects
-> secondaries -> calcRecoilDamage, reproducing Showdown's exact PRNG-frame
order (accuracy -> crit -> damage roll -> secondary roll). Breadth of effect
callbacks is Phase B; data-driven fields (basePower/secondary/boosts/status/
recoil/drain) are handled directly here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from pokepy.showdown.battle import Battle
    from pokepy.showdown.pokemon import Pokemon


class _DictAttr:
    """Attribute-access view over a hit-effect dict (secondary / self / boosts).

    Showdown passes ``moveData`` as an object whose missing fields read as
    ``undefined``; our secondary/self effects are plain dicts. Wrapping them
    here lets the pipeline read ``move_data.boosts`` etc. uniformly (missing ->
    None) exactly like the ActiveMove path.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name):
        return self._d.get(name)

    def __setattr__(self, name, value):
        self._d[name] = value

    def get(self, name, default=None):
        return self._d.get(name, default)


def _as_effect(move_data):
    return _DictAttr(move_data) if isinstance(move_data, dict) else move_data


class BattleActions:
    def __init__(self, battle: "Battle"):
        self.battle = battle

    @property
    def dex(self):
        return self.battle.dex

    # ================================================================== #
    # Switch-in (sim/battle-actions.ts switchIn / runSwitch)
    # ================================================================== #
    def switch_in(self, pokemon, pos, source_effect=None, is_drag=False):
        battle = self.battle
        if not pokemon or pokemon.is_active:
            return False
        side = pokemon.side
        old_active = side.active[pos] if pos < len(side.active) else None

        if old_active is not None:
            if old_active.hp:
                battle.run_event("BeforeSwitchOut", old_active)
                if battle.gen >= 5:
                    battle.each_event("Update")
                if not battle.run_event("SwitchOut", old_active):
                    return False
                if not old_active.hp:
                    return "pursuitfaint"
                battle.single_event("End", old_active.get_ability(), old_active.ability_state, old_active)
                battle.single_event("End", old_active.get_item(), old_active.item_state, old_active)
                battle.queue.cancel_action(old_active)
                old_active.clear_volatile() if hasattr(old_active, "clear_volatile") else None
            old_active.is_active = False
            old_active.active = False
            old_active.is_started = False
            old_active.used_item_this_turn = False
            old_active.position = pokemon.position
            if old_active.fainted:
                old_active.status = ""
            pokemon.position = pos
            side.pokemon[pokemon.position] = pokemon
            side.pokemon[old_active.position] = old_active

        pokemon.is_active = True
        pokemon.active = True
        side.active[pos] = pokemon
        pokemon.active_move_actions = 0
        for move_slot in pokemon.move_slots:
            move_slot["used"] = False
        pokemon.ability_state = battle.init_effect_state({"id": pokemon.ability, "target": pokemon})
        pokemon.item_state = battle.init_effect_state({"id": pokemon.item, "target": pokemon})
        battle.run_event("BeforeSwitchIn", pokemon)
        battle.add("drag" if is_drag else "switch", pokemon, pokemon.species.name)

        if is_drag and battle.gen >= 5:
            self.run_switch(pokemon)
        else:
            battle.queue.insert_choice(self._action("runSwitch", pokemon=pokemon))
        return True

    def run_switch(self, pokemon):
        battle = self.battle
        switchers_in = [pokemon]
        while battle.queue.peek() and battle.queue.peek().get("choice") == "runSwitch":
            nxt = battle.queue.shift()
            switchers_in.append(nxt.get("pokemon"))
        all_active = battle.get_all_active(True)
        battle.speed_sort(all_active)
        battle.speed_order = [p.get_field_position_value() for p in all_active]
        battle.field_event("SwitchIn", switchers_in)
        for poke in switchers_in:
            if not poke.hp:
                continue
            poke.is_started = True
        return True

    @staticmethod
    def _action(choice, **kw):
        from pokepy.showdown.battle_queue import Action

        return Action(choice=choice, **kw)

    # ================================================================== #
    def run_move(self, move_or_name, pokemon: "Pokemon", target_loc: int, target=None,
                 source_effect=None, external_move=False):
        battle = self.battle
        pokemon.active_move_actions += 1
        if target is None:
            target = battle.get_target(pokemon, move_or_name, target_loc)
        base_move = self.dex.get_active_move(move_or_name)
        move = base_move
        battle.set_active_move(move, pokemon, target)

        will_try = battle.run_event("BeforeMove", pokemon, target, move)
        if not will_try:
            battle.run_event("MoveAborted", pokemon, target, move)
            battle.clear_active_move(True)
            pokemon.move_this_turn_result = will_try
            if battle.gen <= 2:
                battle.run_event("AfterMoveSelf", pokemon, target, move)
            return

        pokemon.last_damage = 0
        if not external_move:
            if not pokemon.deduct_pp(base_move, None, target) and move.id != "struggle":
                battle.add("cant", pokemon, "nopp", move)
                battle.clear_active_move(True)
                pokemon.move_this_turn_result = False
                return
            pokemon.move_used(move, target_loc)

        self.use_move(base_move, pokemon, target=target, source_effect=source_effect)
        battle.last_successful_move_this_turn = (
            battle.active_move.id if battle.active_move else None
        )
        if battle.active_move:
            move = battle.active_move
        battle.single_event("AfterMove", move, None, pokemon, target, move)
        battle.run_event("AfterMove", pokemon, target, move)
        if battle.gen == 2:
            foe = pokemon.side.foe
            active_foe = foe.active[0] if foe and foe.active else None
            if not getattr(move, "selfSwitch", None) and active_foe and active_foe.hp:
                battle.run_event("AfterMoveSelf", pokemon, target, move)
        elif battle.gen == 1 and target and target.hp > 0:
            battle.run_event("AfterMoveSelf", pokemon, target, move)
        battle.faint_messages()
        battle.check_win()

    def use_move(self, move, pokemon, target=None, source_effect=None):
        pokemon.move_this_turn_result = None
        old = pokemon.move_this_turn_result
        result = self.use_move_inner(move, pokemon, target=target, source_effect=source_effect)
        if old == pokemon.move_this_turn_result:
            pokemon.move_this_turn_result = result
        return result

    def use_move_inner(self, move_or_name, pokemon, target=None, source_effect=None):
        battle = self.battle
        move = self.dex.get_active_move(move_or_name)
        pokemon.last_move_used = move
        if target is None:
            target = battle.get_random_target(pokemon, move)
        if move.target in ("self", "allies"):
            target = pokemon

        battle.set_active_move(move, pokemon, target)
        battle.single_event("ModifyType", move, None, pokemon, target, move, move)
        battle.single_event("ModifyMove", move, None, pokemon, target, move, move)
        move = battle.run_event("ModifyType", pokemon, target, move, move)
        move = battle.run_event("ModifyMove", pokemon, target, move, move)
        if not move or pokemon.fainted:
            return False

        battle.add_move("move", pokemon, move.name, str(target) if target else "")

        if not target:
            battle.add("-notarget" if battle.gen < 5 else "-fail", pokemon)
            return False

        if move.ignoreImmunity is None:
            move.ignoreImmunity = (move.category == "Status")

        try_move = battle.single_event("TryMove", move, None, pokemon, target, move)
        if try_move:
            try_move = battle.run_event("TryMove", pokemon, target, move)
        if not try_move:
            return try_move

        targets = [target]
        move_result = self.try_spread_move_hit(targets, pokemon, move)

        if move.selfBoost and move_result:
            self.move_hit(pokemon, pokemon, move, move.selfBoost, False, True)
        if not pokemon.hp:
            battle.faint(pokemon, pokemon, move)
        if not move_result:
            battle.single_event("MoveFail", move, None, target, pokemon, move)
            return False
        battle.single_event("AfterMoveSecondarySelf", move, None, pokemon, target, move)
        battle.run_event("AfterMoveSecondarySelf", pokemon, target, move)
        return True

    # ================================================================== #
    def try_spread_move_hit(self, targets, pokemon, move):
        battle = self.battle
        # gen9 step order (no swaps).
        move_steps = [
            self.hit_step_invulnerability_event,
            self.hit_step_try_hit_event,
            self.hit_step_type_immunity,
            self.hit_step_try_immunity,
            self.hit_step_accuracy,
            self.hit_step_break_protect,
            self.hit_step_move_hit_loop,
        ]
        if battle.gen <= 6:
            move_steps[1], move_steps[2] = move_steps[2], move_steps[1]
        if battle.gen == 4:
            move_steps[2], move_steps[4] = move_steps[4], move_steps[2]

        hit_result = (
            battle.single_event("Try", move, None, pokemon, targets[0], move)
            and battle.single_event("PrepareHit", move, {}, targets[0], pokemon, move)
            and battle.run_event("PrepareHit", pokemon, targets[0], move)
        )
        if not hit_result:
            if hit_result is False:
                battle.add("-fail", pokemon)
            return hit_result == battle.NOT_FAIL

        at_least_one_failure = False
        for step in move_steps:
            hit_results = step(targets, pokemon, move)
            if hit_results is None:
                continue
            targets = [t for t, r in zip(targets, hit_results) if r or (r == 0 and r is not False)]
            at_least_one_failure = at_least_one_failure or any(r is False for r in hit_results)
            if not targets:
                break

        move.hitTargets = targets
        move_result = bool(targets)
        if not move_result and not at_least_one_failure:
            pokemon.move_this_turn_result = None
        return move_result

    def hit_step_invulnerability_event(self, targets, pokemon, move):
        battle = self.battle
        results = []
        for target in targets:
            results.append(battle.run_event("Invulnerability", target, pokemon, move))
            if results[-1] is False:
                battle.add("-miss", pokemon, target)
        return results

    def hit_step_try_hit_event(self, targets, pokemon, move):
        battle = self.battle
        results = battle.run_event("TryHit", targets, pokemon, move)
        if not isinstance(results, list):
            results = [results for _ in targets]
        return [(r if r is not None else False) for r in results]

    def hit_step_type_immunity(self, targets, pokemon, move):
        if move.ignoreImmunity is None:
            move.ignoreImmunity = (move.category == "Status")
        return [t.run_immunity(move, True) for t in targets]

    def hit_step_try_immunity(self, targets, pokemon, move):
        battle = self.battle
        results = []
        for target in targets:
            ok = battle.single_event("TryImmunity", move, {}, target, pokemon, move)
            results.append(bool(ok) if ok is not None else True)
        return results

    def hit_step_accuracy(self, targets, pokemon, move):
        battle = self.battle
        results = []
        for target in targets:
            battle.active_target = target
            accuracy = move.accuracy
            if accuracy is not True:
                accuracy = battle.run_event("ModifyAccuracy", target, pokemon, move, accuracy)
                boost = 0
                if not move.ignoreAccuracy:
                    boosts = battle.run_event("ModifyBoost", pokemon, None, None, dict(pokemon.boosts))
                    boost = battle.clamp_int_range(boosts["accuracy"], -6, 6)
                if not move.ignoreEvasion:
                    boosts = battle.run_event("ModifyBoost", target, None, None, dict(target.boosts))
                    boost = battle.clamp_int_range(boost - boosts["evasion"], -6, 6)
                if boost > 0:
                    accuracy = battle.trunc(accuracy * (3 + boost) / 3)
                elif boost < 0:
                    accuracy = battle.trunc(accuracy * 3 / (3 - boost))
            if move.alwaysHit or (move.target == "self" and move.category == "Status"):
                accuracy = True
            else:
                accuracy = battle.run_event("Accuracy", target, pokemon, move, accuracy)
            if accuracy is not True and not battle.random_chance(accuracy, 100):
                battle.add("-miss", pokemon, target)
                results.append(False)
                continue
            results.append(True)
        return results

    def hit_step_break_protect(self, targets, pokemon, move):
        return None

    def hit_step_move_hit_loop(self, targets, pokemon, move):
        battle = self.battle
        damage: List[Any] = [0 for _ in targets]
        move.totalDamage = 0
        pokemon.last_damage = 0
        target_hits = move.multihit or 1
        if isinstance(target_hits, list):
            if target_hits == [2, 5]:
                target_hits = battle.sample([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5])
            else:
                target_hits = battle.random(target_hits[0], target_hits[1] + 1)
        target_hits = int(target_hits)

        null_damage = True
        move_damage: List[Any] = []
        hit = 1
        while hit <= target_hits:
            if any(d is False for d in damage):
                break
            if all((not t or not t.hp) for t in targets):
                break
            move.hit = hit
            move.lastHit = hit == target_hits
            targets_copy = list(targets)
            move_damage, targets_copy = self.spread_move_hit(targets_copy, pokemon, move, move)
            if not any(v is not False for v in move_damage):
                break
            null_damage = False
            for i, md in enumerate(move_damage):
                damage[i] = 0 if (md is True or not md) else md
                move.totalDamage += damage[i]
            battle.each_event("Update")
            if not pokemon.hp and len(targets) == 1:
                hit += 1
                break
            hit += 1

        if hit == 1:
            return [False for _ in damage]
        if null_damage:
            damage = [False for _ in damage]
        battle.faint_messages()

        if (move.recoil) and move.totalDamage:
            battle.damage(self.calc_recoil_damage(move.totalDamage, move, pokemon), pokemon, pokemon, "recoil")
        if move.struggleRecoil:
            recoil = battle.clamp_int_range(round(pokemon.maxhp / 4), 1)
            battle.direct_damage(recoil, pokemon, pokemon, {"id": "strugglerecoil"})

        if not any(v is not False for v in damage):
            return damage
        battle.each_event("Update")
        return damage

    def spread_move_hit(self, targets, pokemon, move, hit_effect=None, is_secondary=False, is_self=False):
        battle = self.battle
        move = self.dex.get_active_move(move)
        move_data = _as_effect(hit_effect) if hit_effect is not None else move
        damage: List[Any] = [True for _ in targets]

        target = targets[0] if targets else None
        if target:
            hit_result = battle.single_event("TryHit", move_data, {}, target, pokemon, move)
            if not hit_result:
                if hit_result is False:
                    battle.add("-fail", pokemon)
                return [[False], targets]

        # 1. damage calc
        damage = self.get_spread_damage(damage, targets, pokemon, move, move_data, is_secondary, is_self)
        for i in range(len(targets)):
            if damage[i] is False:
                targets[i] = False
        # 2. apply damage
        damage = battle.spread_damage(damage, targets, pokemon, move)
        if not isinstance(damage, list):
            damage = [damage]
        for i in range(len(targets)):
            if damage[i] is False:
                targets[i] = False
        # 3. move effects (boosts/status/heal/volatile - data driven)
        damage = self.run_move_effects(damage, targets, pokemon, move, move_data, is_secondary, is_self)
        for i in range(len(targets)):
            if damage[i] is False or damage[i] is None:
                targets[i] = False
        # 5. secondaries
        if move_data.secondaries:
            self.secondaries(targets, pokemon, move, move_data, is_self)

        damaged_targets = [t for i, t in enumerate(targets) if isinstance(damage[i], int) and not isinstance(damage[i], bool) and t]
        damaged_damage = [damage[i] for i, t in enumerate(targets) if isinstance(damage[i], int) and not isinstance(damage[i], bool) and t]
        if damaged_damage and not is_secondary and not is_self and battle.gen >= 5:
            battle.run_event("DamagingHit", damaged_targets, pokemon, move, damaged_damage)
        return [damage, targets]

    def get_spread_damage(self, damage, targets, source, move, move_data, is_secondary=False, is_self=False):
        battle = self.battle
        for i, target in enumerate(targets):
            if not target:
                continue
            battle.active_target = target
            damage[i] = None
            cur = self.get_damage(source, target, move_data)
            if cur is False:
                damage[i] = False
                continue
            if cur is None:
                # Status / effect-only moves (basePower 0) still hit.
                damage[i] = 0
                continue
            damage[i] = cur
        return damage

    def run_move_effects(self, damage, targets, pokemon, move, move_data, is_secondary=False, is_self=False):
        battle = self.battle
        for i, target in enumerate(targets):
            if target is False:
                continue
            if target:
                if move_data.boosts and not target.fainted:
                    battle.boost(move_data.boosts, target, pokemon, move, is_secondary, is_self)
                if move_data.status:
                    target.set_status(move_data.status, pokemon, move)
                if move_data.heal and not target.fainted:
                    amount = target.maxhp * move_data.heal[0] // move_data.heal[1]
                    battle.heal(amount, target, pokemon, move)
                if move_data.volatileStatus:
                    target.volatiles.setdefault(move_data.volatileStatus, {"id": move_data.volatileStatus, "target": target})
            if move_data.self and not is_self:
                self.move_hit(pokemon, pokemon, move, move_data.self, is_secondary, True)
        return damage

    def secondaries(self, targets, source, move, move_data, is_self=False):
        battle = self.battle
        if not move_data.secondaries:
            return
        for target in targets:
            if target is False:
                continue
            secondaries = battle.run_event("ModifySecondaries", target, source, move_data, list(move_data.secondaries))
            for secondary in secondaries:
                secondary_roll = battle.random(100)
                chance = secondary.get("chance")
                if chance is None or secondary_roll < chance:
                    self.move_hit(target, source, move, secondary, True, is_self)

    def move_hit(self, targets, pokemon, move, move_data=None, is_secondary=False, is_self=False):
        if not isinstance(targets, list):
            targets = [targets]
        ret = self.spread_move_hit(targets, pokemon, move, move_data, is_secondary, is_self)[0][0]
        return None if ret is True else ret

    def calc_recoil_damage(self, damage_dealt: int, move, pokemon) -> int:
        recoil = move.recoil
        return self.battle.clamp_int_range(round(damage_dealt * recoil[0] / recoil[1]), 1)

    # ================================================================== #
    # Damage formula (sim/battle-actions.ts:1589 getDamage + 1728 modifyDamage)
    # ================================================================== #
    def get_damage(self, source, target, move, suppress_messages=False):
        battle = self.battle
        if isinstance(move, str):
            move = self.dex.get_active_move(move)

        if not target.run_immunity(move, not suppress_messages):
            return False

        if move.ohko:
            return target.maxhp
        if move.damage == "level":
            return source.level
        elif move.damage:
            return move.damage

        category = battle.get_category(move)
        base_power = move.basePower
        if not base_power:
            return None if base_power == 0 else base_power
        base_power = battle.clamp_int_range(base_power, 1)

        crit_ratio = battle.run_event("ModifyCritRatio", source, target, move, move.critRatio or 0)
        if battle.gen <= 5:
            crit_ratio = battle.clamp_int_range(crit_ratio, 0, 5)
            crit_mult = [0, 16, 8, 4, 3, 2]
        else:
            crit_ratio = battle.clamp_int_range(crit_ratio, 0, 4)
            if battle.gen == 6:
                crit_mult = [0, 16, 8, 2, 1]
            else:
                crit_mult = [0, 24, 8, 2, 1]

        move_hit = target.get_move_hit_data(move)
        move_hit["crit"] = move.willCrit or False
        if move.willCrit is None:
            if crit_ratio:
                move_hit["crit"] = battle.random_chance(1, crit_mult[crit_ratio])
        if move_hit["crit"]:
            move_hit["crit"] = battle.run_event("CriticalHit", target, None, move)

        base_power = battle.run_event("BasePower", source, target, move, base_power, True)
        if not base_power:
            return 0
        base_power = battle.clamp_int_range(base_power, 1)

        level = source.level
        is_physical = move.category == "Physical"
        attack_stat = move.overrideOffensiveStat or ("atk" if is_physical else "spa")
        defense_stat = move.overrideDefensiveStat or ("def" if is_physical else "spd")
        stat_table = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}

        atk_boosts = source.boosts[attack_stat]
        def_boosts = target.boosts[defense_stat]
        if move_hit["crit"]:
            if atk_boosts < 0:
                atk_boosts = 0
            if def_boosts > 0:
                def_boosts = 0

        attack = source.calculate_stat(attack_stat, atk_boosts, 1, source)
        defense = target.calculate_stat(defense_stat, def_boosts, 1, target)
        attack_stat = "atk" if category == "Physical" else "spa"
        attack = battle.run_event("Modify" + stat_table[attack_stat], source, target, move, attack)
        defense = battle.run_event("Modify" + stat_table[defense_stat], target, source, move, defense)

        tr = battle.trunc
        base_damage = tr(tr(tr(tr(2 * level / 5 + 2) * base_power * attack) / defense) / 50)
        return self.modify_damage(base_damage, source, target, move, suppress_messages)

    def modify_damage(self, base_damage, pokemon, target, move, suppress_messages=False):
        if self.battle.gen == 4:
            return self._modify_damage_gen4(base_damage, pokemon, target, move, suppress_messages)
        if self.battle.gen == 3:
            return self._modify_damage_gen3(base_damage, pokemon, target, move, suppress_messages)
        battle = self.battle
        tr = battle.trunc
        if not move.type:
            move.type = "???"
        mtype = move.type
        base_damage += 2

        if move.spreadHit:
            base_damage = battle.modify(base_damage, 0.75)

        # weather modifier (priorityEvent - no frame for vanilla)
        base_damage = battle.priority_event("WeatherModifyDamage", pokemon, target, move, base_damage)

        is_crit = target.get_move_hit_data(move)["crit"]
        if is_crit:
            base_damage = tr(base_damage * (move.critModifier or (1.5 if battle.gen >= 6 else 2)))

        base_damage = battle.randomizer(base_damage)

        if mtype != "???":
            stab = 1
            is_stab = pokemon.has_type(mtype)
            if is_stab:
                stab = 1.5
            if pokemon.terastallized == mtype and pokemon.has_type(mtype):
                stab = 2
            stab = battle.run_event("ModifySTAB", pokemon, target, move, stab)
            base_damage = battle.modify(base_damage, stab)

        type_mod = target.run_effectiveness(move)
        type_mod = battle.clamp_int_range(type_mod, -6, 6)
        target.get_move_hit_data(move)["typeMod"] = type_mod
        if type_mod > 0:
            if not suppress_messages:
                battle.add("-supereffective", target)
            for _ in range(type_mod):
                base_damage *= 2
        if type_mod < 0:
            if not suppress_messages:
                battle.add("-resisted", target)
            for _ in range(0, type_mod, -1):
                base_damage = tr(base_damage / 2)

        if is_crit and not suppress_messages:
            battle.add("-crit", target)

        if pokemon.status == "brn" and move.category == "Physical" and not pokemon.has_ability("guts"):
            if battle.gen < 6 or move.id != "facade":
                base_damage = battle.modify(base_damage, 0.5)

        if battle.gen == 5 and not base_damage:
            base_damage = 1

        base_damage = battle.run_event("ModifyDamage", pokemon, target, move, base_damage)

        if battle.gen != 5 and not base_damage:
            return 1
        return tr(base_damage, 16)

    def _modify_damage_gen4(self, base_damage, pokemon, target, move, suppress_messages=False):
        """Gen4 damage phases (data/mods/gen4/scripts.ts modifyDamage)."""
        battle = self.battle
        if not move.type:
            move.type = "???"
        mtype = move.type

        if pokemon.status == "brn" and base_damage and move.category == "Physical" and not pokemon.has_ability("guts"):
            base_damage = battle.modify(base_damage, 0.5)

        base_damage = battle.run_event("ModifyDamagePhase1", pokemon, target, move, base_damage)

        if move.spreadHit:
            spread_mod = getattr(move, "spreadModifier", None) or 0.75
            base_damage = battle.modify(base_damage, spread_mod)

        base_damage = battle.priority_event("WeatherModifyDamage", pokemon, target, move, base_damage)

        base_damage += 2

        is_crit = target.get_move_hit_data(move)["crit"]
        if is_crit:
            base_damage = battle.modify(base_damage, move.critModifier or 2)

        base_damage = int(battle.run_event("ModifyDamagePhase2", pokemon, target, move, base_damage))

        base_damage = battle.randomizer(base_damage)

        if mtype != "???":
            stab = 1
            if getattr(move, "forceSTAB", None) or pokemon.has_type(mtype):
                stab = 1.5
            stab = battle.run_event("ModifySTAB", pokemon, target, move, stab)
            base_damage = battle.modify(base_damage, stab)

        type_mod = target.run_effectiveness(move)
        type_mod = battle.clamp_int_range(type_mod, -6, 6)
        target.get_move_hit_data(move)["typeMod"] = type_mod
        if type_mod > 0:
            if not suppress_messages:
                battle.add("-supereffective", target)
            for _ in range(type_mod):
                base_damage *= 2
        if type_mod < 0:
            if not suppress_messages:
                battle.add("-resisted", target)
            for _ in range(0, type_mod, -1):
                base_damage = base_damage // 2

        if is_crit and not suppress_messages:
            battle.add("-crit", target)

        base_damage = battle.run_event("ModifyDamage", pokemon, target, move, base_damage)

        if not int(base_damage):
            return 1
        return int(base_damage)

    def _modify_damage_gen3(self, base_damage, pokemon, target, move, suppress_messages=False):
        """Gen3/RSE damage phases (data/mods/gen3/scripts.ts modifyDamage)."""
        battle = self.battle
        if not move.type:
            move.type = "???"
        mtype = move.type

        if pokemon.status == "brn" and base_damage and move.category == "Physical" and not pokemon.has_ability("guts"):
            base_damage = battle.modify(base_damage, 0.5)

        base_damage = battle.run_event("ModifyDamagePhase1", pokemon, target, move, base_damage)

        if move.spreadHit and getattr(move, "target", None) == "allAdjacentFoes":
            base_damage = battle.modify(base_damage, 0.5)

        base_damage = battle.priority_event("WeatherModifyDamage", pokemon, target, move, base_damage)

        if move.category == "Physical" and not int(base_damage):
            base_damage = 1

        base_damage += 2

        is_crit = target.get_move_hit_data(move)["crit"]
        if is_crit:
            base_damage = battle.modify(base_damage, move.critModifier or 2)

        base_damage = int(battle.run_event("ModifyDamagePhase2", pokemon, target, move, base_damage))

        if mtype != "???":
            stab = 1
            if getattr(move, "forceSTAB", None) or pokemon.has_type(mtype):
                stab = 1.5
            stab = battle.run_event("ModifySTAB", pokemon, target, move, stab)
            base_damage = battle.modify(base_damage, stab)

        type_mod = target.run_effectiveness(move)
        type_mod = battle.clamp_int_range(type_mod, -6, 6)
        target.get_move_hit_data(move)["typeMod"] = type_mod
        if type_mod > 0:
            if not suppress_messages:
                battle.add("-supereffective", target)
            for _ in range(type_mod):
                base_damage *= 2
        if type_mod < 0:
            if not suppress_messages:
                battle.add("-resisted", target)
            for _ in range(0, type_mod, -1):
                base_damage = base_damage // 2

        if is_crit and not suppress_messages:
            battle.add("-crit", target)

        base_damage = battle.run_event("ModifyDamage", pokemon, target, move, base_damage)
        base_damage = battle.randomizer(base_damage)

        if not int(base_damage):
            return 1
        return int(base_damage)
