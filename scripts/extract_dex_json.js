/**
 * Dump Pokemon Showdown's per-gen Dex data tables to JSON for the verbatim
 * Python port (pokepy.showdown.dex).
 *
 * Uses Showdown's own compiled Dex so the *data fields* are exact and
 * complete. JSON.stringify naturally drops callback functions (onHit,
 * basePowerCallback, ...); those are hand-translated to Python in Phase B by
 * reading the .ts source. The data-driven fields dumped here (basePower, type,
 * category, accuracy, secondary, boosts, recoil, drain, flags, baseStats,
 * damageTaken, ...) cover the common case with no function translation.
 *
 * Usage:
 *   node scripts/extract_dex_json.js [output_dir] [gen,gen,...]
 *   (defaults: output=pokepy/data/showdown, gens=1,2,3,4,9)
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SHOWDOWN = path.resolve(__dirname, '../../../../server/pokemon-showdown');
const { Dex } = require(path.join(SHOWDOWN, 'dist/sim/dex.js'));

const OUT = process.argv[2]
	? path.resolve(process.argv[2])
	: path.resolve(__dirname, '../pokepy/data/showdown');
const GENS = process.argv[3]
	? process.argv[3].split(',').map(Number)
	: [1, 2, 3, 4, 9];

// JSON round-trip drops functions and undefined; gives plain data.
function clean(obj) {
	return JSON.parse(JSON.stringify(obj));
}

/** List callback property names on a Showdown data object (functions only). */
function listCallbacks(obj) {
	if (!obj || typeof obj !== 'object') return [];
	const keys = [];
	for (const k of Object.keys(obj)) {
		if (typeof obj[k] === 'function') keys.push(k);
	}
	return keys.sort();
}

function dumpTable(arr) {
	const out = {};
	for (const e of arr) {
		if (!e || !e.id) continue;
		if (e.exists === false) continue;
		out[e.id] = clean(e);
	}
	return out;
}

function dumpCallbackManifest(arr) {
	const out = {};
	for (const e of arr) {
		if (!e || !e.id) continue;
		if (e.exists === false) continue;
		const cbs = listCallbacks(e);
		if (cbs.length) out[e.id] = cbs;
	}
	return out;
}

for (const gen of GENS) {
	const d = Dex.forGen(gen);
	const dir = path.join(OUT, `gen${gen}`);
	fs.mkdirSync(dir, { recursive: true });

	const tables = {
		moves: () => d.moves.all(),
		species: () => d.species.all(),
		abilities: () => d.abilities.all(),
		items: () => d.items.all(),
		typechart: () => d.types.all(),
		natures: () => d.natures.all(),
		// Conditions (statuses, weather, terrains, move/volatile conditions) have
		// no .all(); pull the raw modded data table and resolve each via the Dex
		// so effectType/duration/etc. fields are gen-correct.
		conditions: () => Object.keys(d.data.Conditions || {}).map(id => d.conditions.getByID(id)),
	};

	const counts = {};
	const callbackManifest = {};
	for (const [name, getter] of Object.entries(tables)) {
		let raw = [];
		let table = {};
		try {
			raw = getter();
			table = dumpTable(raw);
		} catch (e) {
			console.error(`gen${gen} ${name}: ${e.message}`);
		}
		fs.writeFileSync(path.join(dir, `${name}.json`), JSON.stringify(table));
		counts[name] = Object.keys(table).length;
		if (raw.length) {
			const cbs = dumpCallbackManifest(raw);
			if (Object.keys(cbs).length) callbackManifest[name] = cbs;
		}
	}
	if (Object.keys(callbackManifest).length) {
		fs.writeFileSync(
			path.join(dir, 'callbacks.json'),
			JSON.stringify(callbackManifest, null, 2),
		);
		counts.callbacks = Object.values(callbackManifest).reduce(
			(n, t) => n + Object.keys(t).length, 0,
		);
	}
	console.log(`gen${gen}:`, JSON.stringify(counts));
}
console.log('Done. Output:', OUT);
