"""Full-CSS bb_12_2 memory circuits for an H2 feasibility preflight.

Zero HQC. Does not submit hardware. Does not change Adaptive V1 or the
existing Z-only `build_z_memory_circuit` used by the frozen paper path.

Prep is stabilizer-measurement + Pauli-frame tracking (Stim CSS convention):
  Z-memory |00>_L : data |0>^n, first X-checks define the frame (not detectors)
  X-memory |++>_L : data |+>^n, first Z-checks define the frame (not detectors)

Ancilla layouts:
  separate : 12 data + 5 Z-anc + 5 X-anc = 22 qubits
  reuse    : 12 data + 5 shared ancillas with mid-circuit reset = 17 qubits
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np

from failure_prediction.qldpc_circuit import (
    _require_stim,
    x_logical_basis,
    z_logical_basis,
)
from failure_prediction.qldpc_codes import CSSCode

Basis = Literal["Z", "X"]
Layout = Literal["separate", "reuse"]


def _append_h(circuit, qubits, p: float) -> None:
    if not qubits:
        return
    circuit.append("H", qubits)
    if p:
        circuit.append("DEPOLARIZE1", qubits, p)


def _extract_x(circuit, hx, data, x_anc, p: float) -> None:
    """Measure X-stabilizers: H on ancilla, CX(anc → data), H, then caller measures."""
    _append_h(circuit, x_anc, p)
    circuit.append("TICK")
    hx = hx.tocsr()
    for check, anc in enumerate(x_anc):
        support = hx.indices[hx.indptr[check] : hx.indptr[check + 1]]
        for q in support:
            circuit.append("CX", [int(anc), int(q)])
            if p:
                circuit.append("DEPOLARIZE2", [int(anc), int(q)], p)
        circuit.append("TICK")
    _append_h(circuit, x_anc, p)
    circuit.append("TICK")


def _extract_z(circuit, hz, data, z_anc, p: float) -> None:
    """Measure Z-stabilizers: CX(data → anc), then caller measures."""
    hz = hz.tocsr()
    for check, anc in enumerate(z_anc):
        support = hz.indices[hz.indptr[check] : hz.indptr[check + 1]]
        for q in support:
            circuit.append("CX", [int(q), int(anc)])
            if p:
                circuit.append("DEPOLARIZE2", [int(q), int(anc)], p)
        circuit.append("TICK")


def build_css_memory_circuit(
    code: CSSCode,
    p: float,
    rounds: int,
    basis: Basis = "Z",
    ancilla_layout: Layout = "reuse",
):
    """Full CSS memory: both H_X and H_Z extraction each round.

    Measurement record (hardware packing):
        for r in 0..rounds-1:  n_hx X-check bits, then n_hz Z-check bits
        then n data bits (Z-basis, or X-basis via H-then-M)

    Detectors follow Stim CSS memory:
        determined sector at round 0 is a detector; the complementary
        sector at round 0 is Pauli-frame only. Later rounds XOR both
        sectors. Final detectors reconstruct the determined sector from
        data. OBSERVABLE_INCLUDE is the matching logical basis.
    """
    if basis not in ("Z", "X"):
        raise ValueError(basis)
    if ancilla_layout not in ("separate", "reuse"):
        raise ValueError(ancilla_layout)
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    stim = _require_stim()
    circuit = stim.Circuit()
    n = code.n
    hx = code.hx.tocsr()
    hz = code.hz.tocsr()
    n_hx = int(hx.shape[0])
    n_hz = int(hz.shape[0])
    data = list(range(n))
    if ancilla_layout == "separate":
        z_anc = list(range(n, n + n_hz))
        x_anc = list(range(n + n_hz, n + n_hz + n_hx))
        shared = False
    else:
        z_anc = list(range(n, n + n_hz))
        x_anc = list(range(n, n + n_hx))
        shared = True

    meas_count = 0
    x_round: List[List[int]] = []
    z_round: List[List[int]] = []

    def rec(abs_index: int):
        return stim.target_rec(abs_index - meas_count)

    def reset(qubits):
        circuit.append("R", qubits)
        if p:
            circuit.append("X_ERROR", qubits, p)

    def measure(qubits) -> List[int]:
        nonlocal meas_count
        ids = []
        for q in qubits:
            circuit.append("M", [q], p)
            ids.append(meas_count)
            meas_count += 1
        return ids

    reset(data)
    if basis == "X":
        _append_h(circuit, data, p)
    circuit.append("TICK")

    for r in range(rounds):
        if shared:
            reset(x_anc)
            circuit.append("TICK")
            _extract_x(circuit, hx, data, x_anc, p)
            x_ids = measure(x_anc)
            reset(z_anc)
            circuit.append("TICK")
            _extract_z(circuit, hz, data, z_anc, p)
            z_ids = measure(z_anc)
        else:
            reset(x_anc + z_anc)
            circuit.append("TICK")
            _extract_x(circuit, hx, data, x_anc, p)
            _extract_z(circuit, hz, data, z_anc, p)
            x_ids = measure(x_anc)
            z_ids = measure(z_anc)

        x_round.append(x_ids)
        z_round.append(z_ids)

        for check, abs_m in enumerate(x_ids):
            if basis == "X" and r == 0:
                circuit.append("DETECTOR", [rec(abs_m)])
            elif r > 0:
                circuit.append("DETECTOR", [rec(abs_m), rec(x_round[r - 1][check])])
        for check, abs_m in enumerate(z_ids):
            if basis == "Z" and r == 0:
                circuit.append("DETECTOR", [rec(abs_m)])
            elif r > 0:
                circuit.append("DETECTOR", [rec(abs_m), rec(z_round[r - 1][check])])

        if p:
            circuit.append("DEPOLARIZE1", data, p)
        circuit.append("TICK")

    if basis == "X":
        _append_h(circuit, data, p)
    final = measure(data)

    if basis == "Z":
        checks = hz
        last = z_round[-1]
        logicals = z_logical_basis(code)
    else:
        checks = hx
        last = x_round[-1]
        logicals = x_logical_basis(code)
    checks = checks.tocsr()
    for check, last_abs_m in enumerate(last):
        support = checks.indices[checks.indptr[check] : checks.indptr[check + 1]]
        targets = [rec(last_abs_m)] + [rec(final[int(q)]) for q in support]
        circuit.append("DETECTOR", targets)

    if code.k is not None and logicals.shape[0] != code.k:
        raise ValueError(f"Expected {code.k} {basis} logicals, found {logicals.shape[0]}")
    for obs_idx, logical in enumerate(logicals):
        support = np.flatnonzero(logical)
        targets = [rec(final[int(q)]) for q in support]
        circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)
    return circuit


def measurement_layout(code: CSSCode, rounds: int) -> dict:
    n_hx = int(code.hx.shape[0])
    n_hz = int(code.hz.shape[0])
    per = n_hx + n_hz
    return {
        "order": "each round: n_hx X-check bits, then n_hz Z-check bits; then n data bits",
        "n_hx": n_hx,
        "n_hz": n_hz,
        "n_data": int(code.n),
        "rounds": rounds,
        "bits_per_round": per,
        "n_syndrome_bits": per * rounds,
        "n_meas": per * rounds + int(code.n),
        "hardware_pack": (
            "concat s0[0..9], s1, ..., s{r-1}, d[0..11] by name "
            "(each s{{r}} = 5 X-check bits then 5 Z-check bits). "
            "Do not use pytket lexicographic circuit.bits order."
        ),
    }


def stim_to_pytket_css(stim_circ, n_per_round: int, n_data: int, rounds: int):
    """Translate a quiet full-CSS Stim circuit. One ≤64-bit creg per round."""
    from pytket.circuit import Bit, Circuit

    nq = stim_circ.num_qubits
    n_meas = stim_circ.num_measurements
    circ = Circuit(nq)
    for r in range(rounds):
        circ.add_c_register(f"s{r}", n_per_round)
    circ.add_c_register("d", n_data)

    bit_i = 0
    initialized = set()
    n_syn = n_per_round * rounds

    def meas_bit(idx: int):
        if idx < n_syn:
            return Bit(f"s{idx // n_per_round}", idx % n_per_round)
        return Bit("d", idx - n_syn)

    for op in stim_circ:
        name = op.name
        tgts = [t.value for t in op.targets_copy()]
        if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name == "R":
            for q in tgts:
                if q in initialized:
                    circ.Reset(q)
                initialized.add(q)
            continue
        if name == "H":
            for q in tgts:
                circ.H(q)
        elif name == "CX":
            for i in range(0, len(tgts), 2):
                circ.CX(tgts[i], tgts[i + 1])
        elif name == "M":
            for q in tgts:
                circ.Measure(q, meas_bit(bit_i))
                bit_i += 1
        elif name == "MR":
            for q in tgts:
                circ.Measure(q, meas_bit(bit_i))
                circ.Reset(q)
                bit_i += 1
                initialized.add(q)
        else:
            raise ValueError(f"unsupported stim op for pytket translate: {name}")
    if bit_i != n_meas:
        raise RuntimeError(f"bit count mismatch {bit_i} vs {n_meas}")
    return circ


def peek_codespace(code: CSSCode, basis: Basis) -> dict:
    """Tableau check that stab-prep lands in the CSS codespace with |00>_L / |++>_L.

    After data reset (and H for X-memory), Z-type (resp. X-type) stabilizers and
    logicals are deterministic +1. After one noiseless X- and Z-extract, both
    sectors are deterministic (Pauli frame).
    """
    import stim

    n = code.n
    hx = code.hx.toarray() % 2
    hz = code.hz.toarray() % 2
    zlog = z_logical_basis(code)
    xlog = x_logical_basis(code)

    def pauli(kind: str, support) -> stim.PauliString:
        xs = np.zeros(n, dtype=np.bool_)
        zs = np.zeros(n, dtype=np.bool_)
        for q in np.flatnonzero(support):
            if kind == "X":
                xs[int(q)] = True
            else:
                zs[int(q)] = True
        return stim.PauliString.from_numpy(xs=xs, zs=zs, num_qubits=n)

    def expect(sim: stim.TableauSimulator, kind: str, rows) -> List[float]:
        out = []
        for row in rows:
            out.append(float(sim.peek_observable_expectation(pauli(kind, row))))
        return out

    sim = stim.TableauSimulator()
    sim.do(stim.Circuit("R " + " ".join(str(q) for q in range(n))))
    if basis == "X":
        sim.do(stim.Circuit("H " + " ".join(str(q) for q in range(n))))

    before = {
        "z_stab": expect(sim, "Z", hz),
        "x_stab": expect(sim, "X", hx),
        "z_log": expect(sim, "Z", zlog),
        "x_log": expect(sim, "X", xlog),
    }

    # One noiseless extract using the real circuit, but stop before final readout:
    # run rounds=1 quiet circuit's gates until the last ancilla M of round 0,
    # then peek data. Easier: copy data state by running round-1 circuit and
    # using the fact that noiseless detectors/observables are 0 (checked elsewhere).
    # Here we apply the X/Z extract unitaries without measuring, which is the
    # same as measuring and tracking the frame for commuting CSS stabs when p=0
    # if we do not look at ancillas. Measuring X-stabs projects; without M the
    # data is not yet projected. So we must measure.

    circ = build_css_memory_circuit(code, p=0.0, rounds=1, basis=basis, ancilla_layout="reuse")
    # Drop final data H/M, final DETECTORs, OBSERVABLE_INCLUDE — keep one round.
    prefix = stim.Circuit()
    n_anc_meas = int(code.hx.shape[0]) + int(code.hz.shape[0])
    seen_m = 0
    for op in circ:
        if op.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        if op.name == "M":
            prefix.append(op)
            seen_m += len(op.targets_copy())
            if seen_m >= n_anc_meas:
                break
            continue
        prefix.append(op)

    sim2 = stim.TableauSimulator()
    sim2.do(prefix)
    after = {
        "z_stab": expect(sim2, "Z", hz),
        "x_stab": expect(sim2, "X", hx),
        "z_log": expect(sim2, "Z", zlog),
        "x_log": expect(sim2, "X", xlog),
    }

    def det_pm1(vals) -> bool:
        return all(abs(v) == 1.0 for v in vals)

    if basis == "Z":
        encoded_before = all(v == 1.0 for v in before["z_stab"] + before["z_log"])
        encoded_after = (
            det_pm1(after["z_stab"])
            and det_pm1(after["x_stab"])
            and all(v == 1.0 for v in after["z_stab"] + after["z_log"])
        )
    else:
        encoded_before = all(v == 1.0 for v in before["x_stab"] + before["x_log"])
        encoded_after = (
            det_pm1(after["z_stab"])
            and det_pm1(after["x_stab"])
            and all(v == 1.0 for v in after["x_stab"] + after["x_log"])
        )
    return {
        "basis": basis,
        "before_extract": before,
        "after_one_round": after,
        "data_reset_in_determined_eigenspace": encoded_before,
        "after_extract_full_codespace": encoded_after,
        "note": (
            "Z-memory: |0>^n is +1 for all Z-stabs and Z-logicals; first X-extract "
            "projects onto an X-stab eigenspace (Pauli frame). X-memory: |+>^n is "
            "+1 for all X-stabs and X-logicals; first Z-extract completes the codespace."
        ),
    }
