#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
HQCS - Holographic Quantum Circuit Sampling
HQCS adaptacija za Loto 7/39.

Metodološki izvor:
https://arxiv.org/abs/2610.15083

Samostalna implementacija prema metodologiji rada.
Originalni izvorni kod autora nije korišćen.

KVANTNI MODEL
------------
Jedan memorijski i jedan radni kubit.
Nezavisno parametrizovano kolo na svakom koraku.
Radni kubit ZADRŽAVA prethodni izmereni bit.
Memorijski kubit zadržava kompleksne amplitude.

Uzorak = dva uzastopna kola, predstavljena sa 78 bitova.
Svako kolo ima tačno sedam jedinica.
Uči se zajednička raspodela takvih parova.
NEXT je maksimum uslovne raspodele drugog kola kada je prvo
jednako poslednjem poznatom kolu.

FUNKCIJA UČENJA
--------------
MMD² = E_QQ[k] - 2 E_QP[k] + E_PP[k].

k(x,y) = exp(-gamma * Hamming(x,y)).

Ovo je eksponencijalni kernel nad celim sekvencama.
Kernel ne nameće normalnu raspodelu podacima.
Dodatni increment-MMD iz rada nije uključen: lambda = 0.

E_QQ: tačna kontrakcija dve kopije kvantnog modela.
E_QP: tačna kontrakcija za svaki istorijski uzorak.
E_PP: svi parovi istorijskih uzoraka.
Nema Monte Carlo procena, slučajnih kandidata ili šuma.

KLASIČNI DELOVI
--------------
Deterministička complex128 simulacija, autodiferenciranje,
Adam optimizacija i pretraga maksimuma.
Ovo nije izvršavanje na kvantnom procesoru.

PODACI
------
CSV: bez zaglavlja, sedam brojeva, hronološki od starog ka novom.
Redovi se ne mešaju.
Svaki raspoloživi istorijski prelaz ulazi u završno učenje.
Kontekst pojedinačne predikcije je jedno prethodno kolo.
"""


import csv
import hashlib
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    from numba import njit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
except ImportError as exc:
    raise SystemExit(
        "Potrebni paketi: numpy, torch, numba, qiskit.\n"
        f"Detalj: {exc}"
    )

PATHS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_2975.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_plus_1731.csv",
)

NUMBERS = 39
PICKS = 7
DEPTH = 3

# Veličina kvantne memorije namerno je mala zbog tačnog E_QQ.
# Nije klasični regresor: treniraju se samo uglovi kvantnih kapija.
MEMORY_QUBITS = 1

MAX_EPOCHS = 30
CHECK_EVERY = 5
REGULARIZATIONS = (0.0, 0.0001)
LEARNING_RATE = 0.02
BATCH_SIZE = 128

REAL = torch.float64
COMPLEX = torch.complex128

torch.set_default_dtype(REAL)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def signature(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()[:16]


def read_draws(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) != PICKS:
                raise ValueError(
                    f"{path}, red {line}: očekivano 7 polja."
                )
            try:
                numbers = [int(field.strip()) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc

            if len(set(numbers)) != PICKS or not all(
                1 <= number <= NUMBERS for number in numbers
            ):
                raise ValueError(
                    f"{path}, red {line}: nelegalno kolo {numbers}."
                )
            # Sortiranje brojeva unutar skupa ne menja redosled kola.
            rows.append(sorted(numbers))

    require(len(rows) >= 100, f"{path}: manje od 100 kola.")
    draws = np.asarray(rows, dtype=np.int64)
    bits = np.zeros((len(draws), NUMBERS), dtype=np.int64)
    bits[np.arange(len(draws))[:, None], draws - 1] = 1
    require(np.all(bits.sum(1) == PICKS), "Greška kodiranja.")
    sequences = np.concatenate((bits[:-1], bits[1:]), axis=1)
    return draws, bits, sequences


def local_pauli(matrix, qubit, nq):
    result = np.array([[1.0 + 0j]])
    for q in reversed(range(nq)):
        result = np.kron(
            result, matrix if q == qubit else np.eye(2)
        )
    return torch.tensor(result, dtype=COMPLEX)


def cnot_matrix(nq, control, target):
    size = 1 << nq
    result = np.zeros((size, size), dtype=np.complex128)
    for column in range(size):
        row = column
        if (column >> control) & 1:
            row ^= 1 << target
        result[row, column] = 1.0
    return torch.tensor(result, dtype=COMPLEX)


class HQCS(torch.nn.Module):
    def __init__(self, steps, memory=MEMORY_QUBITS, depth=DEPTH):
        super().__init__()
        self.steps = steps
        self.memory = memory
        self.depth = depth
        self.nq = memory + 1
        self.dm = 1 << memory
        size = 1 << self.nq

        theta = torch.empty((steps, depth, self.nq, 2), dtype=REAL)
        position = torch.arange(steps, dtype=REAL) / max(steps - 1, 1)

        # Deterministička inicijalizacija, bez generatora slučajnih brojeva.
        for layer in range(depth):
            for q in range(self.nq):
                theta[:, layer, q, 0] = (
                    0.75 + 0.04 * q + 0.03 * layer + 0.10 * position
                )
                theta[:, layer, q, 1] = (
                    0.11 + 0.03 * q + 0.02 * layer + 0.05 * position
                )
        self.theta = torch.nn.Parameter(theta)

        y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.register_buffer("eye", torch.eye(size, dtype=COMPLEX))
        self.register_buffer(
            "ys",
            torch.stack([local_pauli(y, q, self.nq)
                         for q in range(self.nq)])
        )
        self.register_buffer(
            "zs",
            torch.stack([local_pauli(z, q, self.nq)
                         for q in range(self.nq)])
        )

        entanglers = []
        self.edges = []
        for layer in range(depth):
            edges = [
                (q, q + 1)
                for q in range(layer % 2, self.nq - 1, 2)
            ]
            self.edges.append(edges)
            matrix = torch.eye(size, dtype=COMPLEX)
            for control, target in edges:
                matrix = cnot_matrix(self.nq, control, target) @ matrix
            entanglers.append(matrix)
        self.register_buffer("entanglers", torch.stack(entanglers))

    def rotation(self, angle, pauli):
        return (
            torch.cos(angle / 2)[:, None, None] * self.eye
            - 1j * torch.sin(angle / 2)[:, None, None] * pauli
        )

    def operators(self):
        u = self.eye[None].expand(self.steps, -1, -1)
        for layer in range(self.depth):
            for q in range(self.nq):
                u = self.rotation(
                    self.theta[:, layer, q, 0], self.ys[q]
                ) @ u
                u = self.rotation(
                    self.theta[:, layer, q, 1], self.zs[q]
                ) @ u
            u = self.entanglers[layer] @ u

        # A[t, izlazni_bit, prethodni_bit, mem_out, mem_in].
        # Radni kubit nije resetovan između koraka.
        return torch.stack([
            torch.stack([
                u[:, x * self.dm:(x + 1) * self.dm,
                  b * self.dm:(b + 1) * self.dm]
                for b in (0, 1)
            ], dim=1)
            for x in (0, 1)
        ], dim=1)

    def circuit(self, step):
        qc = QuantumCircuit(self.nq)
        theta = self.theta.detach().cpu().numpy()
        for layer in range(self.depth):
            for q in range(self.nq):
                qc.ry(float(theta[step, layer, q, 0]), q)
                qc.rz(float(theta[step, layer, q, 1]), q)
            for control, target in self.edges[layer]:
                qc.cx(control, target)
        return qc


def single_mass(a, group, k, targets=None, gamma=0.0):
    """
    Zbir masa legalnih putanja, opciono ponderisan kernelom prema targets.

    rho: batch, prethodni_bit, broj_jedinica, mem_out, mem_in.
    Na granici kola zadržava se samo broj_jedinica=k.
    Brojač se vraća na nulu, ali kvantna memorija i radni bit ostaju.
    """
    steps, _, _, d, _ = a.shape
    batch = 1 if targets is None else len(targets)
    rho = torch.zeros((batch, 2, k + 1, d, d), dtype=COMPLEX)
    rho[:, 0, 0, 0, 0] = 1.0

    for t in range(steps):
        outputs = []
        for x in (0, 1):
            value = sum(
                a[t, x, b] @ rho[:, b] @ a[t, x, b].mH
                for b in (0, 1)
            )
            if x:
                value = torch.cat(
                    (torch.zeros_like(value[:, :1]), value[:, :-1]),
                    dim=1,
                )
            if targets is not None:
                weight = torch.exp(
                    -gamma * (targets[:, t].to(REAL) - x).square()
                )
                value = value * weight[:, None, None, None]
            outputs.append(value)

        rho = torch.stack(outputs, dim=1)
        if (t + 1) % group == 0:
            rho = torch.cat(
                (rho[:, :, k:k + 1],
                 torch.zeros_like(rho[:, :, :k])),
                dim=2,
            )

    return rho[:, :, 0].diagonal(dim1=-2, dim2=-1).sum((-1, -2)).real


def double_mass(a, group, k, gamma):
    """
    Neuslovljeni brojilac E_QQ kernela sa uslovom legalnosti obe kopije.

    Dve nezavisne kopije memorije.
    Čuva se i par prethodno izmerenih bitova.
    """
    steps, _, _, d, _ = a.shape
    dd = d * d
    rho = torch.zeros((4, k + 1, k + 1, dd, dd), dtype=COMPLEX)
    rho[0, 0, 0, 0, 0] = 1.0

    for t in range(steps):
        outputs = []
        for x in (0, 1):
            for y in (0, 1):
                value = torch.zeros_like(rho[0])
                for b in (0, 1):
                    for c in (0, 1):
                        operator = torch.kron(
                            a[t, x, b].contiguous(),
                            a[t, y, c].contiguous(),
                        )
                        value = (
                            value
                            + operator @ rho[2 * b + c] @ operator.mH
                        )

                if x:
                    value = torch.cat(
                        (torch.zeros_like(value[:1]), value[:-1]), dim=0
                    )
                if y:
                    value = torch.cat(
                        (torch.zeros_like(value[:, :1]), value[:, :-1]),
                        dim=1,
                    )
                outputs.append(value * math.exp(-gamma * (x - y) ** 2))

        rho = torch.stack(outputs)
        if (t + 1) % group == 0:
            selected = rho[:, k:k + 1, k:k + 1]
            selected = torch.cat(
                (selected,
                 torch.zeros_like(rho[:, :1, :k])),
                dim=2,
            )
            rho = torch.cat(
                (selected, torch.zeros_like(rho[:, :k])),
                dim=1,
            )

    return rho[:, 0, 0].diagonal(dim1=-2, dim2=-1).sum().real


def distance_histogram(data):
    """Svi parovi; blokiranje služi samo ograničenju RAM-a."""
    x = np.asarray(data, dtype=np.float64)
    totals = x.sum(axis=1)
    histogram = np.zeros(2 * x.shape[1] + 1, dtype=np.int64)

    for start in range(0, len(x), 256):
        stop = min(start + 256, len(x))
        distances = (
            totals[start:stop, None]
            + totals[None, :]
            - 2.0 * (x[start:stop] @ x.T)
        )
        distances = np.rint(distances).astype(np.int64)
        require(np.all(distances >= 0), "Negativna Hamming udaljenost.")
        histogram += np.bincount(
            distances.ravel(), minlength=len(histogram)
        )
    return histogram


def kernel_constant(histogram, gamma):
    distances = np.arange(len(histogram), dtype=np.float64)
    return float(
        np.dot(histogram, np.exp(-gamma * distances)) / histogram.sum()
    )


def bandwidth_from_training(histogram):
    midpoint = (int(histogram.sum()) + 1) // 2
    median = int(np.searchsorted(np.cumsum(histogram), midpoint))
    # Deterministički granični slučaj identičnih uzoraka.
    median = max(median, 1)
    return math.log(2.0) / median, median


def mmd_backward(model, data, gamma, pp, group, k):
    """
    Tačan full-batch gradijent, uz blokiranu obradu istorijskih uzoraka.

    Z je zaseban autograd list da isti graf normalizacije ne mora
    ostati živ tokom svih blokova. Na kraju se primeni lančano pravilo.
    """
    model.zero_grad(set_to_none=True)
    graph_a = model.operators()
    a = graph_a.detach().requires_grad_(True)

    z_graph = single_mass(a, group, k)[0]
    require(
        torch.isfinite(z_graph).item() and float(z_graph.detach()) > 0,
        "Neispravna normalizacija zajedničke raspodele."
    )
    z = z_graph.detach().requires_grad_(True)

    qq = double_mass(a, group, k, gamma) / z.square()
    require(torch.isfinite(qq).item(), "Neispravan E_QQ.")
    qq.backward()
    value = float(qq.detach()) + pp

    count = len(data)
    for start in range(0, count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, count)
        cross = single_mass(
            a, group, k, data[start:stop], gamma
        )
        term = -2.0 * cross.sum() / (count * z)
        require(torch.isfinite(term).item(), "Neispravan E_QP.")
        term.backward()
        value += float(term.detach())

    require(z.grad is not None, "Nedostaje gradijent normalizacije.")
    z_graph.backward(z.grad)
    require(
        a.grad is not None and torch.all(torch.isfinite(a.grad)).item(),
        "Neispravan gradijent kvantnih operatora."
    )
    graph_a.backward(a.grad)

    require(value >= -1e-8, f"MMD² je negativan: {value}.")
    return value


@torch.no_grad()
def evaluate_mmd(model, data, gamma, pp, group, k):
    a = model.operators()
    z = single_mass(a, group, k)[0]
    require(float(z) > 0, "Nulta normalizacija.")
    qq = double_mass(a, group, k, gamma) / z.square()

    cross_total = 0.0
    for start in range(0, len(data), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(data))
        cross_total += float(
            single_mass(a, group, k, data[start:stop], gamma).sum()
        )

    value = float(qq) - 2.0 * cross_total / (len(data) * float(z)) + pp
    require(np.isfinite(value) and value >= -1e-8, "Neispravan MMD².")
    return value


def update(model, optimizer, data, gamma, pp, regularization, group, k):
    value = mmd_backward(model, data, gamma, pp, group, k)
    if regularization:
        (regularization * model.theta.square().mean()).backward()

    require(
        model.theta.grad is not None
        and torch.all(torch.isfinite(model.theta.grad)).item(),
        "Neispravan gradijent kapija."
    )
    optimizer.step()
    require(torch.all(torch.isfinite(model.theta)).item(),
            "Neispravni parametri nakon optimizacije.")
    return value


def fit(data, gamma, pp, regularization, epochs, label):
    model = HQCS(2 * NUMBERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = update(
            model, optimizer, data, gamma, pp,
            regularization, NUMBERS, PICKS,
        )
        print(
            f"  {label}: epoha {epoch}/{epochs}, "
            f"MMD² pre koraka={loss:.9g}, "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return model


def choose_parameters(train, validation, gamma, pp_train, pp_validation):
    best = None
    for regularization in REGULARIZATIONS:
        model = HQCS(2 * NUMBERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, MAX_EPOCHS + 1):
            started = time.perf_counter()
            loss = update(
                model, optimizer, train, gamma, pp_train,
                regularization, NUMBERS, PICKS,
            )
            print(
                f"  Izbor: reg={regularization:g}, "
                f"epoha={epoch}/{MAX_EPOCHS}, train MMD²={loss:.9g}, "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )

            if epoch % CHECK_EVERY:
                continue

            score = evaluate_mmd(
                model, validation, gamma, pp_validation, NUMBERS, PICKS
            )
            print(f"  VALIDACIJA MMD²={score:.9g}", flush=True)
            candidate = (score, epoch, regularization)
            if best is None or candidate < best:
                best = candidate

    require(best is not None, "Parametri nisu izabrani.")
    return best[2], best[1], best[0]


def initial_memory(d):
    state = np.zeros(d, dtype=np.complex128)
    state[0] = 1.0
    return state


def encode_prefix(a, bits):
    state = initial_memory(a.shape[-1])
    previous = 0
    for t, bit in enumerate(bits):
        bit = int(bit)
        state = a[t, bit, previous] @ state
        norm = np.linalg.norm(state)
        require(norm > 0 and np.isfinite(norm), "Nulta masa konteksta.")
        state /= norm
        previous = bit
    return state, previous


def path_mass(a, bits, state=None, previous=0):
    if state is None:
        state = initial_memory(a.shape[-1])
    else:
        state = state.copy()

    for t, bit in enumerate(bits):
        bit = int(bit)
        state = a[t, bit, previous] @ state
        previous = bit
    return float(np.vdot(state, state).real)


def suffix_effects(a, k):
    n, _, _, d, _ = a.shape
    effects = np.zeros((n + 1, k + 1, 2, d, d), dtype=np.complex128)
    for previous in (0, 1):
        effects[n, 0, previous] = np.eye(d)

    for t in reversed(range(n)):
        for r in range(k + 1):
            for previous in (0, 1):
                a0 = a[t, 0, previous]
                value = a0.conj().T @ effects[t + 1, r, 0] @ a0
                if r:
                    a1 = a[t, 1, previous]
                    value += (
                        a1.conj().T @ effects[t + 1, r - 1, 1] @ a1
                    )
                effects[t, r, previous] = (value + value.conj().T) / 2
    return effects


def audit_next(a, state, previous, k):
    n, _, _, d, _ = a.shape
    effects = suffix_effects(a, k)
    z = float(np.vdot(state, effects[0, k, previous] @ state).real)
    require(z > 0 and np.isfinite(z), "Nulta masa legalnog nastavka.")

    rho = np.zeros((n + 1, 2, k + 1, d, d), dtype=np.complex128)
    rho[0, previous, 0] = np.outer(state, state.conj())
    marginals = np.zeros(n)

    for t in range(n):
        for b in (0, 1):
            for count in range(k + 1):
                for x in (0, 1):
                    if count + x > k:
                        continue
                    operator = a[t, x, b]
                    density = operator @ rho[t, b, count] @ operator.conj().T
                    rho[t + 1, x, count + x] += density

                    if x:
                        remaining = k - count - 1
                        marginals[t] += float(np.trace(
                            effects[t + 1, remaining, 1] @ density
                        ).real) / z

    z_forward = float(sum(np.trace(rho[n, b, k]).real for b in (0, 1)))
    np.testing.assert_allclose(z, z_forward, rtol=2e-9, atol=1e-14)
    np.testing.assert_allclose(marginals.sum(), k, atol=2e-8, rtol=0)
    require(
        np.all(marginals >= -1e-9) and np.all(marginals <= 1 + 1e-9),
        "Neispravne marginale."
    )
    return effects, z, marginals


@njit(cache=False)
def mv(matrix, vector):
    d = len(vector)
    result = np.zeros(d, dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            result[i] += matrix[i, j] * vector[j]
    return result


@njit(cache=False)
def quadratic(matrix, vector):
    product = mv(matrix, vector)
    value = 0.0
    for i in range(len(vector)):
        value += (vector[i].conjugate() * product[i]).real
    return value


@njit(cache=False)
def search_maximum(a, effects, state, previous, k):
    """
    Pretraga celog legalnog prostora uz dokazivu matematičku granicu:
    masa jednog lista ne može preći zbir masa svih nastavaka čvora.

    Računanje je numeričko; koristi se konzervativna tolerancija.
    Nema ograničenja broja kandidata i nema beam pretrage.
    """
    n = len(a)
    d = len(state)
    capacity = 2 * n + 4
    states = np.zeros((capacity, d), dtype=np.complex128)
    positions = np.zeros(capacity, dtype=np.int64)
    remaining = np.zeros(capacity, dtype=np.int64)
    previous_bits = np.zeros(capacity, dtype=np.int64)
    masks = np.zeros(capacity, dtype=np.int64)
    bounds = np.zeros(capacity)

    states[0] = state
    remaining[0] = k
    previous_bits[0] = previous
    bounds[0] = quadratic(effects[0, k, previous], state)
    tolerance = abs(bounds[0]) * 1e-10
    top = 1

    best = -1.0
    best_mask = np.int64(0)
    visited = 0

    while top:
        top -= 1
        t = positions[top]
        r = remaining[top]
        b = previous_bits[top]
        mask = masks[top]
        bound = bounds[top]
        vector = states[top].copy()
        visited += 1

        if best >= 0 and bound + tolerance < best:
            continue

        if t == n:
            if r == 0:
                mass = 0.0
                for j in range(d):
                    mass += (vector[j].conjugate() * vector[j]).real
                if mass > best or (mass == best and mask < best_mask):
                    best = mass
                    best_mask = mask
            continue

        children = np.zeros((2, d), dtype=np.complex128)
        child_bounds = np.zeros(2)
        valid = np.zeros(2, dtype=np.int64)

        for x in (0, 1):
            need = r - x
            if 0 <= need <= n - t - 1:
                children[x] = mv(a[t, x, b], vector)
                child_bounds[x] = quadratic(
                    effects[t + 1, need, x], children[x]
                )
                valid[x] = 1

        first = 0 if child_bounds[0] <= child_bounds[1] else 1
        for order in (0, 1):
            x = first if order == 0 else 1 - first
            if not valid[x]:
                continue
            states[top] = children[x]
            positions[top] = t + 1
            remaining[top] = r - x
            previous_bits[top] = x
            masks[top] = mask | (np.int64(x) << t)
            bounds[top] = child_bounds[x]
            top += 1

    return best_mask, best, visited


def mask_to_bits(mask, length):
    return np.asarray(
        [(int(mask) >> i) & 1 for i in range(length)], dtype=np.int64
    )


@torch.no_grad()
def qiskit_check(model):
    a = model.operators().cpu().numpy()
    independent = np.empty_like(a)
    identity = np.eye(model.dm)

    for t in range(model.steps):
        unitary = np.asarray(Operator(model.circuit(t)).data)
        for x in (0, 1):
            for b in (0, 1):
                independent[t, x, b] = unitary[
                    x * model.dm:(x + 1) * model.dm,
                    b * model.dm:(b + 1) * model.dm,
                ]
        for b in (0, 1):
            complete = sum(
                a[t, x, b].conj().T @ a[t, x, b] for x in (0, 1)
            )
            np.testing.assert_allclose(
                complete, identity, atol=3e-12, rtol=0
            )

    np.testing.assert_allclose(a, independent, atol=3e-12, rtol=0)
    return independent


def self_checks():
    print("Provere malog problema: Qiskit, MMD, gradijenti, normalizacija, MAP.",
          flush=True)

    group, k, gamma = 3, 1, 0.23
    model = HQCS(2 * group, memory=1, depth=3)
    a_np = qiskit_check(model)

    all_bits = np.asarray([
        mask_to_bits(mask, model.steps)
        for mask in range(1 << model.steps)
    ])
    raw = np.asarray([path_mass(a_np, row) for row in all_bits])
    np.testing.assert_allclose(raw.sum(), 1.0, atol=3e-12)

    # Odloženo merenje: stari radni kubiti čuvaju zapise.
    # CNOT prenosi prethodni bit u novi radni kubit pre sledećeg U.
    qc = QuantumCircuit(model.memory + model.steps)
    for t in range(model.steps):
        work = model.memory + t
        if t:
            qc.cx(work - 1, work)
        qc.compose(
            model.circuit(t),
            qubits=list(range(model.memory)) + [work],
            inplace=True,
        )
    amplitudes = np.asarray(Statevector.from_instruction(qc).data)
    independent_raw = (
        np.abs(amplitudes.reshape(1 << model.steps, model.dm)) ** 2
    ).sum(axis=1)
    np.testing.assert_allclose(raw, independent_raw, atol=3e-12, rtol=3e-11)

    legal = (
        (all_bits[:, :group].sum(1) == k)
        & (all_bits[:, group:].sum(1) == k)
    )
    legal_bits = all_bits[legal]
    legal_raw = raw[legal]
    probability = legal_raw / legal_raw.sum()

    distances = np.abs(
        legal_bits[:, None, :] - legal_bits[None, :, :]
    ).sum(2)
    gram = np.exp(-gamma * distances)
    require(np.linalg.eigvalsh(gram).min() > -1e-10, "Kernel nije PSD.")

    selected = [0, 2, 4, 8]
    data = torch.tensor(legal_bits[selected], dtype=torch.long)
    pp = float(gram[np.ix_(selected, selected)].mean())

    a = model.operators()
    z = single_mass(a, group, k)[0]
    qq = double_mass(a, group, k, gamma) / z.square()
    qp = single_mass(a, group, k, data, gamma) / z

    np.testing.assert_allclose(
        float(z.detach()), legal_raw.sum(), atol=3e-12, rtol=3e-10
    )
    np.testing.assert_allclose(
        float(qq.detach()), probability @ gram @ probability,
        atol=3e-11, rtol=3e-9
    )
    np.testing.assert_allclose(
        qp.detach().numpy(), probability @ gram[:, selected],
        atol=3e-11, rtol=3e-9
    )

    direct_loss = qq - 2 * qp.mean() + pp
    direct_loss.backward()
    expected_gradient = model.theta.grad.detach().clone()

    split_value = mmd_backward(model, data, gamma, pp, group, k)
    np.testing.assert_allclose(
        split_value, float(direct_loss.detach()), atol=3e-11, rtol=3e-9
    )
    np.testing.assert_allclose(
        model.theta.grad.numpy(), expected_gradient.numpy(),
        atol=3e-10, rtol=3e-7
    )
    require(float(expected_gradient.abs().max()) > 1e-10,
            "Degenerisani gradijenti.")

    index = (2, 1, 1, 0)
    original = float(model.theta[index].detach())
    eps = 1e-6
    finite = []
    for delta in (eps, -eps):
        with torch.no_grad():
            model.theta[index] = original + delta
        finite.append(evaluate_mmd(model, data, gamma, pp, group, k))
    with torch.no_grad():
        model.theta[index] = original
    np.testing.assert_allclose(
        float(expected_gradient[index]),
        (finite[0] - finite[1]) / (2 * eps),
        atol=3e-7, rtol=5e-4,
    )

    # Provera legalne normalizacije i maksimuma za k=0,...,group.
    prefix = legal_bits[0, :group]
    state, previous = encode_prefix(a_np[:group], prefix)
    output_a = a_np[group:]

    for count in range(group + 1):
        effects, normalizer, _ = audit_next(output_a, state, previous, count)
        candidates = [
            mask_to_bits(mask, group)
            for mask in range(1 << group)
            if mask_to_bits(mask, group).sum() == count
        ]
        masses = np.array([
            path_mass(output_a, row, state, previous) for row in candidates
        ])
        mask, maximum, _ = search_maximum(
            output_a, effects, state, previous, count
        )
        require(mask_to_bits(mask, group).sum() == count, "Nelegalni MAP.")
        np.testing.assert_allclose(normalizer, masses.sum(), atol=3e-12)
        np.testing.assert_allclose(maximum, masses.max(), atol=3e-12)

    print("Sve ugrađene provere prošle.", flush=True)


def conditional_test_nll(model, sequences):
    """Parametri ostaju fiksni; prethodno stvarno kolo je poznat kontekst."""
    with torch.no_grad():
        a = model.operators().cpu().numpy()
    first, second = a[:NUMBERS], a[NUMBERS:]
    effects = suffix_effects(second, PICKS)
    losses = []

    for sequence in sequences:
        state, previous = encode_prefix(first, sequence[:NUMBERS])
        z = float(np.vdot(state, effects[0, PICKS, previous] @ state).real)
        mass = path_mass(second, sequence[NUMBERS:], state, previous)
        require(
            z > 0 and mass > 0 and np.isfinite(z) and np.isfinite(mass),
            "Neispravna test uslovna masa."
        )
        losses.append(-math.log(mass / z))
    return float(np.mean(losses))


def longest_run(numbers):
    maximum = current = 1
    for left, right in zip(numbers[:-1], numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        maximum = max(maximum, current)
    return maximum


def diagnostics(first_a, output_a, bits, draws, combination, marginals):
    messages = []
    run = longest_run(combination)
    if run >= 5:
        messages.append(
            f"Niz od {run} uzastopnih brojeva zahteva proveru modela."
        )

    spans = draws[:, -1] - draws[:, 0]
    span = combination[-1] - combination[0]
    if span <= float(np.quantile(spans, 0.01)):
        messages.append(
            f"Raspon {span} je u donjih 1% istorijskih raspona: "
            "izrazito grupisanje."
        )

    repetitions = int(np.all(draws == np.asarray(combination), axis=1).sum())
    if repetitions:
        messages.append(
            f"Kombinacija postoji {repetitions} puta u istoriji; "
            "proveriti memorisanje obrasca."
        )

    sensitivity = 0.0
    for row in bits[max(0, len(bits) - 9):-1]:
        state, previous = encode_prefix(first_a, row)
        _, _, other = audit_next(output_a, state, previous, PICKS)
        sensitivity = max(sensitivity, float(np.max(np.abs(other - marginals))))

    print(f"  Osetljivost na promenu prethodnog kola: {sensitivity:.9g}",
          flush=True)
    if sensitivity < 1e-6:
        messages.append(
            "Raspodela gotovo ne reaguje na promenu konteksta: "
            "moguće ignorisanje istorijskog uslova."
        )

    for message in messages:
        print("  PROVERA MODELA: " + message, flush=True)
    if not messages:
        print(
            "  Ove dijagnostike nisu označile problem; "
            "to nije potvrda prediktivne vrednosti.",
            flush=True,
        )
    print("  Dijagnostike ne menjaju izabranu kombinaciju.", flush=True)


def process(path):
    print("\n" + "=" * 76, flush=True)
    print(Path(path).name, flush=True)
    draws, bits, sequences = read_draws(path)

    total = len(sequences)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_np = sequences[:train_end]
    validation_np = sequences[train_end:validation_end]
    test_np = sequences[validation_end:]
    trainval_np = sequences[:validation_end]

    train = torch.tensor(train_np, dtype=torch.long)
    validation = torch.tensor(validation_np, dtype=torch.long)
    test = torch.tensor(test_np, dtype=torch.long)
    trainval = torch.tensor(trainval_np, dtype=torch.long)
    full = torch.tensor(sequences, dtype=torch.long)

    print(
        f"Kola={len(draws)}, svi prelazi={total}; "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}.",
        flush=True,
    )
    print("Računam kernel statistiku iz treninga, bez uzorkovanja...",
          flush=True)

    train_hist = distance_histogram(train_np)
    gamma, median = bandwidth_from_training(train_hist)
    pp_train = kernel_constant(train_hist, gamma)
    pp_validation = kernel_constant(distance_histogram(validation_np), gamma)

    print(
        f"Kernel: medijana train Hamming udaljenosti={median}; "
        f"gamma={gamma:.10g}. Ostaje fiksan u svim fazama.",
        flush=True,
    )

    regularization, epochs, validation_score = choose_parameters(
        train, validation, gamma, pp_train, pp_validation
    )
    print(
        f"IZABRANO: reg={regularization:g}, epohe={epochs}, "
        f"validation MMD²={validation_score:.9g}",
        flush=True,
    )

    pp_trainval = kernel_constant(distance_histogram(trainval_np), gamma)
    test_model = fit(
        trainval, gamma, pp_trainval, regularization, epochs, "TRAIN+VALIDATION"
    )

    # Test se koristi tek posle završetka izbora parametara.
    pp_test = kernel_constant(distance_histogram(test_np), gamma)
    test_mmd = evaluate_mmd(
        test_model, test, gamma, pp_test, NUMBERS, PICKS
    )
    test_nll = conditional_test_nll(test_model, test_np)

    print(
        f"ODVOJENI TEST: MMD² parova={test_mmd:.9g}; "
        f"uslovni NLL sledećeg kola={test_nll:.8f}.",
        flush=True,
    )
    print(
        "To su metrike raspodele, ne procenat budućih pogodaka.",
        flush=True,
    )
    del test_model

    pp_full = kernel_constant(distance_histogram(sequences), gamma)
    final_model = fit(
        full, gamma, pp_full, regularization, epochs, "CELA ISTORIJA"
    )

    independent_a = qiskit_check(final_model)
    with torch.no_grad():
        a = final_model.operators().cpu().numpy()

    first_a, output_a = a[:NUMBERS], a[NUMBERS:]
    state, previous = encode_prefix(first_a, bits[-1])
    effects, z, marginals = audit_next(output_a, state, previous, PICKS)

    print(
        f"Tražim maksimum među svih {math.comb(NUMBERS, PICKS):,} "
        "legalnih kombinacija. Pretraga može potrajati.",
        flush=True,
    )
    started = time.perf_counter()
    mask, maximum, visited = search_maximum(
        output_a, effects, state, previous, PICKS
    )
    elapsed = time.perf_counter() - started

    output_bits = mask_to_bits(mask, NUMBERS)
    combination = (np.flatnonzero(output_bits) + 1).tolist()
    require(
        len(combination) == PICKS
        and len(set(combination)) == PICKS
        and all(1 <= number <= NUMBERS for number in combination),
        "Nelegalni NEXT."
    )

    direct = path_mass(output_a, output_bits, state, previous)
    independent_state, independent_previous = encode_prefix(
        independent_a[:NUMBERS], bits[-1]
    )
    independent_mass = path_mass(
        independent_a[NUMBERS:], output_bits,
        independent_state, independent_previous,
    )
    np.testing.assert_allclose(maximum, direct, rtol=3e-9, atol=1e-15)
    np.testing.assert_allclose(
        maximum, independent_mass, rtol=3e-8, atol=1e-15
    )
    model_mass = maximum / z
    require(
        np.isfinite(model_mass) and 0 < model_mass <= 1 + 1e-10,
        "Neispravna normalizovana masa maksimuma."
    )

    print(
        f"Normalizacija nastavka Z={z:.12g}; "
        f"zbir marginala={marginals.sum():.12f}.",
        flush=True,
    )
    print(
        f"Pretraga: {elapsed:.1f}s, posećeni čvorovi={visited:,}.",
        flush=True,
    )
    diagnostics(first_a, output_a, bits, draws, combination, marginals)

    print(
        f"\nNEXT [{Path(path).name}]: "
        + " ".join(f"{number:02d}" for number in combination),
        flush=True,
    )
    print(
        f"Masa u naučenoj uslovnoj raspodeli: {model_mass:.12g}.\n"
        "To je interna modelska masa, ne potvrđena verovatnoća izvlačenja.",
        flush=True,
    )

    return {
        "file": Path(path).name,
        "next": tuple(combination),
        "input": signature(bits),
        "model": signature(a),
        "marginals": marginals,
    }


def main():
    print(
        "HQCS Loto 7/39\n"
        "Deterministička kvantna simulacija; klasični Adam i MAP pretraga.\n"
        "Nema slučajnog uzorkovanja, dopune brojeva ni kozmetičkih filtera.\n"
        "Svaki CSV se uči zasebno.",
        flush=True,
    )
    for path in PATHS:
        require(Path(path).is_file(), f"Nedostaje CSV: {path}")

    self_checks()
    results = [process(path) for path in PATHS]

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            left, right = results[i], results[j]
            if left["next"] != right["next"]:
                continue
            difference = float(np.max(np.abs(
                left["marginals"] - right["marginals"]
            )))
            print(
                "\nPROVERA ISTOG NEXT-a:\n"
                f"  {left['file']}\n"
                f"  {right['file']}\n"
                f"  Ulazi: {left['input']} / {right['input']}\n"
                f"  Modeli: {left['model']} / {right['model']}\n"
                f"  Maksimalna razlika marginala: {difference:.10g}\n"
                "  Podudaranje zahteva proveru zajedničkog maksimuma "
                "ili degeneracije. Kombinacije nisu veštački razdvojene.",
                flush=True,
            )

    print("\nObrada završena.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena pretraga nije potvrđen maksimum.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception:
        print(
            "\nObrada je zaustavljena zbog greške. "
            "Zamenska kombinacija se ne generiše.",
            file=sys.stderr,
        )
        raise



"""
HQCS Loto 7/39
Deterministička kvantna simulacija; klasični Adam i MAP pretraga.
Nema slučajnog uzorkovanja, dopune brojeva ni kozmetičkih filtera.
Svaki CSV se uči zasebno.
Provere: Qiskit, MMD, gradijenti, normalizacija, MAP.
Sve ugrađene provere prošle.

============================================================================
loto7_4706_k84.csv
Kola=4706, svi prelazi=4705; train=3293, validation=706, test=706.
Računam kernel statistiku iz treninga, bez uzorkovanja...
Kernel: medijana train Hamming udaljenosti=24; gamma=0.02888113252. Ostaje fiksan u svim fazama.
  Izbor: reg=0, epoha=1/30, train MMD²=0.0018305524, 4.9s
  Izbor: reg=0, epoha=2/30, train MMD²=0.00295505294, 4.7s
  Izbor: reg=0, epoha=3/30, train MMD²=0.00138216351, 4.8s
  Izbor: reg=0, epoha=4/30, train MMD²=0.00128715634, 4.7s
  Izbor: reg=0, epoha=5/30, train MMD²=0.00145724421, 4.7s
  VALIDACIJA MMD²=0.00182993171
  Izbor: reg=0, epoha=6/30, train MMD²=0.00115808237, 4.7s
  Izbor: reg=0, epoha=7/30, train MMD²=0.000728492445, 4.9s
  Izbor: reg=0, epoha=8/30, train MMD²=0.000581593172, 4.7s
  Izbor: reg=0, epoha=9/30, train MMD²=0.000675755042, 4.7s
  Izbor: reg=0, epoha=10/30, train MMD²=0.00076563847, 4.8s
  VALIDACIJA MMD²=0.00134890334
  Izbor: reg=0, epoha=11/30, train MMD²=0.000749238038, 4.8s
  Izbor: reg=0, epoha=12/30, train MMD²=0.000670648054, 4.7s
  Izbor: reg=0, epoha=13/30, train MMD²=0.000593006589, 4.7s
  Izbor: reg=0, epoha=14/30, train MMD²=0.000540776863, 4.7s
  Izbor: reg=0, epoha=15/30, train MMD²=0.000502416528, 4.7s
  VALIDACIJA MMD²=0.00110111079
  Izbor: reg=0, epoha=16/30, train MMD²=0.000461803455, 4.6s
  Izbor: reg=0, epoha=17/30, train MMD²=0.000425399176, 4.7s
  Izbor: reg=0, epoha=18/30, train MMD²=0.00040831693, 4.7s
  Izbor: reg=0, epoha=19/30, train MMD²=0.000408671937, 4.7s
  Izbor: reg=0, epoha=20/30, train MMD²=0.000405437445, 4.7s
  VALIDACIJA MMD²=0.00105397743
  Izbor: reg=0, epoha=21/30, train MMD²=0.000382567387, 4.7s
  Izbor: reg=0, epoha=22/30, train MMD²=0.000349581619, 4.7s
  Izbor: reg=0, epoha=23/30, train MMD²=0.000327677974, 4.7s
  Izbor: reg=0, epoha=24/30, train MMD²=0.000323465223, 4.6s
  Izbor: reg=0, epoha=25/30, train MMD²=0.000325997573, 4.7s
  VALIDACIJA MMD²=0.000962843829
  Izbor: reg=0, epoha=26/30, train MMD²=0.000322091225, 4.8s
  Izbor: reg=0, epoha=27/30, train MMD²=0.000308790586, 4.7s
  Izbor: reg=0, epoha=28/30, train MMD²=0.000291924254, 4.7s
  Izbor: reg=0, epoha=29/30, train MMD²=0.000278455047, 4.6s
  Izbor: reg=0, epoha=30/30, train MMD²=0.000271237669, 4.6s
  VALIDACIJA MMD²=0.00100090364
  Izbor: reg=0.0001, epoha=1/30, train MMD²=0.0018305524, 4.7s
  Izbor: reg=0.0001, epoha=2/30, train MMD²=0.00295526275, 4.7s
  Izbor: reg=0.0001, epoha=3/30, train MMD²=0.00138221405, 4.6s
  Izbor: reg=0.0001, epoha=4/30, train MMD²=0.00128608011, 4.6s
  Izbor: reg=0.0001, epoha=5/30, train MMD²=0.00145646737, 4.6s
  VALIDACIJA MMD²=0.00183190136
  Izbor: reg=0.0001, epoha=6/30, train MMD²=0.00115896259, 4.7s
  Izbor: reg=0.0001, epoha=7/30, train MMD²=0.000729406241, 4.6s
  Izbor: reg=0.0001, epoha=8/30, train MMD²=0.00058127626, 4.6s
  Izbor: reg=0.0001, epoha=9/30, train MMD²=0.000674222348, 4.6s
  Izbor: reg=0.0001, epoha=10/30, train MMD²=0.000764053963, 4.6s
  VALIDACIJA MMD²=0.00134749584
  Izbor: reg=0.0001, epoha=11/30, train MMD²=0.000748870402, 4.6s
  Izbor: reg=0.0001, epoha=12/30, train MMD²=0.00067145621, 4.6s
  Izbor: reg=0.0001, epoha=13/30, train MMD²=0.000593579185, 4.7s
  Izbor: reg=0.0001, epoha=14/30, train MMD²=0.00054008308, 4.7s
  Izbor: reg=0.0001, epoha=15/30, train MMD²=0.00050079706, 4.7s
  VALIDACIJA MMD²=0.00110034775
  Izbor: reg=0.0001, epoha=16/30, train MMD²=0.000460236594, 4.8s
  Izbor: reg=0.0001, epoha=17/30, train MMD²=0.000424400989, 4.8s
  Izbor: reg=0.0001, epoha=18/30, train MMD²=0.000407812828, 4.8s
  Izbor: reg=0.0001, epoha=19/30, train MMD²=0.000408354789, 4.7s
  Izbor: reg=0.0001, epoha=20/30, train MMD²=0.000404947024, 4.6s
  VALIDACIJA MMD²=0.00105373893
  Izbor: reg=0.0001, epoha=21/30, train MMD²=0.000381735607, 4.6s
  Izbor: reg=0.0001, epoha=22/30, train MMD²=0.000348511088, 4.6s
  Izbor: reg=0.0001, epoha=23/30, train MMD²=0.000326528237, 4.6s
  Izbor: reg=0.0001, epoha=24/30, train MMD²=0.000322315998, 4.6s
  Izbor: reg=0.0001, epoha=25/30, train MMD²=0.000324902918, 4.6s
  VALIDACIJA MMD²=0.000961468956
  Izbor: reg=0.0001, epoha=26/30, train MMD²=0.000321127992, 4.6s
  Izbor: reg=0.0001, epoha=27/30, train MMD²=0.00030798962, 4.7s
  Izbor: reg=0.0001, epoha=28/30, train MMD²=0.000291183125, 4.8s
  Izbor: reg=0.0001, epoha=29/30, train MMD²=0.000277598924, 4.7s
  Izbor: reg=0.0001, epoha=30/30, train MMD²=0.000270176744, 4.7s
  VALIDACIJA MMD²=0.000999706362
IZABRANO: reg=0.0001, epohe=25, validation MMD²=0.000961468956
  TRAIN+VALIDATION: epoha 1/25, MMD² pre koraka=0.00183451212, 5.7s
  TRAIN+VALIDATION: epoha 2/25, MMD² pre koraka=0.00285676318, 5.7s
  TRAIN+VALIDATION: epoha 3/25, MMD² pre koraka=0.00141080862, 5.6s
  TRAIN+VALIDATION: epoha 4/25, MMD² pre koraka=0.00123039007, 5.6s
  TRAIN+VALIDATION: epoha 5/25, MMD² pre koraka=0.00135891367, 5.6s
  TRAIN+VALIDATION: epoha 6/25, MMD² pre koraka=0.00107909173, 5.8s
  TRAIN+VALIDATION: epoha 7/25, MMD² pre koraka=0.000713134745, 5.9s
  TRAIN+VALIDATION: epoha 8/25, MMD² pre koraka=0.000596441588, 5.7s
  TRAIN+VALIDATION: epoha 9/25, MMD² pre koraka=0.000669015384, 5.7s
  TRAIN+VALIDATION: epoha 10/25, MMD² pre koraka=0.000745427902, 5.8s
  TRAIN+VALIDATION: epoha 11/25, MMD² pre koraka=0.000742947807, 5.7s
  TRAIN+VALIDATION: epoha 12/25, MMD² pre koraka=0.000676508968, 5.8s
  TRAIN+VALIDATION: epoha 13/25, MMD² pre koraka=0.000588380538, 5.7s
  TRAIN+VALIDATION: epoha 14/25, MMD² pre koraka=0.00051494263, 5.6s
  TRAIN+VALIDATION: epoha 15/25, MMD² pre koraka=0.000467411897, 5.6s
  TRAIN+VALIDATION: epoha 16/25, MMD² pre koraka=0.000436301052, 5.8s
  TRAIN+VALIDATION: epoha 17/25, MMD² pre koraka=0.00041340836, 5.8s
  TRAIN+VALIDATION: epoha 18/25, MMD² pre koraka=0.000402943269, 5.6s
  TRAIN+VALIDATION: epoha 19/25, MMD² pre koraka=0.000404565272, 5.7s
  TRAIN+VALIDATION: epoha 20/25, MMD² pre koraka=0.000402264597, 5.6s
  TRAIN+VALIDATION: epoha 21/25, MMD² pre koraka=0.000382912748, 5.7s
  TRAIN+VALIDATION: epoha 22/25, MMD² pre koraka=0.000351027788, 5.8s
  TRAIN+VALIDATION: epoha 23/25, MMD² pre koraka=0.000323025622, 5.9s
  TRAIN+VALIDATION: epoha 24/25, MMD² pre koraka=0.00031150539, 5.6s
  TRAIN+VALIDATION: epoha 25/25, MMD² pre koraka=0.000313339328, 5.7s
ODVOJENI TEST: MMD² parova=0.000900948631; uslovni NLL sledećeg kola=19.68106529.
To su metrike raspodele, ne procenat budućih pogodaka.
  CELA ISTORIJA: epoha 1/25, MMD² pre koraka=0.00183064162, 6.4s
  CELA ISTORIJA: epoha 2/25, MMD² pre koraka=0.00271006882, 6.3s
  CELA ISTORIJA: epoha 3/25, MMD² pre koraka=0.00143050311, 6.5s
  CELA ISTORIJA: epoha 4/25, MMD² pre koraka=0.00124407837, 6.4s
  CELA ISTORIJA: epoha 5/25, MMD² pre koraka=0.00131208353, 6.4s
  CELA ISTORIJA: epoha 6/25, MMD² pre koraka=0.00100906748, 6.5s
  CELA ISTORIJA: epoha 7/25, MMD² pre koraka=0.000637267003, 6.3s
  CELA ISTORIJA: epoha 8/25, MMD² pre koraka=0.000538315972, 6.4s
  CELA ISTORIJA: epoha 9/25, MMD² pre koraka=0.000655915106, 6.4s
  CELA ISTORIJA: epoha 10/25, MMD² pre koraka=0.000758870917, 6.4s
  CELA ISTORIJA: epoha 11/25, MMD² pre koraka=0.000745065286, 6.4s
  CELA ISTORIJA: epoha 12/25, MMD² pre koraka=0.00065376975, 6.4s
  CELA ISTORIJA: epoha 13/25, MMD² pre koraka=0.000557317684, 6.3s
  CELA ISTORIJA: epoha 14/25, MMD² pre koraka=0.000491269038, 6.4s
  CELA ISTORIJA: epoha 15/25, MMD² pre koraka=0.000451077428, 6.3s
  CELA ISTORIJA: epoha 16/25, MMD² pre koraka=0.000423154489, 6.4s
  CELA ISTORIJA: epoha 17/25, MMD² pre koraka=0.000401647088, 6.4s
  CELA ISTORIJA: epoha 18/25, MMD² pre koraka=0.000389495412, 6.4s
  CELA ISTORIJA: epoha 19/25, MMD² pre koraka=0.000389044668, 6.4s
  CELA ISTORIJA: epoha 20/25, MMD² pre koraka=0.000389802027, 6.3s
  CELA ISTORIJA: epoha 21/25, MMD² pre koraka=0.000376715523, 6.4s
  CELA ISTORIJA: epoha 22/25, MMD² pre koraka=0.000349502188, 6.4s
  CELA ISTORIJA: epoha 23/25, MMD² pre koraka=0.00032181377, 6.3s
  CELA ISTORIJA: epoha 24/25, MMD² pre koraka=0.000304149229, 6.4s
  CELA ISTORIJA: epoha 25/25, MMD² pre koraka=0.000295863577, 6.3s
Tražim maksimum među svih 15,380,937 legalnih kombinacija. Pretraga može potrajati.
Normalizacija nastavka Z=0.000425611270659; zbir marginala=7.000000000000.
Pretraga: 0.0s, posećeni čvorovi=66,284.
  Osetljivost na promenu prethodnog kola: 0.197460002
  Ove dijagnostike nisu označile problem; to nije potvrda prediktivne vrednosti.
  Dijagnostike ne menjaju izabranu kombinaciju.

NEXT [loto7_4706_k84.csv]: 19 22 25 28 31 34 37
Masa u naučenoj uslovnoj raspodeli: 7.29951926661e-05.
To je interna modelska masa, ne potvrđena verovatnoća izvlačenja.



============================================================================
loto7_4706_k84_loto_2975.csv
Kola=2975, svi prelazi=2974; train=2081, validation=446, test=447.
Računam kernel statistiku iz treninga, bez uzorkovanja...
Kernel: medijana train Hamming udaljenosti=24; gamma=0.02888113252. Ostaje fiksan u svim fazama.
  Izbor: reg=0, epoha=1/30, train MMD²=0.00201570903, 3.0s
  Izbor: reg=0, epoha=2/30, train MMD²=0.00288954961, 3.0s
  Izbor: reg=0, epoha=3/30, train MMD²=0.00145866785, 3.0s
  Izbor: reg=0, epoha=4/30, train MMD²=0.00145616207, 3.0s
  Izbor: reg=0, epoha=5/30, train MMD²=0.00146214123, 3.0s
  VALIDACIJA MMD²=0.00218608522
  Izbor: reg=0, epoha=6/30, train MMD²=0.00113246725, 3.0s
  Izbor: reg=0, epoha=7/30, train MMD²=0.000774508879, 3.0s
  Izbor: reg=0, epoha=8/30, train MMD²=0.000620768641, 3.0s
  Izbor: reg=0, epoha=9/30, train MMD²=0.000684636509, 3.0s
  Izbor: reg=0, epoha=10/30, train MMD²=0.000778745319, 3.0s
  VALIDACIJA MMD²=0.0021519574
  Izbor: reg=0, epoha=11/30, train MMD²=0.000762712232, 3.0s
  Izbor: reg=0, epoha=12/30, train MMD²=0.000677593358, 3.0s
  Izbor: reg=0, epoha=13/30, train MMD²=0.000612264137, 3.0s
  Izbor: reg=0, epoha=14/30, train MMD²=0.000577743127, 3.0s
  Izbor: reg=0, epoha=15/30, train MMD²=0.000540330253, 3.0s
  VALIDACIJA MMD²=0.00163501435
  Izbor: reg=0, epoha=16/30, train MMD²=0.000494734013, 3.0s
  Izbor: reg=0, epoha=17/30, train MMD²=0.000459861821, 3.0s
  Izbor: reg=0, epoha=18/30, train MMD²=0.000442180519, 3.0s
  Izbor: reg=0, epoha=19/30, train MMD²=0.000429303572, 3.0s
  Izbor: reg=0, epoha=20/30, train MMD²=0.000412787404, 3.0s
  VALIDACIJA MMD²=0.00177035847
  Izbor: reg=0, epoha=21/30, train MMD²=0.000396443884, 3.0s
  Izbor: reg=0, epoha=22/30, train MMD²=0.000384430881, 3.0s
  Izbor: reg=0, epoha=23/30, train MMD²=0.000374919974, 3.0s
  Izbor: reg=0, epoha=24/30, train MMD²=0.00036409944, 3.0s
  Izbor: reg=0, epoha=25/30, train MMD²=0.000350381407, 3.0s
  VALIDACIJA MMD²=0.00158600371
  Izbor: reg=0, epoha=26/30, train MMD²=0.000335663748, 3.0s
  Izbor: reg=0, epoha=27/30, train MMD²=0.000324173772, 3.1s
  Izbor: reg=0, epoha=28/30, train MMD²=0.00031818151, 3.0s
  Izbor: reg=0, epoha=29/30, train MMD²=0.000314911588, 3.0s
  Izbor: reg=0, epoha=30/30, train MMD²=0.000309373681, 3.0s
  VALIDACIJA MMD²=0.00160944849
  Izbor: reg=0.0001, epoha=1/30, train MMD²=0.00201570903, 3.0s
  Izbor: reg=0.0001, epoha=2/30, train MMD²=0.00287997736, 3.0s
  Izbor: reg=0.0001, epoha=3/30, train MMD²=0.00146077493, 3.0s
  Izbor: reg=0.0001, epoha=4/30, train MMD²=0.00145352656, 3.0s
  Izbor: reg=0.0001, epoha=5/30, train MMD²=0.00145534746, 3.0s
  VALIDACIJA MMD²=0.00218060495
  Izbor: reg=0.0001, epoha=6/30, train MMD²=0.00112822721, 3.1s
  Izbor: reg=0.0001, epoha=7/30, train MMD²=0.000771693257, 3.1s
  Izbor: reg=0.0001, epoha=8/30, train MMD²=0.000617746847, 3.0s
  Izbor: reg=0.0001, epoha=9/30, train MMD²=0.000682666066, 3.0s
  Izbor: reg=0.0001, epoha=10/30, train MMD²=0.000777694676, 3.0s
  VALIDACIJA MMD²=0.0021556404
  Izbor: reg=0.0001, epoha=11/30, train MMD²=0.000760971563, 3.0s
  Izbor: reg=0.0001, epoha=12/30, train MMD²=0.000675486826, 3.0s
  Izbor: reg=0.0001, epoha=13/30, train MMD²=0.000610659915, 3.2s
  Izbor: reg=0.0001, epoha=14/30, train MMD²=0.000575984944, 3.0s
  Izbor: reg=0.0001, epoha=15/30, train MMD²=0.000538090431, 3.0s
  VALIDACIJA MMD²=0.00162889405
  Izbor: reg=0.0001, epoha=16/30, train MMD²=0.000493162179, 3.0s
  Izbor: reg=0.0001, epoha=17/30, train MMD²=0.000459332201, 3.0s
  Izbor: reg=0.0001, epoha=18/30, train MMD²=0.000441506549, 3.0s
  Izbor: reg=0.0001, epoha=19/30, train MMD²=0.000427630656, 3.1s
  Izbor: reg=0.0001, epoha=20/30, train MMD²=0.000410841516, 3.1s
  VALIDACIJA MMD²=0.00176708568
  Izbor: reg=0.0001, epoha=21/30, train MMD²=0.000395190142, 3.0s
  Izbor: reg=0.0001, epoha=22/30, train MMD²=0.000383622342, 3.0s
  Izbor: reg=0.0001, epoha=23/30, train MMD²=0.000373625151, 3.0s
  Izbor: reg=0.0001, epoha=24/30, train MMD²=0.000362174753, 3.1s
  Izbor: reg=0.0001, epoha=25/30, train MMD²=0.000348662295, 3.0s
  VALIDACIJA MMD²=0.00158935017
  Izbor: reg=0.0001, epoha=26/30, train MMD²=0.000334833444, 3.0s
  Izbor: reg=0.0001, epoha=27/30, train MMD²=0.000323888963, 3.0s
  Izbor: reg=0.0001, epoha=28/30, train MMD²=0.00031744963, 3.0s
  Izbor: reg=0.0001, epoha=29/30, train MMD²=0.000313202281, 3.0s
  Izbor: reg=0.0001, epoha=30/30, train MMD²=0.000307204719, 3.0s
  VALIDACIJA MMD²=0.00160711603
IZABRANO: reg=0, epohe=25, validation MMD²=0.00158600371
  TRAIN+VALIDATION: epoha 1/25, MMD² pre koraka=0.00194684441, 3.7s
  TRAIN+VALIDATION: epoha 2/25, MMD² pre koraka=0.00285750464, 3.6s
  TRAIN+VALIDATION: epoha 3/25, MMD² pre koraka=0.00153528022, 3.7s
  TRAIN+VALIDATION: epoha 4/25, MMD² pre koraka=0.00144679545, 3.6s
  TRAIN+VALIDATION: epoha 5/25, MMD² pre koraka=0.0013936537, 3.6s
  TRAIN+VALIDATION: epoha 6/25, MMD² pre koraka=0.00107923722, 3.5s
  TRAIN+VALIDATION: epoha 7/25, MMD² pre koraka=0.000776472862, 3.6s
  TRAIN+VALIDATION: epoha 8/25, MMD² pre koraka=0.000664820068, 3.5s
  TRAIN+VALIDATION: epoha 9/25, MMD² pre koraka=0.000712139602, 3.5s
  TRAIN+VALIDATION: epoha 10/25, MMD² pre koraka=0.000753936241, 3.6s
  TRAIN+VALIDATION: epoha 11/25, MMD² pre koraka=0.000712435518, 3.6s
  TRAIN+VALIDATION: epoha 12/25, MMD² pre koraka=0.000652516415, 3.5s
  TRAIN+VALIDATION: epoha 13/25, MMD² pre koraka=0.000628530109, 3.6s
  TRAIN+VALIDATION: epoha 14/25, MMD² pre koraka=0.000610012901, 3.6s
  TRAIN+VALIDATION: epoha 15/25, MMD² pre koraka=0.000557249216, 3.6s
  TRAIN+VALIDATION: epoha 16/25, MMD² pre koraka=0.000485338482, 3.6s
  TRAIN+VALIDATION: epoha 17/25, MMD² pre koraka=0.000432586431, 3.5s
  TRAIN+VALIDATION: epoha 18/25, MMD² pre koraka=0.000410564593, 3.5s
  TRAIN+VALIDATION: epoha 19/25, MMD² pre koraka=0.000404051922, 3.5s
  TRAIN+VALIDATION: epoha 20/25, MMD² pre koraka=0.000403145629, 3.5s
  TRAIN+VALIDATION: epoha 21/25, MMD² pre koraka=0.000405411381, 3.6s
  TRAIN+VALIDATION: epoha 22/25, MMD² pre koraka=0.00040225564, 3.6s
  TRAIN+VALIDATION: epoha 23/25, MMD² pre koraka=0.000384744755, 3.5s
  TRAIN+VALIDATION: epoha 24/25, MMD² pre koraka=0.000355293511, 3.6s
  TRAIN+VALIDATION: epoha 25/25, MMD² pre koraka=0.000326769356, 3.6s
ODVOJENI TEST: MMD² parova=0.00155423247; uslovni NLL sledećeg kola=19.62034331.
To su metrike raspodele, ne procenat budućih pogodaka.
  CELA ISTORIJA: epoha 1/25, MMD² pre koraka=0.00194666637, 4.1s
  CELA ISTORIJA: epoha 2/25, MMD² pre koraka=0.00261121002, 4.1s
  CELA ISTORIJA: epoha 3/25, MMD² pre koraka=0.00152384231, 4.2s
  CELA ISTORIJA: epoha 4/25, MMD² pre koraka=0.00132228276, 4.2s
  CELA ISTORIJA: epoha 5/25, MMD² pre koraka=0.00132790775, 4.2s
  CELA ISTORIJA: epoha 6/25, MMD² pre koraka=0.00105827502, 4.2s
  CELA ISTORIJA: epoha 7/25, MMD² pre koraka=0.000687746386, 4.1s
  CELA ISTORIJA: epoha 8/25, MMD² pre koraka=0.000546156556, 4.1s
  CELA ISTORIJA: epoha 9/25, MMD² pre koraka=0.000637237611, 4.2s
  CELA ISTORIJA: epoha 10/25, MMD² pre koraka=0.000737546035, 4.1s
  CELA ISTORIJA: epoha 11/25, MMD² pre koraka=0.000726402107, 4.2s
  CELA ISTORIJA: epoha 12/25, MMD² pre koraka=0.000647805935, 4.1s
  CELA ISTORIJA: epoha 13/25, MMD² pre koraka=0.000578694753, 4.2s
  CELA ISTORIJA: epoha 14/25, MMD² pre koraka=0.000536901011, 4.2s
  CELA ISTORIJA: epoha 15/25, MMD² pre koraka=0.0004970406, 4.2s
  CELA ISTORIJA: epoha 16/25, MMD² pre koraka=0.000447107083, 4.2s
  CELA ISTORIJA: epoha 17/25, MMD² pre koraka=0.000403404402, 4.2s
  CELA ISTORIJA: epoha 18/25, MMD² pre koraka=0.000386608029, 4.1s
  CELA ISTORIJA: epoha 19/25, MMD² pre koraka=0.000393110593, 4.1s
  CELA ISTORIJA: epoha 20/25, MMD² pre koraka=0.000397187778, 4.2s
  CELA ISTORIJA: epoha 21/25, MMD² pre koraka=0.000382187685, 4.2s
  CELA ISTORIJA: epoha 22/25, MMD² pre koraka=0.000354929844, 4.2s
  CELA ISTORIJA: epoha 23/25, MMD² pre koraka=0.000330859503, 4.2s
  CELA ISTORIJA: epoha 24/25, MMD² pre koraka=0.000317288473, 4.2s
  CELA ISTORIJA: epoha 25/25, MMD² pre koraka=0.000311446919, 4.2s
Tražim maksimum među svih 15,380,937 legalnih kombinacija. Pretraga može potrajati.
Normalizacija nastavka Z=0.000543125730056; zbir marginala=7.000000000000.
Pretraga: 0.0s, posećeni čvorovi=89,502.
  Osetljivost na promenu prethodnog kola: 0.174897821
  Ove dijagnostike nisu označile problem; to nije potvrda prediktivne vrednosti.
  Dijagnostike ne menjaju izabranu kombinaciju.

NEXT [loto7_4706_k84_loto_2975.csv]: 19 22 25 28 31 34 37
Masa u naučenoj uslovnoj raspodeli: 5.45005370446e-05.
To je interna modelska masa, ne potvrđena verovatnoća izvlačenja.



============================================================================
loto7_4706_k84_loto_plus_1731.csv
Kola=1731, svi prelazi=1730; train=1211, validation=259, test=260.
Računam kernel statistiku iz treninga, bez uzorkovanja...
Kernel: medijana train Hamming udaljenosti=24; gamma=0.02888113252. Ostaje fiksan u svim fazama.
  Izbor: reg=0, epoha=1/30, train MMD²=0.00199666991, 1.9s
  Izbor: reg=0, epoha=2/30, train MMD²=0.00235343202, 1.8s
  Izbor: reg=0, epoha=3/30, train MMD²=0.00156394737, 1.9s
  Izbor: reg=0, epoha=4/30, train MMD²=0.00109705975, 1.8s
  Izbor: reg=0, epoha=5/30, train MMD²=0.000947060388, 1.8s
  VALIDACIJA MMD²=0.00297411259
  Izbor: reg=0, epoha=6/30, train MMD²=0.000843446368, 1.8s
  Izbor: reg=0, epoha=7/30, train MMD²=0.000735151838, 1.9s
  Izbor: reg=0, epoha=8/30, train MMD²=0.000691176227, 1.9s
  Izbor: reg=0, epoha=9/30, train MMD²=0.00070522164, 1.8s
  Izbor: reg=0, epoha=10/30, train MMD²=0.00071945703, 1.9s
  VALIDACIJA MMD²=0.00392904957
  Izbor: reg=0, epoha=11/30, train MMD²=0.000692952451, 1.8s
  Izbor: reg=0, epoha=12/30, train MMD²=0.000635628668, 1.9s
  Izbor: reg=0, epoha=13/30, train MMD²=0.000581166733, 1.8s
  Izbor: reg=0, epoha=14/30, train MMD²=0.000541819583, 1.8s
  Izbor: reg=0, epoha=15/30, train MMD²=0.000507796467, 1.9s
  VALIDACIJA MMD²=0.00291609359
  Izbor: reg=0, epoha=16/30, train MMD²=0.000468610113, 1.9s
  Izbor: reg=0, epoha=17/30, train MMD²=0.000435816049, 1.8s
  Izbor: reg=0, epoha=18/30, train MMD²=0.000426731258, 1.8s
  Izbor: reg=0, epoha=19/30, train MMD²=0.000431431091, 1.8s
  Izbor: reg=0, epoha=20/30, train MMD²=0.00042788651, 1.9s
  VALIDACIJA MMD²=0.00339285995
  Izbor: reg=0, epoha=21/30, train MMD²=0.000412589719, 1.8s
  Izbor: reg=0, epoha=22/30, train MMD²=0.000396100516, 1.9s
  Izbor: reg=0, epoha=23/30, train MMD²=0.000385154185, 1.9s
  Izbor: reg=0, epoha=24/30, train MMD²=0.00037724067, 1.9s
  Izbor: reg=0, epoha=25/30, train MMD²=0.000367673173, 1.8s
  VALIDACIJA MMD²=0.00316641524
  Izbor: reg=0, epoha=26/30, train MMD²=0.000356978024, 1.8s
  Izbor: reg=0, epoha=27/30, train MMD²=0.000346850182, 1.9s
  Izbor: reg=0, epoha=28/30, train MMD²=0.000336416138, 2.0s
  Izbor: reg=0, epoha=29/30, train MMD²=0.00032672211, 1.9s
  Izbor: reg=0, epoha=30/30, train MMD²=0.000321928872, 1.8s
  VALIDACIJA MMD²=0.00324308178
  Izbor: reg=0.0001, epoha=1/30, train MMD²=0.00199666991, 1.8s
  Izbor: reg=0.0001, epoha=2/30, train MMD²=0.00235608157, 1.8s
  Izbor: reg=0.0001, epoha=3/30, train MMD²=0.00156804209, 1.9s
  Izbor: reg=0.0001, epoha=4/30, train MMD²=0.00109568133, 1.8s
  Izbor: reg=0.0001, epoha=5/30, train MMD²=0.000949463322, 1.8s
  VALIDACIJA MMD²=0.00296909889
  Izbor: reg=0.0001, epoha=6/30, train MMD²=0.000846830822, 1.8s
  Izbor: reg=0.0001, epoha=7/30, train MMD²=0.000731312715, 1.9s
  Izbor: reg=0.0001, epoha=8/30, train MMD²=0.000689664291, 1.8s
  Izbor: reg=0.0001, epoha=9/30, train MMD²=0.000708607859, 1.8s
  Izbor: reg=0.0001, epoha=10/30, train MMD²=0.000721964052, 1.8s
  VALIDACIJA MMD²=0.00390964002
  Izbor: reg=0.0001, epoha=11/30, train MMD²=0.00069236884, 1.8s
  Izbor: reg=0.0001, epoha=12/30, train MMD²=0.000634093543, 1.8s
  Izbor: reg=0.0001, epoha=13/30, train MMD²=0.000581447245, 1.8s
  Izbor: reg=0.0001, epoha=14/30, train MMD²=0.000542701193, 1.8s
  Izbor: reg=0.0001, epoha=15/30, train MMD²=0.000506417528, 1.8s
  VALIDACIJA MMD²=0.0029188163
  Izbor: reg=0.0001, epoha=16/30, train MMD²=0.000466280424, 1.8s
  Izbor: reg=0.0001, epoha=17/30, train MMD²=0.000435704954, 1.8s
  Izbor: reg=0.0001, epoha=18/30, train MMD²=0.000427872732, 1.8s
  Izbor: reg=0.0001, epoha=19/30, train MMD²=0.000431254909, 1.9s
  Izbor: reg=0.0001, epoha=20/30, train MMD²=0.000426416651, 1.8s
  VALIDACIJA MMD²=0.00339490174
  Izbor: reg=0.0001, epoha=21/30, train MMD²=0.000411240803, 1.8s
  Izbor: reg=0.0001, epoha=22/30, train MMD²=0.00039534953, 1.8s
  Izbor: reg=0.0001, epoha=23/30, train MMD²=0.000384742195, 1.8s
  Izbor: reg=0.0001, epoha=24/30, train MMD²=0.000377093266, 1.9s
  Izbor: reg=0.0001, epoha=25/30, train MMD²=0.000367451587, 1.8s
  VALIDACIJA MMD²=0.00315981202
  Izbor: reg=0.0001, epoha=26/30, train MMD²=0.000355746832, 1.8s
  Izbor: reg=0.0001, epoha=27/30, train MMD²=0.000344645161, 1.9s
  Izbor: reg=0.0001, epoha=28/30, train MMD²=0.000334824833, 1.8s
  Izbor: reg=0.0001, epoha=29/30, train MMD²=0.000326679271, 1.9s
  Izbor: reg=0.0001, epoha=30/30, train MMD²=0.000322329111, 1.9s
  VALIDACIJA MMD²=0.00324266369
IZABRANO: reg=0, epohe=15, validation MMD²=0.00291609359
  TRAIN+VALIDATION: epoha 1/15, MMD² pre koraka=0.00200097087, 2.2s
  TRAIN+VALIDATION: epoha 2/15, MMD² pre koraka=0.00257259259, 2.2s
  TRAIN+VALIDATION: epoha 3/15, MMD² pre koraka=0.00154431557, 2.2s
  TRAIN+VALIDATION: epoha 4/15, MMD² pre koraka=0.00118178781, 2.2s
  TRAIN+VALIDATION: epoha 5/15, MMD² pre koraka=0.00114055373, 2.2s
  TRAIN+VALIDATION: epoha 6/15, MMD² pre koraka=0.000951366073, 2.2s
  TRAIN+VALIDATION: epoha 7/15, MMD² pre koraka=0.000705384081, 2.2s
  TRAIN+VALIDATION: epoha 8/15, MMD² pre koraka=0.000623183386, 2.2s
  TRAIN+VALIDATION: epoha 9/15, MMD² pre koraka=0.000699643543, 2.1s
  TRAIN+VALIDATION: epoha 10/15, MMD² pre koraka=0.000775674071, 2.2s
  TRAIN+VALIDATION: epoha 11/15, MMD² pre koraka=0.000763550794, 2.2s
  TRAIN+VALIDATION: epoha 12/15, MMD² pre koraka=0.000687606172, 2.2s
  TRAIN+VALIDATION: epoha 13/15, MMD² pre koraka=0.000593144, 2.2s
  TRAIN+VALIDATION: epoha 14/15, MMD² pre koraka=0.000518299933, 2.2s
  TRAIN+VALIDATION: epoha 15/15, MMD² pre koraka=0.0004875462, 2.1s
ODVOJENI TEST: MMD² parova=0.00249234774; uslovni NLL sledećeg kola=20.13728897.
To su metrike raspodele, ne procenat budućih pogodaka.
  CELA ISTORIJA: epoha 1/15, MMD² pre koraka=0.00192453261, 2.5s
  CELA ISTORIJA: epoha 2/15, MMD² pre koraka=0.00248942349, 2.5s
  CELA ISTORIJA: epoha 3/15, MMD² pre koraka=0.00141233533, 2.5s
  CELA ISTORIJA: epoha 4/15, MMD² pre koraka=0.00115398957, 2.5s
  CELA ISTORIJA: epoha 5/15, MMD² pre koraka=0.00116616185, 2.5s
  CELA ISTORIJA: epoha 6/15, MMD² pre koraka=0.000886413588, 2.6s
  CELA ISTORIJA: epoha 7/15, MMD² pre koraka=0.000619243062, 2.5s
  CELA ISTORIJA: epoha 8/15, MMD² pre koraka=0.000609738299, 2.5s
  CELA ISTORIJA: epoha 9/15, MMD² pre koraka=0.000722036886, 2.5s
  CELA ISTORIJA: epoha 10/15, MMD² pre koraka=0.000770873143, 2.5s
  CELA ISTORIJA: epoha 11/15, MMD² pre koraka=0.000718909469, 2.5s
  CELA ISTORIJA: epoha 12/15, MMD² pre koraka=0.000623535836, 2.5s
  CELA ISTORIJA: epoha 13/15, MMD² pre koraka=0.000549184551, 2.5s
  CELA ISTORIJA: epoha 14/15, MMD² pre koraka=0.000516847127, 2.5s
  CELA ISTORIJA: epoha 15/15, MMD² pre koraka=0.000503542624, 2.5s
Tražim maksimum među svih 15,380,937 legalnih kombinacija. Pretraga može potrajati.
Normalizacija nastavka Z=0.000245293022803; zbir marginala=7.000000000000.
Pretraga: 0.0s, posećeni čvorovi=35,338.
  Osetljivost na promenu prethodnog kola: 0.179437217
  Ove dijagnostike nisu označile problem; to nije potvrda prediktivne vrednosti.
  Dijagnostike ne menjaju izabranu kombinaciju.

NEXT [loto7_4706_k84_loto_plus_1731.csv]: 19 22 25 28 31 34 37
Masa u naučenoj uslovnoj raspodeli: 0.000141296430959.
To je interna modelska masa, ne potvrđena verovatnoća izvlačenja.



PROVERA ISTOG NEXT-a:
  loto7_4706_k84.csv
  loto7_4706_k84_loto_2975.csv
  Ulazi: 96d38aa413f09d9c / 14dbe5d3efb2e282
  Modeli: d758ffa2783fff8f / 890a8294cc39cff7
  Maksimalna razlika marginala: 0.02337126125
  Podudaranje zahteva proveru zajedničkog maksimuma ili degeneracije. Kombinacije nisu veštački razdvojene.


PROVERA ISTOG NEXT-a:
  loto7_4706_k84.csv
  loto7_4706_k84_loto_plus_1731.csv
  Ulazi: 96d38aa413f09d9c / 357bf8ceabbee6e0
  Modeli: d758ffa2783fff8f / 16498140307a6643
  Maksimalna razlika marginala: 0.03254304531
  Podudaranje zahteva proveru zajedničkog maksimuma ili degeneracije. Kombinacije nisu veštački razdvojene.


PROVERA ISTOG NEXT-a:
  loto7_4706_k84_loto_2975.csv
  loto7_4706_k84_loto_plus_1731.csv
  Ulazi: 14dbe5d3efb2e282 / 357bf8ceabbee6e0
  Modeli: 890a8294cc39cff7 / 16498140307a6643
  Maksimalna razlika marginala: 0.02707784665
  Podudaranje zahteva proveru zajedničkog maksimuma ili degeneracije. Kombinacije nisu veštački razdvojene.

Obrada završena.
"""



"""
MMD formule iz HQCS rada

HQCS zadržava prethodni izmereni bit u radnom kubitu; ne resetuje ga kao CoMB. 
MMD računam sabiranjem preko kvantnih prelaza 
— uključujući poređenje dve kopije modela 
— bez generisanja slučajnih kombinacija.
"""
