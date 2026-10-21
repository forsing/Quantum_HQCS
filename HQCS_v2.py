#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
HQCS Loto 7/39 — V2

Samostalna adaptacija:
https://arxiv.org/abs/2610.15083

Kvantni model:
- dva memorijska kubita i jedan radni kubit;
- radni kubit zadržava prethodni izmereni bit;
- posebno parametrizovano kolo na svakoj poziciji;
- kompleksne amplitude, bez klasičnog ML izlaznog sloja.

Učenje:
- prethodno kolo uslovljava raspodelu sledećeg kola;
- uslovni MMD kernel-score preko cele kombinacije;
- mešavina tri eksponencijalna Hamming kernela;
- tačna očekivanja, bez uzorkovanja;
- klasični Adam;
- izbor inicijalizacije i epohe po uslovnom validacionom NLL-u;
- zaseban test, pa završno učenje na SVIM istorijskim prelazima.

Uslovni kernel-score:
 E[k(X,X') | h] - 2 E[k(X,y) | h] + k(y,y).

Za jedno opaženo y to je MMD²(Q(.|h), delta_y).
U očekivanju po podacima minimizuje odstupanje od uslovne raspodele.
Ovo nije isti broj kao zajednički MMD iz V1.

Nema slučajnih inicijalizacija, mešanja, uzorkovanja, dopunjavanja,
zaokruživanja ili kozmetičkih filtera.

Kod radi determinističku klasičnu simulaciju kvantnog modela.
Nije izvršavanje na kvantnom procesoru.
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
    from torch.utils.checkpoint import checkpoint
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

N = 39
K = 7
MEMORY = 2
DEPTH = 4
BATCH = 64
MAX_EPOCHS = 30
CHECK_EVERY = 5
PATIENCE = 3
LEARNING_RATE = 0.01
PROFILES = (0, 1)
KERNEL_MULTIPLIERS = (1.0, 4.0, 16.0)

REAL = torch.float64
COMPLEX = torch.complex128
torch.set_default_dtype(REAL)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class TrainingRejected(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def digest(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()[:16]


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not x.strip() for x in fields):
                continue
            if len(fields) != K:
                raise ValueError(f"{path}, red {line}: očekivano 7 polja.")
            try:
                values = [int(x.strip()) for x in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: polje nije ceo broj."
                ) from exc
            if len(set(values)) != K or not all(1 <= x <= N for x in values):
                raise ValueError(
                    f"{path}, red {line}: nelegalna kombinacija {values}."
                )
            rows.append(sorted(values))

    require(len(rows) >= 100, f"{path}: manje od 100 kola.")
    draws = np.asarray(rows, dtype=np.int64)
    bits = np.zeros((len(draws), N), dtype=np.int64)
    bits[np.arange(len(draws))[:, None], draws - 1] = 1
    return draws, bits


def local_operator(single, q, nq):
    result = np.array([[1.0 + 0j]])
    for j in reversed(range(nq)):
        result = np.kron(result, single if j == q else np.eye(2))
    return torch.tensor(result, dtype=COMPLEX)


def cnot(nq, control, target):
    result = torch.zeros((1 << nq, 1 << nq), dtype=COMPLEX)
    for column in range(1 << nq):
        row = column ^ ((1 << target) if (column >> control) & 1 else 0)
        result[row, column] = 1.0
    return result


class HQCS(torch.nn.Module):
    def __init__(self, group=N, memory=MEMORY, depth=DEPTH, profile=0):
        super().__init__()
        self.group = group
        self.steps = 2 * group
        self.memory = memory
        self.nq = memory + 1
        self.d = 1 << memory
        self.depth = depth
        self.profile = profile

        theta = torch.empty(
            (self.steps, depth, self.nq, 2), dtype=REAL
        )

        # Nema zavisnosti inicijalizacije od pozicije broja.
        # Dva unapred određena početka; nisu slučajni ni pseudonasumični.
        for layer in range(depth):
            for q in range(self.nq):
                if profile == 0:
                    ry = 0.70 + 0.11 * q + 0.07 * layer
                    rz = 0.23 + 0.09 * q - 0.04 * layer
                else:
                    ry = 1.10 - 0.08 * q + 0.05 * layer
                    rz = -0.31 + 0.06 * q + 0.08 * layer
                theta[:, layer, q, 0] = ry
                theta[:, layer, q, 1] = rz
        self.theta = torch.nn.Parameter(theta)

        size = 1 << self.nq
        self.register_buffer("eye", torch.eye(size, dtype=COMPLEX))
        y = np.array([[0, -1j], [1j, 0]])
        z = np.array([[1, 0], [0, -1]])
        self.register_buffer(
            "ys", torch.stack([
                local_operator(y, q, self.nq) for q in range(self.nq)
            ])
        )
        self.register_buffer(
            "zs", torch.stack([
                local_operator(z, q, self.nq) for q in range(self.nq)
            ])
        )

        self.edges = []
        entanglers = []
        for layer in range(depth):
            edges = [
                (q, q + 1)
                for q in range(layer % 2, self.nq - 1, 2)
            ]
            self.edges.append(edges)
            entangler = torch.eye(size, dtype=COMPLEX)
            for control, target in edges:
                entangler = cnot(self.nq, control, target) @ entangler
            entanglers.append(entangler)
        self.register_buffer("entanglers", torch.stack(entanglers))

    def rotation(self, angles, pauli):
        return (
            torch.cos(angles / 2)[:, None, None] * self.eye
            - 1j * torch.sin(angles / 2)[:, None, None] * pauli
        )

    def operators(self):
        unitary = self.eye[None].expand(self.steps, -1, -1)
        for layer in range(self.depth):
            for q in range(self.nq):
                unitary = self.rotation(
                    self.theta[:, layer, q, 0], self.ys[q]
                ) @ unitary
                unitary = self.rotation(
                    self.theta[:, layer, q, 1], self.zs[q]
                ) @ unitary
            unitary = self.entanglers[layer] @ unitary

        # [korak, novi_bit, prethodni_bit, memorija_out, memorija_in]
        return torch.stack([
            torch.stack([
                unitary[:, x*self.d:(x+1)*self.d,
                        b*self.d:(b+1)*self.d]
                for b in (0, 1)
            ], dim=1)
            for x in (0, 1)
        ], dim=1)

    def circuit(self, t):
        theta = self.theta.detach().cpu().numpy()
        qc = QuantumCircuit(self.nq)
        for layer in range(self.depth):
            for q in range(self.nq):
                qc.ry(float(theta[t, layer, q, 0]), q)
                qc.rz(float(theta[t, layer, q, 1]), q)
            for control, target in self.edges[layer]:
                qc.cx(control, target)
        return qc


def encode(a, context, empty=False):
    batch = len(context)
    d = a.shape[-1]
    state = torch.zeros((batch, d), dtype=COMPLEX)
    state[:, 0] = 1.0
    previous = torch.zeros(batch, dtype=torch.long)
    if empty:
        return state, previous

    for t in range(context.shape[1]):
        x = context[:, t]
        selected = a[t, x, previous]
        state = torch.einsum("bij,bj->bi", selected, state)
        norm2 = state.abs().square().sum(1)
        require(
            torch.all(torch.isfinite(norm2) & (norm2 > 0)).item(),
            "Nulta ili neispravna masa konteksta."
        )
        state = state / torch.sqrt(norm2)[:, None]
        previous = x
    return state, previous


def quadratic(state, effects):
    value = torch.einsum(
        "bi,bij,bj->b", state.conj(), effects, state
    ).real
    require(torch.all(torch.isfinite(value)).item(), "Nefinitna masa.")
    require(float(value.detach().min()) >= -1e-10, "Negativna masa.")
    return value.clamp_min(0.0)


def single_root(a, k):
    """Efekat svih nastavaka sa tačno k jedinica."""
    d = a.shape[-1]
    following = torch.zeros((2, k+1, d, d), dtype=COMPLEX)
    following[:, 0] = torch.eye(d, dtype=COMPLEX)

    for t in reversed(range(len(a))):
        outputs = []
        for previous in (0, 1):
            value = torch.zeros_like(following[0])
            for x in (0, 1):
                tail = following[x]
                if x:
                    tail = torch.cat(
                        (torch.zeros_like(tail[:1]), tail[:-1]), dim=0
                    )
                operator = a[t, x, previous]
                value = value + operator.mH @ tail @ operator
            outputs.append(value)
        following = torch.stack(outputs)
    return following[:, k]


def pair_pullback(effect, a, b):
    """
    (a⊗b)† effect (a⊗b), bez eksplicitnog velikog Kronecker množenja.
    """
    d = a.shape[0]
    shape = effect.shape[:-2]
    h = effect.reshape(*shape, d, d, d, d)
    h = torch.einsum("ai,...abcd->...ibcd", a.conj(), h)
    h = torch.einsum("bj,...ibcd->...ijcd", b.conj(), h)
    h = torch.einsum("ck,...ijcd->...ijkd", a, h)
    h = torch.einsum("dl,...ijkd->...ijkl", b, h)
    return h.reshape(*shape, d*d, d*d)


def pair_step(following, operators, gamma):
    outputs = []
    for previous1 in (0, 1):
        for previous2 in (0, 1):
            value = torch.zeros_like(following[0])
            for x in (0, 1):
                for y in (0, 1):
                    tail = following[2*x+y]
                    if x:
                        tail = torch.cat(
                            (torch.zeros_like(tail[:1]), tail[:-1]), dim=0
                        )
                    if y:
                        tail = torch.cat(
                            (torch.zeros_like(tail[:, :1]), tail[:, :-1]),
                            dim=1,
                        )
                    value = value + math.exp(-gamma*(x-y)**2) * pair_pullback(
                        tail, operators[x, previous1], operators[y, previous2]
                    )
            outputs.append(value)
    return torch.stack(outputs)


def pair_root(a, k, gamma):
    d = a.shape[-1]
    following = torch.zeros(
        (4, k+1, k+1, d*d, d*d), dtype=COMPLEX
    )
    following[:, 0, 0] = torch.eye(d*d, dtype=COMPLEX)
    for t in reversed(range(len(a))):
        if torch.is_grad_enabled() and a.requires_grad:
            # Ponovno računanje tokom backward-a smanjuje potrebni RAM.
            following = checkpoint(
                pair_step, following, a[t], gamma,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            following = pair_step(following, a[t], gamma)
    return following[:, k, k]


def weighted_cross(a, state, previous, targets, k, gamma):
    """
    Suma raw_mass(x|state) * kernel(x,target) za sve legalne x.
    """
    batch, d = state.shape
    density = state[:, :, None] * state.conj()[:, None, :]
    work = torch.nn.functional.one_hot(previous, num_classes=2).to(REAL)
    first = work[:, :, None, None] * density[:, None]
    rho = torch.cat(
        (
            first[:, :, None],
            torch.zeros((batch, 2, k, d, d), dtype=COMPLEX),
        ),
        dim=2,
    )

    for t in range(len(a)):
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
            weight = torch.exp(
                -gamma * (targets[:, t].to(REAL)-x).square()
            )
            outputs.append(value * weight[:, None, None, None])
        rho = torch.stack(outputs, dim=1)

    return rho[:, :, k].diagonal(dim1=-2, dim2=-1).sum((-1, -2)).real


def make_effects(a, group, k, gammas):
    b = single_root(a[group:], k)
    h = sum(pair_root(a[group:], k, g) for g in gammas) / len(gammas)
    return b, h


def conditional_scores(a, context, target, b, h, k, gammas):
    group = context.shape[1]
    state, previous = encode(a[:group], context)
    z = quadratic(state, b[previous])
    require(torch.all(z > 0).item(), "Nulta legalna normalizacija.")

    pair_state = (
        state[:, :, None] * state[:, None, :]
    ).reshape(len(state), -1)
    qq = quadratic(pair_state, h[3*previous]) / z.square()

    cross = sum(
        weighted_cross(a[group:], state, previous, target, k, gamma)
        for gamma in gammas
    ) / len(gammas)
    scores = qq - 2.0 * cross / z + 1.0

    require(torch.all(torch.isfinite(scores)).item(), "Neispravan score.")
    require(float(scores.detach().min()) >= -1e-8, "Negativan MMD score.")
    return scores


def full_gradient(model, contexts, targets, k, gammas):
    """
    Svi uzorci učestvuju pre jednog Adam koraka.
    Blokovi ograničavaju RAM, ne predstavljaju slučajni mini-batch.
    """
    model.zero_grad(set_to_none=True)
    graph_a = model.operators()
    a = graph_a.detach().requires_grad_(True)

    graph_b, graph_h = make_effects(a, model.group, k, gammas)
    b = graph_b.detach().requires_grad_(True)
    h = graph_h.detach().requires_grad_(True)

    total = 0.0
    count = len(contexts)
    for start in range(0, count, BATCH):
        stop = min(start+BATCH, count)
        score = conditional_scores(
            a, contexts[start:stop], targets[start:stop],
            b, h, k, gammas,
        ).sum() / count
        score.backward()
        total += float(score.detach())

    require(b.grad is not None and h.grad is not None, "Nedostaje gradijent.")
    graph_b.backward(b.grad)
    graph_h.backward(h.grad)
    require(
        a.grad is not None and torch.all(torch.isfinite(a.grad)).item(),
        "Neispravan gradijent instrumenata."
    )
    graph_a.backward(a.grad)
    require(
        model.theta.grad is not None
        and torch.all(torch.isfinite(model.theta.grad)).item(),
        "Neispravan gradijent kapija."
    )
    return total


@torch.no_grad()
def evaluate_score(model, contexts, targets, k, gammas):
    a = model.operators()
    b, h = make_effects(a, model.group, k, gammas)
    total = 0.0
    for start in range(0, len(contexts), BATCH):
        stop = min(start+BATCH, len(contexts))
        total += float(conditional_scores(
            a, contexts[start:stop], targets[start:stop],
            b, h, k, gammas,
        ).sum())
    return total / len(contexts)


@torch.no_grad()
def evaluate_nll(model, contexts, targets, k, empty=False):
    a = model.operators()
    group = model.group
    b = single_root(a[group:], k)
    values = []

    for start in range(0, len(contexts), BATCH):
        stop = min(start+BATCH, len(contexts))
        context = contexts[start:stop]
        target = targets[start:stop]
        state, previous = encode(a[:group], context, empty=empty)
        z = quadratic(state, b[previous])
        require(torch.all(z > 0).item(), "Nulta normalizacija NLL-a.")
        log_mass = torch.zeros(len(context), dtype=REAL)

        for t in range(group):
            x = target[:, t]
            state = torch.einsum(
                "bij,bj->bi", a[group+t, x, previous], state
            )
            norm2 = state.abs().square().sum(1)
            require(
                torch.all(torch.isfinite(norm2) & (norm2 > 0)).item(),
                "Nulta masa opaženog izlaza."
            )
            log_mass += torch.log(norm2)
            state /= torch.sqrt(norm2)[:, None]
            previous = x

        nll = -log_mass + torch.log(z)
        require(torch.all(torch.isfinite(nll)).item(), "Nefinitni NLL.")
        values.extend(nll.tolist())

    return np.asarray(values)


def choose_kernel(targets):
    """Širine kernela koriste samo trening, sve parove bez uzorkovanja."""
    x = np.asarray(targets, dtype=np.float64)
    sums = x.sum(1)
    histogram = np.zeros(x.shape[1]+1, dtype=np.int64)
    for start in range(0, len(x), 256):
        stop = min(start+256, len(x))
        distance = (
            sums[start:stop, None] + sums[None, :]
            - 2.0 * (x[start:stop] @ x.T)
        )
        distance = np.rint(distance).astype(np.int64)
        require(np.all(distance >= 0), "Neispravna udaljenost.")
        histogram += np.bincount(distance.ravel(), minlength=len(histogram))

    midpoint = (int(histogram.sum())+1)//2
    median = int(np.searchsorted(np.cumsum(histogram), midpoint))
    base = math.log(2.0) / max(median, 1)
    return tuple(base*x for x in KERNEL_MULTIPLIERS), median


def np_encode(a, bits):
    state = np.zeros(a.shape[-1], dtype=np.complex128)
    state[0] = 1.0
    previous = 0
    for t, bit in enumerate(bits):
        bit = int(bit)
        state = a[t, bit, previous] @ state
        norm = np.linalg.norm(state)
        require(norm > 0 and np.isfinite(norm), "Neispravan kontekst.")
        state /= norm
        previous = bit
    return state, previous


def path_mass(a, bits, state=None, previous=0):
    if state is None:
        state = np.zeros(a.shape[-1], dtype=np.complex128)
        state[0] = 1.0
    else:
        state = state.copy()
    for t, bit in enumerate(bits):
        bit = int(bit)
        state = a[t, bit, previous] @ state
        previous = bit
    return float(np.vdot(state, state).real)


def np_suffix(a, k):
    n, _, _, d, _ = a.shape
    result = np.zeros((n+1, k+1, 2, d, d), dtype=np.complex128)
    result[n, 0, 0] = result[n, 0, 1] = np.eye(d)

    for t in reversed(range(n)):
        for count in range(k+1):
            for previous in (0, 1):
                value = np.zeros((d, d), dtype=np.complex128)
                for x in (0, 1):
                    if count < x:
                        continue
                    operator = a[t, x, previous]
                    value += (
                        operator.conj().T @ result[t+1, count-x, x] @ operator
                    )
                result[t, count, previous] = (value + value.conj().T)/2
    return result


def audit_distribution(a, state, previous, k):
    n, _, _, d, _ = a.shape
    suffix = np_suffix(a, k)
    z = float(np.vdot(state, suffix[0, k, previous] @ state).real)
    require(z > 0 and np.isfinite(z), "Nulta legalna raspodela.")

    rho = np.zeros((n+1, 2, k+1, d, d), dtype=np.complex128)
    rho[0, previous, 0] = np.outer(state, state.conj())
    marginals = np.zeros(n)

    for t in range(n):
        for b in (0, 1):
            for count in range(k+1):
                for x in (0, 1):
                    if count+x > k:
                        continue
                    operator = a[t, x, b]
                    density = operator @ rho[t, b, count] @ operator.conj().T
                    rho[t+1, x, count+x] += density
                    if x:
                        marginals[t] += float(np.trace(
                            suffix[t+1, k-count-1, 1] @ density
                        ).real) / z

    forward_z = sum(float(np.trace(rho[n, b, k]).real) for b in (0, 1))
    np.testing.assert_allclose(z, forward_z, rtol=3e-9, atol=1e-14)
    np.testing.assert_allclose(marginals.sum(), k, atol=3e-8, rtol=0)
    require(
        np.all(marginals >= -1e-9) and np.all(marginals <= 1+1e-9),
        "Marginale van dozvoljenih granica."
    )
    return suffix, z, marginals


@njit(cache=False)
def matvec(matrix, vector):
    result = np.zeros(len(vector), dtype=np.complex128)
    for i in range(len(vector)):
        for j in range(len(vector)):
            result[i] += matrix[i, j]*vector[j]
    return result


@njit(cache=False)
def effect_value(effect, vector):
    product = matvec(effect, vector)
    value = 0.0
    for i in range(len(vector)):
        value += (vector[i].conjugate()*product[i]).real
    return value


@njit(cache=False)
def find_map(a, suffix, state, previous, k):
    """Branch-and-bound bez ograničenja broja kandidata."""
    n, d = len(a), len(state)
    capacity = 2*n+4
    vectors = np.zeros((capacity, d), dtype=np.complex128)
    positions = np.zeros(capacity, dtype=np.int64)
    counts = np.zeros(capacity, dtype=np.int64)
    previous_bits = np.zeros(capacity, dtype=np.int64)
    masks = np.zeros(capacity, dtype=np.int64)
    bounds = np.zeros(capacity)

    vectors[0] = state
    counts[0] = k
    previous_bits[0] = previous
    bounds[0] = effect_value(suffix[0, k, previous], state)
    tolerance = abs(bounds[0])*1e-10
    top = 1
    best = -1.0
    best_mask = np.int64(0)
    visited = 0

    while top:
        top -= 1
        t = positions[top]
        r = counts[top]
        b = previous_bits[top]
        mask = masks[top]
        bound = bounds[top]
        vector = vectors[top].copy()
        visited += 1

        if best >= 0 and bound+tolerance < best:
            continue
        if t == n:
            if r == 0:
                mass = 0.0
                for j in range(d):
                    mass += (vector[j].conjugate()*vector[j]).real
                if mass > best or (mass == best and mask < best_mask):
                    best, best_mask = mass, mask
            continue

        children = np.zeros((2, d), dtype=np.complex128)
        child_bounds = np.zeros(2)
        valid = np.zeros(2, dtype=np.int64)
        for x in (0, 1):
            if 0 <= r-x <= n-t-1:
                children[x] = matvec(a[t, x, b], vector)
                child_bounds[x] = effect_value(
                    suffix[t+1, r-x, x], children[x]
                )
                valid[x] = 1

        first = 0 if child_bounds[0] <= child_bounds[1] else 1
        for order in (0, 1):
            x = first if order == 0 else 1-first
            if not valid[x]:
                continue
            vectors[top] = children[x]
            positions[top] = t+1
            counts[top] = r-x
            previous_bits[top] = x
            masks[top] = mask | (np.int64(x) << t)
            bounds[top] = child_bounds[x]
            top += 1

    return best_mask, best, visited


def bits_from_mask(mask, n):
    return np.asarray([(int(mask) >> i) & 1 for i in range(n)], dtype=np.int64)


def pattern_description(numbers):
    differences = np.diff(numbers)
    if len(differences) and np.all(differences == differences[0]):
        return f"ARITMETIČKI NIZ +{int(differences[0])}"
    longest = current = 1
    for difference in differences:
        current = current+1 if difference == 1 else 1
        longest = max(longest, current)
    if longest >= 5:
        return f"NIZ OD {longest} UZASTOPNIH BROJEVA"
    return ""


@torch.no_grad()
def map_report(model, context, label):
    a = model.operators().cpu().numpy()
    group = model.group
    state, previous = np_encode(a[:group], context)
    suffix, z, marginals = audit_distribution(a[group:], state, previous, K)
    mask, mass, visited = find_map(a[group:], suffix, state, previous, K)
    output = bits_from_mask(mask, group)
    numbers = tuple((np.flatnonzero(output)+1).tolist())
    np.testing.assert_allclose(
        mass, path_mass(a[group:], output, state, previous),
        rtol=3e-9, atol=1e-15,
    )
    pattern = pattern_description(numbers)
    print(
        f"  {label}: {' '.join(f'{x:02d}' for x in numbers)}"
        + (f" | PROVERITI: {pattern}" if pattern else ""),
        flush=True,
    )
    return {
        "numbers": numbers, "bits": output, "mass": mass/z,
        "z": z, "marginals": marginals, "visited": visited,
        "operators": a,
    }


@torch.no_grad()
def check_qiskit(model):
    a = model.operators().cpu().numpy()
    independent = np.empty_like(a)
    for t in range(model.steps):
        u = np.asarray(Operator(model.circuit(t)).data)
        for x in (0, 1):
            for b in (0, 1):
                independent[t, x, b] = u[
                    x*model.d:(x+1)*model.d,
                    b*model.d:(b+1)*model.d,
                ]
        for b in (0, 1):
            complete = sum(
                a[t, x, b].conj().T @ a[t, x, b] for x in (0, 1)
            )
            np.testing.assert_allclose(
                complete, np.eye(model.d), atol=4e-12, rtol=0
            )
    np.testing.assert_allclose(a, independent, atol=4e-12, rtol=0)
    return independent


def self_checks():
    print("V2 provere: Qiskit, uslovni MMD, gradijenti, granice i MAP...",
          flush=True)
    group, k = 3, 1
    gammas = (0.12, 0.48, 1.92)
    model = HQCS(group=group, memory=1, depth=4, profile=0)
    a_np = check_qiskit(model)

    all_sequences = np.asarray([
        bits_from_mask(mask, 2*group) for mask in range(1 << (2*group))
    ])
    raw = np.asarray([path_mass(a_np, row) for row in all_sequences])
    np.testing.assert_allclose(raw.sum(), 1.0, atol=5e-12)

    qc = QuantumCircuit(model.memory+model.steps)
    for t in range(model.steps):
        work = model.memory+t
        if t:
            qc.cx(work-1, work)
        qc.compose(
            model.circuit(t),
            qubits=list(range(model.memory))+[work],
            inplace=True,
        )
    amplitudes = np.asarray(Statevector.from_instruction(qc).data)
    qiskit_raw = (
        np.abs(amplitudes.reshape(1 << model.steps, model.d))**2
    ).sum(1)
    np.testing.assert_allclose(raw, qiskit_raw, atol=5e-12, rtol=4e-10)

    outputs = np.asarray([
        bits_from_mask(mask, group)
        for mask in range(1 << group)
        if bits_from_mask(mask, group).sum() == k
    ])
    contexts_np = outputs[:2]
    targets_np = outputs[[1, 2]]
    contexts = torch.tensor(contexts_np, dtype=torch.long)
    targets = torch.tensor(targets_np, dtype=torch.long)

    distance = np.abs(outputs[:, None]-outputs[None, :]).sum(2)
    gram = sum(np.exp(-g*distance) for g in gammas)/len(gammas)
    require(np.linalg.eigvalsh(gram).min() >= -1e-10, "Kernel nije PSD.")

    expected = []
    for context, target in zip(contexts_np, targets_np):
        state, previous = np_encode(a_np[:group], context)
        masses = np.asarray([
            path_mass(a_np[group:], output, state, previous)
            for output in outputs
        ])
        probabilities = masses/masses.sum()
        target_index = int(np.flatnonzero(np.all(outputs == target, axis=1))[0])
        expected.append(
            probabilities @ gram @ probabilities
            - 2*(probabilities @ gram[:, target_index]) + 1
        )

    a = model.operators()
    b, h = make_effects(a, group, k, gammas)
    scores = conditional_scores(a, contexts, targets, b, h, k, gammas)
    np.testing.assert_allclose(
        scores.detach().numpy(), expected, atol=5e-10, rtol=5e-8
    )
    scores.mean().backward()
    direct_gradient = model.theta.grad.detach().clone()
    require(float(direct_gradient.abs().max()) > 1e-10,
            "Gradijent je degenerisan.")

    split_value = full_gradient(model, contexts, targets, k, gammas)
    np.testing.assert_allclose(split_value, np.mean(expected), atol=5e-10)
    np.testing.assert_allclose(
        model.theta.grad.numpy(), direct_gradient.numpy(),
        atol=5e-9, rtol=5e-6,
    )

    for index in ((1, 0, 0, 0), (4, 1, 1, 0)):
        original = float(model.theta[index].detach())
        eps = 1e-6
        values = []
        for delta in (eps, -eps):
            with torch.no_grad():
                model.theta[index] = original+delta
            values.append(evaluate_score(model, contexts, targets, k, gammas))
        with torch.no_grad():
            model.theta[index] = original
        np.testing.assert_allclose(
            float(direct_gradient[index]),
            (values[0]-values[1])/(2*eps),
            atol=5e-7, rtol=1e-3,
        )

    state, previous = np_encode(a_np[:group], contexts_np[0])
    for count in range(group+1):
        suffix, z, _ = audit_distribution(a_np[group:], state, previous, count)
        candidates = [
            bits_from_mask(mask, group)
            for mask in range(1 << group)
            if bits_from_mask(mask, group).sum() == count
        ]
        masses = np.asarray([
            path_mass(a_np[group:], row, state, previous)
            for row in candidates
        ])
        mask, maximum, _ = find_map(
            a_np[group:], suffix, state, previous, count
        )
        require(bits_from_mask(mask, group).sum() == count, "Nelegalni MAP.")
        np.testing.assert_allclose(z, masses.sum(), atol=5e-12)
        np.testing.assert_allclose(maximum, masses.max(), atol=5e-12)

    # Provera faktorizovanog pair_pullback i za V2 memoriju d=4.
    m4 = HQCS(group=1, memory=2, depth=4, profile=1)
    aa = m4.operators()[0, 0, 0].detach()
    bb = m4.operators()[0, 1, 1].detach()
    vector = torch.arange(1, 17, dtype=REAL).to(COMPLEX)
    effect = vector[:, None]*vector.conj()[None, :]
    joint = torch.kron(aa.contiguous(), bb.contiguous())
    torch.testing.assert_close(
        pair_pullback(effect, aa, bb), joint.mH @ effect @ joint,
        atol=1e-11, rtol=1e-10,
    )
    print("Sve ugrađene V2 provere prošle.", flush=True)


def train_epoch(model, optimizer, contexts, targets, gammas):
    started = time.perf_counter()
    score = full_gradient(model, contexts, targets, K, gammas)
    gradient = float(model.theta.grad.norm())
    optimizer.step()
    require(torch.all(torch.isfinite(model.theta)).item(),
            "Nefinitni parametri.")
    return score, gradient, time.perf_counter()-started


def choose_parameters(contexts, targets, train_end, validation_end, gammas):
    xc, yc = contexts[:train_end], targets[:train_end]
    xv, yv = contexts[train_end:validation_end], targets[train_end:validation_end]
    best = None
    best_initial = math.inf
    fixed_training_context = xc[-1].numpy()

    for profile in PROFILES:
        model = HQCS(profile=profile)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        initial = float(evaluate_nll(model, xv, yv, K).mean())
        best_initial = min(best_initial, initial)
        local_best = initial
        stale = 0

        print(f"\nProfil {profile}: početni validation NLL={initial:.8f}",
              flush=True)
        map_report(model, fixed_training_context, "POČETNI MAP — train kontekst")

        for epoch in range(1, MAX_EPOCHS+1):
            score, gradient, elapsed = train_epoch(
                model, optimizer, xc, yc, gammas
            )
            print(
                f"  Profil={profile}, epoha={epoch}/{MAX_EPOCHS}, "
                f"kernel-score={score:.8f}, grad={gradient:.5g}, "
                f"{elapsed:.1f}s",
                flush=True,
            )
            if epoch % CHECK_EVERY:
                continue

            validation_nll = float(evaluate_nll(model, xv, yv, K).mean())
            print(f"  VALIDACIJA uslovni NLL={validation_nll:.8f}", flush=True)
            map_report(
                model, fixed_training_context,
                f"MAP posle {epoch} epoha — isti train kontekst",
            )

            candidate = (validation_nll, epoch, profile)
            if best is None or candidate < best:
                best = candidate

            if validation_nll < local_best-1e-6:
                local_best = validation_nll
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                print("  Zaustavljanje: nema novog validacionog napretka.",
                      flush=True)
                break

    require(best is not None, "Nema izabranog modela.")
    if best[0] >= best_initial-1e-6:
        raise TrainingRejected(
            "Nijedan obučeni kandidat nije poboljšao najbolji početni "
            "validacioni NLL. NEXT se ne predstavlja kao uspešno naučen."
        )

    print(
        f"\nIZABRANO: profil={best[2]}, epohe={best[1]}, "
        f"validation NLL={best[0]:.8f}; "
        f"najbolji početni NLL={best_initial:.8f}.",
        flush=True,
    )
    return best[2], best[1]


def fit_fixed(contexts, targets, gammas, profile, epochs, label,
              diagnostic_context=None):
    model = HQCS(profile=profile)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial_theta = model.theta.detach().clone()
    initial_prediction = None

    if diagnostic_context is not None:
        initial_prediction = map_report(
            model, diagnostic_context, label+" POČETNI MAP"
        )

    for epoch in range(1, epochs+1):
        score, gradient, elapsed = train_epoch(
            model, optimizer, contexts, targets, gammas
        )
        print(
            f"  {label}: {epoch}/{epochs}, kernel-score={score:.8f}, "
            f"grad={gradient:.5g}, {elapsed:.1f}s",
            flush=True,
        )
        if diagnostic_context is not None and (
            epoch % CHECK_EVERY == 0 or epoch == epochs
        ):
            map_report(model, diagnostic_context, f"{label} MAP epoha {epoch}")

    change = float((model.theta.detach()-initial_theta).norm())
    print(f"  {label}: L2 promena parametara={change:.9g}", flush=True)
    return model, initial_prediction


def final_diagnostics(model, bits, draws, prediction, initial_prediction):
    a = prediction["operators"]
    messages = []
    pattern = pattern_description(prediction["numbers"])
    if pattern:
        messages.append(pattern+" — proveriti strukturni artefakt modela.")

    if initial_prediction is not None:
        if prediction["numbers"] == initial_prediction["numbers"]:
            messages.append(
                "Završni maksimum jednak je maksimumu pre učenja."
            )
        print(
            "  Promena marginala od inicijalizacije: "
            f"{np.max(np.abs(prediction['marginals'] - initial_prediction['marginals'])):.9g}",
            flush=True,
        )

    span = prediction["numbers"][-1]-prediction["numbers"][0]
    history_spans = draws[:, -1]-draws[:, 0]
    if span <= float(np.quantile(history_spans, 0.01)):
        messages.append("Izrazito grupisanje: raspon u donjih 1% istorije.")

    repeated = int(np.all(
        draws == np.asarray(prediction["numbers"]), axis=1
    ).sum())
    if repeated:
        messages.append(f"Kombinacija se pojavljuje {repeated} puta u istoriji.")

    suffix = np_suffix(a[N:], K)
    same = 0
    examined = 0
    sensitivity = 0.0
    for row in bits[max(0, len(bits)-5):-1]:
        state, previous = np_encode(a[:N], row)
        _, _, marginals = audit_distribution(a[N:], state, previous, K)
        sensitivity = max(
            sensitivity,
            float(np.max(np.abs(marginals-prediction["marginals"]))),
        )
        mask, _, _ = find_map(a[N:], suffix, state, previous, K)
        numbers = tuple((np.flatnonzero(bits_from_mask(mask, N))+1).tolist())
        same += int(numbers == prediction["numbers"])
        examined += 1

    print(
        f"  Promena marginala pri promeni konteksta={sensitivity:.9g}; "
        f"isti maksimum u {same}/{examined} dodatnih konteksta.",
        flush=True,
    )
    if sensitivity < 1e-6:
        messages.append("Raspodela praktično ne reaguje na proverene kontekste.")
    if examined and same == examined:
        messages.append(
            "Maksimum ostaje isti u svim dodatno proverenim kontekstima."
        )

    for message in messages:
        print("  PROVERA MODELA: "+message, flush=True)
    if messages:
        print(
            "  Rezultat zahteva proveru; nije zamenjen ili filtriran.",
            flush=True,
        )
    else:
        print(
            "  Navedene dijagnostike nisu označile problem. "
            "To samo po sebi ne potvrđuje prediktivnu vrednost.",
            flush=True,
        )
    return messages


def process_file(path):
    print("\n"+"="*76, flush=True)
    print(Path(path).name, flush=True)
    draws, bits = read_csv(path)
    contexts = torch.tensor(bits[:-1], dtype=torch.long)
    targets = torch.tensor(bits[1:], dtype=torch.long)
    count = len(contexts)
    train_end = int(0.70*count)
    validation_end = int(0.85*count)

    print(
        f"Kola={len(draws)}, prelazi={count}; train={train_end}, "
        f"validation={validation_end-train_end}, test={count-validation_end}.",
        flush=True,
    )
    gammas, median = choose_kernel(bits[1:train_end+1])
    print(
        f"Kernel medijana iz treninga={median}; "
        f"gammas={tuple(round(x, 9) for x in gammas)}.",
        flush=True,
    )

    profile, epochs = choose_parameters(
        contexts, targets, train_end, validation_end, gammas
    )

    test_model, _ = fit_fixed(
        contexts[:validation_end], targets[:validation_end],
        gammas, profile, epochs, "TRAIN+VALIDATION",
    )
    xt = contexts[validation_end:]
    yt = targets[validation_end:]
    nll = evaluate_nll(test_model, xt, yt, K)
    ablated = evaluate_nll(test_model, xt, yt, K, empty=True)
    score = evaluate_score(test_model, xt, yt, K, gammas)

    print(
        f"ODVOJENI TEST: uslovni NLL={nll.mean():.8f}; "
        f"kernel-score={score:.8f}.",
        flush=True,
    )
    print(
        f"TEST, uklonjen kontekst: NLL={ablated.mean():.8f}; "
        f"razlika bez-sa={(ablated-nll).mean():.8f}.",
        flush=True,
    )
    print(
        "Ablacija koristi isti model sa praznom memorijom; "
        "nije zasebno obučen kontrolni model.",
        flush=True,
    )
    if ablated.mean() <= nll.mean():
        print(
            "PROVERA MODELA: na testu kontekst nije poboljšao NLL "
            "u odnosu na ovu ablaciju.",
            flush=True,
        )
    del test_model

    final_model, initial_prediction = fit_fixed(
        contexts, targets, gammas, profile, epochs, "CELA ISTORIJA",
        diagnostic_context=bits[-1],
    )
    independent = check_qiskit(final_model)
    prediction = map_report(final_model, bits[-1], "ZAVRŠNI DIJAGNOSTIČKI MAP")

    state, previous = np_encode(independent[:N], bits[-1])
    independent_mass = path_mass(
        independent[N:], prediction["bits"], state, previous
    )
    np.testing.assert_allclose(
        prediction["mass"]*prediction["z"], independent_mass,
        rtol=5e-8, atol=1e-15,
    )
    require(
        len(prediction["numbers"]) == K
        and len(set(prediction["numbers"])) == K
        and all(1 <= x <= N for x in prediction["numbers"]),
        "Nelegalni NEXT."
    )
    require(
        np.isfinite(prediction["mass"]) and 0 < prediction["mass"] <= 1+1e-9,
        "Neispravna modelska masa."
    )

    print(
        f"  Z={prediction['z']:.12g}; "
        f"zbir marginala={prediction['marginals'].sum():.12f}; "
        f"MAP čvorovi={prediction['visited']:,}.",
        flush=True,
    )
    messages = final_diagnostics(
        final_model, bits, draws, prediction, initial_prediction
    )
    print(
        "\nNEXT ["+Path(path).name+"]: "
        +" ".join(f"{x:02d}" for x in prediction["numbers"]),
        flush=True,
    )
    print(
        f"Interna modelska masa={prediction['mass']:.12g}. "
        "Nije potvrđena verovatnoća budućeg izvlačenja.",
        flush=True,
    )

    return {
        "file": Path(path).name,
        "numbers": prediction["numbers"],
        "marginals": prediction["marginals"],
        "input_hash": digest(bits),
        "model_hash": digest(prediction["operators"]),
        "warnings": messages,
    }


def main():
    print(
        "HQCS Loto 7/39 — V2\n"
        "Uslovni MMD na tri skale; izbor po validacionom uslovnom NLL-u.\n"
        "Deterministička simulacija, dva memorijska i jedan radni kubit.\n"
        "Klasični Adam i deterministička MAP pretraga.\n"
        "V2 ima skuplje učenje od V1.",
        flush=True,
    )
    for path in PATHS:
        require(Path(path).is_file(), f"Nedostaje CSV: {path}")
    self_checks()

    results = []
    rejected = []
    for path in PATHS:
        try:
            results.append(process_file(path))
        except TrainingRejected as exc:
            rejected.append(Path(path).name)
            print(
                f"\nMODEL NIJE PRIHVAĆEN [{Path(path).name}]: {exc}\n"
                "Nema zamenske kombinacije. Nastavljam sledeći CSV.",
                flush=True,
            )

    for i in range(len(results)):
        for j in range(i+1, len(results)):
            left, right = results[i], results[j]
            if left["numbers"] != right["numbers"]:
                continue
            difference = float(np.max(np.abs(
                left["marginals"]-right["marginals"]
            )))
            print(
                "\nPROVERA ISTOG NEXT-a:\n"
                f"  {left['file']}\n"
                f"  {right['file']}\n"
                f"  Ulazi: {left['input_hash']} / {right['input_hash']}\n"
                f"  Modeli: {left['model_hash']} / {right['model_hash']}\n"
                f"  Maksimalna razlika marginala={difference:.10g}\n"
                "  Zajednički maksimum zahteva dalju proveru. "
                "Rezultati nisu veštački razdvojeni.",
                flush=True,
            )

    if rejected:
        print(
            "\nObrada završena uz neprihvaćene modele: "
            +", ".join(rejected),
            flush=True,
        )
    else:
        print("\nObrada sva tri CSV-a završena.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena pretraga ne potvrđuje maksimum.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception:
        print(
            "\nZaustavljeno zbog greške. "
            "Ne generišem zamensku kombinaciju.",
            file=sys.stderr,
        )
        raise



"""
HQCS Loto 7/39 — V2
Uslovni MMD na tri skale; izbor po validacionom uslovnom NLL-u.
Deterministička simulacija, dva memorijska i jedan radni kubit.
Klasični Adam i deterministička MAP pretraga.
V2 ima skuplje učenje od V1.
V2 provere: Qiskit, uslovni MMD, gradijenti, granice i MAP...
Sve ugrađene V2 provere prošle.


============================================================================
loto7_4706_k84.csv
Kola=4706, prelazi=4705; train=3293, validation=706, test=706.
Kernel medijana iz treninga=12; gammas=(0.057762265, 0.23104906, 0.924196241).

Profil 0: početni validation NLL=18.61513054
  POČETNI MAP — train kontekst: 01 06 08 12 14 18 20
  Profil=0, epoha=1/30, kernel-score=0.80340794, grad=0.0055285, 18.6s
  Profil=0, epoha=2/30, kernel-score=0.80300962, grad=0.0044703, 18.6s
  Profil=0, epoha=3/30, kernel-score=0.80269870, grad=0.0036957, 18.7s
  Profil=0, epoha=4/30, kernel-score=0.80249670, grad=0.0037759, 18.7s
  Profil=0, epoha=5/30, kernel-score=0.80235078, grad=0.0029264, 18.7s
  VALIDACIJA uslovni NLL=18.32757323
  MAP posle 5 epoha — isti train kontekst: 01 03 08 17 19 23 25
  Profil=0, epoha=6/30, kernel-score=0.80221774, grad=0.0025066, 18.5s
  Profil=0, epoha=7/30, kernel-score=0.80210258, grad=0.0022488, 18.5s
  Profil=0, epoha=8/30, kernel-score=0.80200825, grad=0.0022186, 18.6s
  Profil=0, epoha=9/30, kernel-score=0.80192835, grad=0.0022567, 18.5s
  Profil=0, epoha=10/30, kernel-score=0.80183932, grad=0.0020122, 18.8s
  VALIDACIJA uslovni NLL=18.08206052
  MAP posle 10 epoha — isti train kontekst: 01 03 08 14 23 29 35
  Profil=0, epoha=11/30, kernel-score=0.80174983, grad=0.001817, 18.9s
  Profil=0, epoha=12/30, kernel-score=0.80168651, grad=0.0016358, 18.4s
  Profil=0, epoha=13/30, kernel-score=0.80163103, grad=0.0015675, 18.7s
  Profil=0, epoha=14/30, kernel-score=0.80158311, grad=0.0015815, 18.7s
  Profil=0, epoha=15/30, kernel-score=0.80153564, grad=0.0015838, 18.5s
  VALIDACIJA uslovni NLL=17.87410872
  MAP posle 15 epoha — isti train kontekst: 01 03 11 17 23 27 33
  Profil=0, epoha=16/30, kernel-score=0.80148918, grad=0.0015796, 18.6s
  Profil=0, epoha=17/30, kernel-score=0.80143312, grad=0.0019372, 18.6s
  Profil=0, epoha=18/30, kernel-score=0.80139814, grad=0.001496, 19.0s
  Profil=0, epoha=19/30, kernel-score=0.80135225, grad=0.0013754, 18.5s
  Profil=0, epoha=20/30, kernel-score=0.80130824, grad=0.0016749, 19.1s
  VALIDACIJA uslovni NLL=17.67520541
  MAP posle 20 epoha — isti train kontekst: 01 03 05 07 09 28 34
  Profil=0, epoha=21/30, kernel-score=0.80126983, grad=0.0021954, 18.8s
  Profil=0, epoha=22/30, kernel-score=0.80124817, grad=0.0017219, 18.6s
  Profil=0, epoha=23/30, kernel-score=0.80121102, grad=0.0013303, 18.6s
  Profil=0, epoha=24/30, kernel-score=0.80118725, grad=0.0020585, 18.3s
  Profil=0, epoha=25/30, kernel-score=0.80116327, grad=0.0012658, 18.2s
  VALIDACIJA uslovni NLL=17.59082984
  MAP posle 25 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=0, epoha=26/30, kernel-score=0.80114187, grad=0.001264, 18.3s
  Profil=0, epoha=27/30, kernel-score=0.80111033, grad=0.0013753, 18.5s
  Profil=0, epoha=28/30, kernel-score=0.80109283, grad=0.0017485, 18.4s
  Profil=0, epoha=29/30, kernel-score=0.80106250, grad=0.0014365, 18.3s
  Profil=0, epoha=30/30, kernel-score=0.80104506, grad=0.0017493, 18.2s
  VALIDACIJA uslovni NLL=17.51780992
  MAP posle 30 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2

Profil 1: početni validation NLL=19.78953100
  POČETNI MAP — train kontekst: 26 28 30 32 34 36 38 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=1/30, kernel-score=0.80468877, grad=0.0074346, 18.5s
  Profil=1, epoha=2/30, kernel-score=0.80361655, grad=0.0053324, 18.2s
  Profil=1, epoha=3/30, kernel-score=0.80304054, grad=0.0047748, 18.5s
  Profil=1, epoha=4/30, kernel-score=0.80261202, grad=0.0035174, 18.4s
  Profil=1, epoha=5/30, kernel-score=0.80231633, grad=0.0023894, 18.3s
  VALIDACIJA uslovni NLL=18.59043676
  MAP posle 5 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=6/30, kernel-score=0.80221765, grad=0.002531, 18.3s
  Profil=1, epoha=7/30, kernel-score=0.80215944, grad=0.0027598, 18.3s
  Profil=1, epoha=8/30, kernel-score=0.80208053, grad=0.0024405, 18.3s
  Profil=1, epoha=9/30, kernel-score=0.80200389, grad=0.0023632, 18.3s
  Profil=1, epoha=10/30, kernel-score=0.80192266, grad=0.0019801, 18.2s
  VALIDACIJA uslovni NLL=18.36402296
  MAP posle 10 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=11/30, kernel-score=0.80184878, grad=0.0017574, 18.3s
  Profil=1, epoha=12/30, kernel-score=0.80178481, grad=0.0017697, 18.3s
  Profil=1, epoha=13/30, kernel-score=0.80172972, grad=0.0018001, 18.3s
  Profil=1, epoha=14/30, kernel-score=0.80166804, grad=0.0016248, 18.3s
  Profil=1, epoha=15/30, kernel-score=0.80161106, grad=0.0015146, 18.3s
  VALIDACIJA uslovni NLL=18.24970434
  MAP posle 15 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=16/30, kernel-score=0.80155994, grad=0.0014496, 18.3s
  Profil=1, epoha=17/30, kernel-score=0.80150863, grad=0.0014715, 18.5s
  Profil=1, epoha=18/30, kernel-score=0.80146530, grad=0.0014693, 18.2s
  Profil=1, epoha=19/30, kernel-score=0.80141913, grad=0.0015413, 18.3s
  Profil=1, epoha=20/30, kernel-score=0.80136800, grad=0.0013114, 18.5s
  VALIDACIJA uslovni NLL=18.16811603
  MAP posle 20 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=21/30, kernel-score=0.80131959, grad=0.0013926, 18.4s
  Profil=1, epoha=22/30, kernel-score=0.80128398, grad=0.001358, 17.9s
  Profil=1, epoha=23/30, kernel-score=0.80124983, grad=0.001231, 17.8s
  Profil=1, epoha=24/30, kernel-score=0.80122267, grad=0.0013062, 17.8s
  Profil=1, epoha=25/30, kernel-score=0.80119638, grad=0.0014612, 17.8s
  VALIDACIJA uslovni NLL=18.09519706
  MAP posle 25 epoha — isti train kontekst: 01 03 05 07 09 11 13 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=26/30, kernel-score=0.80115584, grad=0.0013514, 18.2s
  Profil=1, epoha=27/30, kernel-score=0.80112848, grad=0.0014195, 17.9s
  Profil=1, epoha=28/30, kernel-score=0.80110382, grad=0.001205, 17.9s
  Profil=1, epoha=29/30, kernel-score=0.80107923, grad=0.0013922, 17.8s
  Profil=1, epoha=30/30, kernel-score=0.80104749, grad=0.0015414, 17.9s
  VALIDACIJA uslovni NLL=18.01591311
  MAP posle 30 epoha — isti train kontekst: 05 07 09 11 13 15 17 | PROVERITI: ARITMETIČKI NIZ +2

IZABRANO: profil=0, epohe=30, validation NLL=17.51780992; najbolji početni NLL=18.61513054.
  TRAIN+VALIDATION: 1/30, kernel-score=0.80340045, grad=0.0054463, 21.0s
  TRAIN+VALIDATION: 2/30, kernel-score=0.80296250, grad=0.0046389, 20.9s
  TRAIN+VALIDATION: 3/30, kernel-score=0.80271739, grad=0.0040692, 21.2s
  TRAIN+VALIDATION: 4/30, kernel-score=0.80252597, grad=0.0033392, 20.9s
  TRAIN+VALIDATION: 5/30, kernel-score=0.80238276, grad=0.0025463, 21.0s
  TRAIN+VALIDATION: 6/30, kernel-score=0.80226394, grad=0.0025727, 20.9s
  TRAIN+VALIDATION: 7/30, kernel-score=0.80216553, grad=0.0021789, 20.8s
  TRAIN+VALIDATION: 8/30, kernel-score=0.80208501, grad=0.002515, 20.9s
  TRAIN+VALIDATION: 9/30, kernel-score=0.80200687, grad=0.0020806, 20.8s
  TRAIN+VALIDATION: 10/30, kernel-score=0.80194002, grad=0.0024232, 20.8s
  TRAIN+VALIDATION: 11/30, kernel-score=0.80186417, grad=0.0017893, 20.8s
  TRAIN+VALIDATION: 12/30, kernel-score=0.80178303, grad=0.0016098, 20.7s
  TRAIN+VALIDATION: 13/30, kernel-score=0.80172824, grad=0.0030714, 20.7s
  TRAIN+VALIDATION: 14/30, kernel-score=0.80167445, grad=0.0019241, 20.9s
  TRAIN+VALIDATION: 15/30, kernel-score=0.80163066, grad=0.0015295, 20.7s
  TRAIN+VALIDATION: 16/30, kernel-score=0.80158705, grad=0.0017133, 20.8s
  TRAIN+VALIDATION: 17/30, kernel-score=0.80154054, grad=0.0016327, 21.3s
  TRAIN+VALIDATION: 18/30, kernel-score=0.80150080, grad=0.0019586, 20.8s
  TRAIN+VALIDATION: 19/30, kernel-score=0.80145081, grad=0.0013318, 20.9s
  TRAIN+VALIDATION: 20/30, kernel-score=0.80140885, grad=0.0014093, 20.8s
  TRAIN+VALIDATION: 21/30, kernel-score=0.80136857, grad=0.0012888, 21.0s
  TRAIN+VALIDATION: 22/30, kernel-score=0.80133517, grad=0.0013215, 21.3s
  TRAIN+VALIDATION: 23/30, kernel-score=0.80130777, grad=0.0017211, 21.1s
  TRAIN+VALIDATION: 24/30, kernel-score=0.80127036, grad=0.0014529, 20.8s
  TRAIN+VALIDATION: 25/30, kernel-score=0.80124162, grad=0.0022137, 20.8s
  TRAIN+VALIDATION: 26/30, kernel-score=0.80122788, grad=0.0015957, 20.8s
  TRAIN+VALIDATION: 27/30, kernel-score=0.80121338, grad=0.0016303, 20.8s
  TRAIN+VALIDATION: 28/30, kernel-score=0.80118446, grad=0.0015704, 20.9s
  TRAIN+VALIDATION: 29/30, kernel-score=0.80116525, grad=0.0013817, 21.2s
  TRAIN+VALIDATION: 30/30, kernel-score=0.80114779, grad=0.0014106, 21.1s
  TRAIN+VALIDATION: L2 promena parametara=2.90214626
ODVOJENI TEST: uslovni NLL=17.52793352; kernel-score=0.80197670.
TEST, uklonjen kontekst: NLL=17.63710323; razlika bez-sa=0.10916972.
Ablacija koristi isti model sa praznom memorijom; nije zasebno obučen kontrolni model.
  CELA ISTORIJA POČETNI MAP: 03 07 09 13 15 19 21
  CELA ISTORIJA: 1/30, kernel-score=0.80345061, grad=0.0053944, 24.6s
  CELA ISTORIJA: 2/30, kernel-score=0.80302550, grad=0.0043884, 24.7s
  CELA ISTORIJA: 3/30, kernel-score=0.80277415, grad=0.0035579, 24.8s
  CELA ISTORIJA: 4/30, kernel-score=0.80257743, grad=0.0028576, 25.2s
  CELA ISTORIJA: 5/30, kernel-score=0.80242416, grad=0.0025228, 25.3s
  CELA ISTORIJA MAP epoha 5: 05 11 15 17 23 27 29
  CELA ISTORIJA: 6/30, kernel-score=0.80229045, grad=0.0021546, 24.9s
  CELA ISTORIJA: 7/30, kernel-score=0.80219585, grad=0.0020828, 24.5s
  CELA ISTORIJA: 8/30, kernel-score=0.80211387, grad=0.0021322, 24.6s
  CELA ISTORIJA: 9/30, kernel-score=0.80203066, grad=0.0021292, 24.5s
  CELA ISTORIJA: 10/30, kernel-score=0.80195108, grad=0.0024604, 24.3s
  CELA ISTORIJA MAP epoha 10: 08 14 23 27 29 33 35
  CELA ISTORIJA: 11/30, kernel-score=0.80188481, grad=0.001687, 24.4s
  CELA ISTORIJA: 12/30, kernel-score=0.80180733, grad=0.0014608, 24.6s
  CELA ISTORIJA: 13/30, kernel-score=0.80174318, grad=0.0013468, 24.5s
  CELA ISTORIJA: 14/30, kernel-score=0.80169504, grad=0.0015055, 24.5s
  CELA ISTORIJA: 15/30, kernel-score=0.80165709, grad=0.0013902, 24.5s
  CELA ISTORIJA MAP epoha 15: 08 14 21 23 26 28 33
  CELA ISTORIJA: 16/30, kernel-score=0.80162129, grad=0.0014411, 24.5s
  CELA ISTORIJA: 17/30, kernel-score=0.80157176, grad=0.0016887, 24.5s
  CELA ISTORIJA: 18/30, kernel-score=0.80152849, grad=0.0014573, 24.4s
  CELA ISTORIJA: 19/30, kernel-score=0.80148706, grad=0.0019935, 24.5s
  CELA ISTORIJA: 20/30, kernel-score=0.80145421, grad=0.0012289, 24.5s
  CELA ISTORIJA MAP epoha 20: 08 14 21 23 26 28 33
  CELA ISTORIJA: 21/30, kernel-score=0.80141352, grad=0.0012066, 24.3s
  CELA ISTORIJA: 22/30, kernel-score=0.80138849, grad=0.00156, 24.9s
  CELA ISTORIJA: 23/30, kernel-score=0.80136009, grad=0.0011137, 25.1s
  CELA ISTORIJA: 24/30, kernel-score=0.80133597, grad=0.0015355, 25.2s
  CELA ISTORIJA: 25/30, kernel-score=0.80131767, grad=0.0011363, 24.6s
  CELA ISTORIJA MAP epoha 25: 03 05 07 09 11 13 15 | PROVERITI: ARITMETIČKI NIZ +2
  CELA ISTORIJA: 26/30, kernel-score=0.80130005, grad=0.0011023, 25.0s
  CELA ISTORIJA: 27/30, kernel-score=0.80127571, grad=0.00095959, 24.8s
  CELA ISTORIJA: 28/30, kernel-score=0.80125613, grad=0.00096574, 24.1s
  CELA ISTORIJA: 29/30, kernel-score=0.80123849, grad=0.00092761, 24.1s
  CELA ISTORIJA: 30/30, kernel-score=0.80122012, grad=0.0010584, 23.9s
  CELA ISTORIJA MAP epoha 30: 01 04 06 08 10 12 14
  CELA ISTORIJA: L2 promena parametara=2.88398369
  ZAVRŠNI DIJAGNOSTIČKI MAP: 01 04 06 08 10 12 14
  Z=4.84341328699e-06; zbir marginala=7.000000000000; MAP čvorovi=631,711.
  Promena marginala od inicijalizacije: 0.0756063177
  Promena marginala pri promeni konteksta=0.0562675544; isti maksimum u 1/4 dodatnih konteksta.
  PROVERA MODELA: Izrazito grupisanje: raspon u donjih 1% istorije.
  Rezultat zahteva proveru; nije zamenjen ili filtriran.

NEXT [loto7_4706_k84.csv]: 01 04 06 08 10 12 14
Interna modelska masa=1.0178858781e-05. Nije potvrđena verovatnoća budućeg izvlačenja.



============================================================================ 
loto7_4706_k84_loto_2975.csv
Kola=2975, prelazi=2974; train=2081, validation=446, test=447.
Kernel medijana iz treninga=12; gammas=(0.057762265, 0.23104906, 0.924196241).

Profil 0: početni validation NLL=18.66095240
  POČETNI MAP — train kontekst: 02 06 08 12 14 18 20
  Profil=0, epoha=1/30, kernel-score=0.80350336, grad=0.0062669, 13.1s
  Profil=0, epoha=2/30, kernel-score=0.80288536, grad=0.0040976, 12.9s
  Profil=0, epoha=3/30, kernel-score=0.80263010, grad=0.0040056, 13.0s
  Profil=0, epoha=4/30, kernel-score=0.80238341, grad=0.0033, 13.0s
  Profil=0, epoha=5/30, kernel-score=0.80223738, grad=0.003975, 12.9s
  VALIDACIJA uslovni NLL=18.29994803
  MAP posle 5 epoha — isti train kontekst: 05 11 17 21 23 27 29
  Profil=0, epoha=6/30, kernel-score=0.80213004, grad=0.0027084, 13.0s
  Profil=0, epoha=7/30, kernel-score=0.80205677, grad=0.0023638, 12.9s
  Profil=0, epoha=8/30, kernel-score=0.80197324, grad=0.0020394, 13.0s
  Profil=0, epoha=9/30, kernel-score=0.80187897, grad=0.0018836, 13.2s
  Profil=0, epoha=10/30, kernel-score=0.80179094, grad=0.0019607, 13.4s
  VALIDACIJA uslovni NLL=18.00840276
  MAP posle 10 epoha — isti train kontekst: 06 08 13 19 23 25 31
  Profil=0, epoha=11/30, kernel-score=0.80170131, grad=0.0018447, 13.5s
  Profil=0, epoha=12/30, kernel-score=0.80163286, grad=0.0018811, 13.3s
  Profil=0, epoha=13/30, kernel-score=0.80156687, grad=0.0019098, 13.4s
  Profil=0, epoha=14/30, kernel-score=0.80148329, grad=0.0018031, 13.1s
  Profil=0, epoha=15/30, kernel-score=0.80141831, grad=0.0017126, 12.9s
  VALIDACIJA uslovni NLL=17.76815694
  MAP posle 15 epoha — isti train kontekst: 02 04 09 15 24 30 36
  Profil=0, epoha=16/30, kernel-score=0.80136750, grad=0.0020225, 12.9s
  Profil=0, epoha=17/30, kernel-score=0.80131542, grad=0.0019784, 12.8s
  Profil=0, epoha=18/30, kernel-score=0.80127342, grad=0.0020214, 12.9s
  Profil=0, epoha=19/30, kernel-score=0.80121891, grad=0.0017664, 12.9s
  Profil=0, epoha=20/30, kernel-score=0.80119299, grad=0.0018467, 12.9s
  VALIDACIJA uslovni NLL=17.61367680
  MAP posle 20 epoha — isti train kontekst: 06 08 10 12 14 16 18 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=0, epoha=21/30, kernel-score=0.80115440, grad=0.0014664, 12.9s
  Profil=0, epoha=22/30, kernel-score=0.80111209, grad=0.0015501, 12.9s
  Profil=0, epoha=23/30, kernel-score=0.80106445, grad=0.0015293, 13.0s
  Profil=0, epoha=24/30, kernel-score=0.80102602, grad=0.001666, 13.0s
  Profil=0, epoha=25/30, kernel-score=0.80099451, grad=0.0019908, 12.9s
  VALIDACIJA uslovni NLL=17.52133159
  MAP posle 25 epoha — isti train kontekst: 07 08 10 12 14 16 18
  Profil=0, epoha=26/30, kernel-score=0.80094935, grad=0.0025317, 13.0s
  Profil=0, epoha=27/30, kernel-score=0.80091074, grad=0.001725, 12.9s
  Profil=0, epoha=28/30, kernel-score=0.80088866, grad=0.002598, 13.0s
  Profil=0, epoha=29/30, kernel-score=0.80085564, grad=0.0017217, 12.9s
  Profil=0, epoha=30/30, kernel-score=0.80083184, grad=0.0016863, 13.0s
  VALIDACIJA uslovni NLL=17.47014698
  MAP posle 30 epoha — isti train kontekst: 07 08 10 12 14 16 18

Profil 1: početni validation NLL=19.51270170
  POČETNI MAP — train kontekst: 02 04 22 24 30 32 38
  Profil=1, epoha=1/30, kernel-score=0.80476789, grad=0.0083296, 13.0s
  Profil=1, epoha=2/30, kernel-score=0.80362474, grad=0.0059821, 12.9s
  Profil=1, epoha=3/30, kernel-score=0.80299641, grad=0.0047404, 13.0s
  Profil=1, epoha=4/30, kernel-score=0.80256367, grad=0.0039614, 12.9s
  Profil=1, epoha=5/30, kernel-score=0.80225813, grad=0.0032753, 12.9s
  VALIDACIJA uslovni NLL=18.40111166
  MAP posle 5 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=6/30, kernel-score=0.80208332, grad=0.0023548, 12.9s
  Profil=1, epoha=7/30, kernel-score=0.80202270, grad=0.0026852, 13.0s
  Profil=1, epoha=8/30, kernel-score=0.80197474, grad=0.003029, 12.9s
  Profil=1, epoha=9/30, kernel-score=0.80187173, grad=0.002781, 12.9s
  Profil=1, epoha=10/30, kernel-score=0.80177176, grad=0.0024079, 13.0s
  VALIDACIJA uslovni NLL=18.19757627
  MAP posle 10 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=11/30, kernel-score=0.80169441, grad=0.0024023, 13.0s
  Profil=1, epoha=12/30, kernel-score=0.80162530, grad=0.0021876, 13.4s
  Profil=1, epoha=13/30, kernel-score=0.80156177, grad=0.0027274, 12.9s
  Profil=1, epoha=14/30, kernel-score=0.80150985, grad=0.0026509, 12.9s
  Profil=1, epoha=15/30, kernel-score=0.80145753, grad=0.0020592, 13.1s
  VALIDACIJA uslovni NLL=18.14706646
  MAP posle 15 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=16/30, kernel-score=0.80139622, grad=0.0018011, 13.1s
  Profil=1, epoha=17/30, kernel-score=0.80133549, grad=0.0017325, 13.0s
  Profil=1, epoha=18/30, kernel-score=0.80127064, grad=0.0015799, 13.0s
  Profil=1, epoha=19/30, kernel-score=0.80121148, grad=0.001476, 13.3s
  Profil=1, epoha=20/30, kernel-score=0.80116461, grad=0.0022659, 13.2s
  VALIDACIJA uslovni NLL=18.00948084
  MAP posle 20 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=21/30, kernel-score=0.80113583, grad=0.0015036, 13.0s
  Profil=1, epoha=22/30, kernel-score=0.80109707, grad=0.0013931, 12.9s
  Profil=1, epoha=23/30, kernel-score=0.80105297, grad=0.0013235, 12.9s
  Profil=1, epoha=24/30, kernel-score=0.80101040, grad=0.0016875, 13.2s
  Profil=1, epoha=25/30, kernel-score=0.80096414, grad=0.0012885, 13.2s
  VALIDACIJA uslovni NLL=17.91546416
  MAP posle 25 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=26/30, kernel-score=0.80092584, grad=0.0017045, 13.0s
  Profil=1, epoha=27/30, kernel-score=0.80088708, grad=0.0017011, 13.0s
  Profil=1, epoha=28/30, kernel-score=0.80085008, grad=0.0019651, 13.0s
  Profil=1, epoha=29/30, kernel-score=0.80081320, grad=0.0015693, 12.9s
  Profil=1, epoha=30/30, kernel-score=0.80077568, grad=0.0016603, 13.1s
  VALIDACIJA uslovni NLL=17.80929104
  MAP posle 30 epoha — isti train kontekst: 07 22 24 26 28 30 32

IZABRANO: profil=0, epohe=30, validation NLL=17.47014698; najbolji početni NLL=18.66095240.
  TRAIN+VALIDATION: 1/30, kernel-score=0.80350587, grad=0.0061569, 14.9s
  TRAIN+VALIDATION: 2/30, kernel-score=0.80291839, grad=0.0041375, 15.0s
  TRAIN+VALIDATION: 3/30, kernel-score=0.80268683, grad=0.0040408, 14.8s
  TRAIN+VALIDATION: 4/30, kernel-score=0.80244446, grad=0.0029194, 14.9s
  TRAIN+VALIDATION: 5/30, kernel-score=0.80229789, grad=0.0027247, 14.9s
  TRAIN+VALIDATION: 6/30, kernel-score=0.80220572, grad=0.0027765, 14.8s
  TRAIN+VALIDATION: 7/30, kernel-score=0.80210002, grad=0.0022401, 14.9s
  TRAIN+VALIDATION: 8/30, kernel-score=0.80199076, grad=0.0019369, 14.9s
  TRAIN+VALIDATION: 9/30, kernel-score=0.80189789, grad=0.0095039, 14.9s
  TRAIN+VALIDATION: 10/30, kernel-score=0.80183937, grad=0.0020549, 14.9s
  TRAIN+VALIDATION: 11/30, kernel-score=0.80177099, grad=0.0020518, 14.8s
  TRAIN+VALIDATION: 12/30, kernel-score=0.80169394, grad=0.0019737, 15.0s
  TRAIN+VALIDATION: 13/30, kernel-score=0.80161891, grad=0.0020637, 15.1s
  TRAIN+VALIDATION: 14/30, kernel-score=0.80154961, grad=0.0015424, 15.2s
  TRAIN+VALIDATION: 15/30, kernel-score=0.80149454, grad=0.0014549, 15.3s
  TRAIN+VALIDATION: 16/30, kernel-score=0.80144848, grad=0.0015062, 15.3s
  TRAIN+VALIDATION: 17/30, kernel-score=0.80140001, grad=0.0012953, 15.2s
  TRAIN+VALIDATION: 18/30, kernel-score=0.80135548, grad=0.001181, 15.3s
  TRAIN+VALIDATION: 19/30, kernel-score=0.80131872, grad=0.0011441, 15.0s
  TRAIN+VALIDATION: 20/30, kernel-score=0.80128557, grad=0.0021566, 15.3s
  TRAIN+VALIDATION: 21/30, kernel-score=0.80125646, grad=0.0017711, 14.9s
  TRAIN+VALIDATION: 22/30, kernel-score=0.80122982, grad=0.001512, 15.1s
  TRAIN+VALIDATION: 23/30, kernel-score=0.80119027, grad=0.0016095, 15.0s
  TRAIN+VALIDATION: 24/30, kernel-score=0.80117397, grad=0.0021169, 15.3s
  TRAIN+VALIDATION: 25/30, kernel-score=0.80114051, grad=0.0014107, 15.3s
  TRAIN+VALIDATION: 26/30, kernel-score=0.80111229, grad=0.0013932, 15.2s
  TRAIN+VALIDATION: 27/30, kernel-score=0.80108325, grad=0.0013273, 15.1s
  TRAIN+VALIDATION: 28/30, kernel-score=0.80105573, grad=0.0011883, 15.1s
  TRAIN+VALIDATION: 29/30, kernel-score=0.80103483, grad=0.0010775, 15.5s
  TRAIN+VALIDATION: 30/30, kernel-score=0.80101442, grad=0.0011147, 15.4s
  TRAIN+VALIDATION: L2 promena parametara=2.88359393
ODVOJENI TEST: uslovni NLL=17.56569956; kernel-score=0.80190020.
TEST, uklonjen kontekst: NLL=17.63402157; razlika bez-sa=0.06832201.
Ablacija koristi isti model sa praznom memorijom; nije zasebno obučen kontrolni model.
  CELA ISTORIJA POČETNI MAP: 02 06 08 12 14 18 20
  CELA ISTORIJA: 1/30, kernel-score=0.80354939, grad=0.0060938, 17.1s
  CELA ISTORIJA: 2/30, kernel-score=0.80301620, grad=0.0047442, 17.3s
  CELA ISTORIJA: 3/30, kernel-score=0.80274458, grad=0.004883, 17.4s
  CELA ISTORIJA: 4/30, kernel-score=0.80249701, grad=0.0030852, 17.2s
  CELA ISTORIJA: 5/30, kernel-score=0.80232222, grad=0.0028414, 16.9s
  CELA ISTORIJA MAP epoha 5: 05 11 17 21 23 27 29
  CELA ISTORIJA: 6/30, kernel-score=0.80222613, grad=0.0025941, 16.8s
  CELA ISTORIJA: 7/30, kernel-score=0.80213794, grad=0.0022687, 16.9s
  CELA ISTORIJA: 8/30, kernel-score=0.80205623, grad=0.0019355, 16.9s
  CELA ISTORIJA: 9/30, kernel-score=0.80198524, grad=0.0020751, 16.8s
  CELA ISTORIJA: 10/30, kernel-score=0.80190487, grad=0.0020961, 16.9s
  CELA ISTORIJA MAP epoha 10: 03 09 18 24 30 34 36
  CELA ISTORIJA: 11/30, kernel-score=0.80181811, grad=0.0020976, 16.9s
  CELA ISTORIJA: 12/30, kernel-score=0.80173258, grad=0.0019238, 16.8s
  CELA ISTORIJA: 13/30, kernel-score=0.80164980, grad=0.0016334, 16.9s
  CELA ISTORIJA: 14/30, kernel-score=0.80158080, grad=0.0015389, 16.8s
  CELA ISTORIJA: 15/30, kernel-score=0.80152042, grad=0.0014813, 17.0s
  CELA ISTORIJA MAP epoha 15: 03 09 21 23 27 29 33
  CELA ISTORIJA: 16/30, kernel-score=0.80147188, grad=0.0016942, 17.1s
  CELA ISTORIJA: 17/30, kernel-score=0.80142193, grad=0.0013888, 17.0s
  CELA ISTORIJA: 18/30, kernel-score=0.80137742, grad=0.0013547, 16.9s
  CELA ISTORIJA: 19/30, kernel-score=0.80133414, grad=0.001328, 16.7s
  CELA ISTORIJA: 20/30, kernel-score=0.80130384, grad=0.0013669, 16.6s
  CELA ISTORIJA MAP epoha 20: 06 08 10 12 14 16 31
  CELA ISTORIJA: 21/30, kernel-score=0.80127114, grad=0.0011956, 16.8s
  CELA ISTORIJA: 22/30, kernel-score=0.80124630, grad=0.0012825, 16.5s
  CELA ISTORIJA: 23/30, kernel-score=0.80121838, grad=0.0024751, 16.5s
  CELA ISTORIJA: 24/30, kernel-score=0.80119255, grad=0.0012113, 16.5s
  CELA ISTORIJA: 25/30, kernel-score=0.80116578, grad=0.0014009, 16.4s
  CELA ISTORIJA MAP epoha 25: 07 08 10 12 14 16 18
  CELA ISTORIJA: 26/30, kernel-score=0.80114893, grad=0.0010375, 16.6s
  CELA ISTORIJA: 27/30, kernel-score=0.80113839, grad=0.0014352, 16.5s
  CELA ISTORIJA: 28/30, kernel-score=0.80110810, grad=0.0011392, 16.6s
  CELA ISTORIJA: 29/30, kernel-score=0.80108402, grad=0.0013241, 16.4s
  CELA ISTORIJA: 30/30, kernel-score=0.80105548, grad=0.0011364, 16.5s
  CELA ISTORIJA MAP epoha 30: 07 08 10 12 14 16 18
  CELA ISTORIJA: L2 promena parametara=2.82218065
  ZAVRŠNI DIJAGNOSTIČKI MAP: 07 08 10 12 14 16 18
  Z=2.69006686851e-06; zbir marginala=7.000000000000; MAP čvorovi=333,054.
  Promena marginala od inicijalizacije: 0.122462346
  Promena marginala pri promeni konteksta=0.11142628; isti maksimum u 3/4 dodatnih konteksta.
  PROVERA MODELA: Izrazito grupisanje: raspon u donjih 1% istorije.
  Rezultat zahteva proveru; nije zamenjen ili filtriran.

NEXT [loto7_4706_k84_loto_2975.csv]: 07 08 10 12 14 16 18
Interna modelska masa=1.66197709939e-05. Nije potvrđena verovatnoća budućeg izvlačenja.



============================================================================
loto7_4706_k84_loto_plus_1731.csv
Kola=1731, prelazi=1730; train=1211, validation=259, test=260.
Kernel medijana iz treninga=12; gammas=(0.057762265, 0.23104906, 0.924196241).

Profil 0: početni validation NLL=18.72583036
  POČETNI MAP — train kontekst: 01 03 09 13 15 19 21
  Profil=0, epoha=1/30, kernel-score=0.80332878, grad=0.0061039, 9.0s
  Profil=0, epoha=2/30, kernel-score=0.80271120, grad=0.0043493, 8.9s
  Profil=0, epoha=3/30, kernel-score=0.80238365, grad=0.0039731, 8.9s
  Profil=0, epoha=4/30, kernel-score=0.80213340, grad=0.0034923, 8.9s
  Profil=0, epoha=5/30, kernel-score=0.80191266, grad=0.0034769, 9.0s
  VALIDACIJA uslovni NLL=18.49361041
  MAP posle 5 epoha — isti train kontekst: 09 11 17 21 23 27 29
  Profil=0, epoha=6/30, kernel-score=0.80177082, grad=0.0034247, 9.0s
  Profil=0, epoha=7/30, kernel-score=0.80166477, grad=0.0033968, 8.9s
  Profil=0, epoha=8/30, kernel-score=0.80150620, grad=0.0025007, 9.0s
  Profil=0, epoha=9/30, kernel-score=0.80136562, grad=0.0027816, 8.9s
  Profil=0, epoha=10/30, kernel-score=0.80126989, grad=0.0025911, 9.1s
  VALIDACIJA uslovni NLL=18.18430450
  MAP posle 10 epoha — isti train kontekst: 07 13 19 23 25 29 31
  Profil=0, epoha=11/30, kernel-score=0.80117456, grad=0.0025016, 9.1s
  Profil=0, epoha=12/30, kernel-score=0.80111690, grad=0.0033961, 9.1s
  Profil=0, epoha=13/30, kernel-score=0.80103632, grad=0.0032211, 9.1s
  Profil=0, epoha=14/30, kernel-score=0.80097634, grad=0.0028599, 9.1s
  Profil=0, epoha=15/30, kernel-score=0.80091836, grad=0.0024793, 9.3s
  VALIDACIJA uslovni NLL=18.00771858
  MAP posle 15 epoha — isti train kontekst: 07 13 19 23 25 29 31
  Profil=0, epoha=16/30, kernel-score=0.80083950, grad=0.0023295, 9.1s
  Profil=0, epoha=17/30, kernel-score=0.80080191, grad=0.0023483, 9.3s
  Profil=0, epoha=18/30, kernel-score=0.80074284, grad=0.0023565, 9.0s
  Profil=0, epoha=19/30, kernel-score=0.80068239, grad=0.0023643, 9.1s
  Profil=0, epoha=20/30, kernel-score=0.80063484, grad=0.0024156, 9.1s
  VALIDACIJA uslovni NLL=17.91655602
  MAP posle 20 epoha — isti train kontekst: 07 13 19 23 25 29 31
  Profil=0, epoha=21/30, kernel-score=0.80057733, grad=0.0020628, 9.2s
  Profil=0, epoha=22/30, kernel-score=0.80052763, grad=0.0021182, 9.1s
  Profil=0, epoha=23/30, kernel-score=0.80046865, grad=0.0016943, 9.1s
  Profil=0, epoha=24/30, kernel-score=0.80041689, grad=0.0018583, 9.1s
  Profil=0, epoha=25/30, kernel-score=0.80036268, grad=0.0020907, 8.9s
  VALIDACIJA uslovni NLL=17.83128673
  MAP posle 25 epoha — isti train kontekst: 16 18 21 23 26 28 33
  Profil=0, epoha=26/30, kernel-score=0.80030090, grad=0.0019659, 9.2s
  Profil=0, epoha=27/30, kernel-score=0.80023342, grad=0.0017745, 9.4s
  Profil=0, epoha=28/30, kernel-score=0.80017135, grad=0.0020553, 9.3s
  Profil=0, epoha=29/30, kernel-score=0.80011178, grad=0.0024282, 9.6s
  Profil=0, epoha=30/30, kernel-score=0.80007105, grad=0.0026357, 9.6s
  VALIDACIJA uslovni NLL=17.83342959
  MAP posle 30 epoha — isti train kontekst: 13 19 23 25 27 29 33

Profil 1: početni validation NLL=19.64519904
  POČETNI MAP — train kontekst: 02 04 22 24 30 32 38
  Profil=1, epoha=1/30, kernel-score=0.80453556, grad=0.0071932, 9.3s
  Profil=1, epoha=2/30, kernel-score=0.80352647, grad=0.0058407, 9.5s
  Profil=1, epoha=3/30, kernel-score=0.80285662, grad=0.004481, 9.5s
  Profil=1, epoha=4/30, kernel-score=0.80245647, grad=0.0038704, 9.3s
  Profil=1, epoha=5/30, kernel-score=0.80216438, grad=0.0032372, 9.1s
  VALIDACIJA uslovni NLL=18.81144477
  MAP posle 5 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=6/30, kernel-score=0.80200009, grad=0.0031859, 9.1s
  Profil=1, epoha=7/30, kernel-score=0.80188074, grad=0.0033157, 9.1s
  Profil=1, epoha=8/30, kernel-score=0.80176748, grad=0.0032227, 9.2s
  Profil=1, epoha=9/30, kernel-score=0.80162725, grad=0.0029432, 9.3s
  Profil=1, epoha=10/30, kernel-score=0.80149281, grad=0.002648, 9.3s
  VALIDACIJA uslovni NLL=18.52510145
  MAP posle 10 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=11/30, kernel-score=0.80136237, grad=0.0026039, 9.0s
  Profil=1, epoha=12/30, kernel-score=0.80124073, grad=0.0026788, 8.9s
  Profil=1, epoha=13/30, kernel-score=0.80113175, grad=0.0026792, 9.0s
  Profil=1, epoha=14/30, kernel-score=0.80102594, grad=0.0028069, 8.9s
  Profil=1, epoha=15/30, kernel-score=0.80097709, grad=0.0026618, 9.0s
  VALIDACIJA uslovni NLL=18.43906943
  MAP posle 15 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=16/30, kernel-score=0.80091155, grad=0.0025819, 8.9s
  Profil=1, epoha=17/30, kernel-score=0.80082618, grad=0.0025746, 9.1s
  Profil=1, epoha=18/30, kernel-score=0.80074012, grad=0.0026222, 8.9s
  Profil=1, epoha=19/30, kernel-score=0.80067870, grad=0.0032032, 9.0s
  Profil=1, epoha=20/30, kernel-score=0.80061510, grad=0.0024049, 9.0s
  VALIDACIJA uslovni NLL=18.26108695
  MAP posle 20 epoha — isti train kontekst: 02 04 06 08 10 12 14 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=21/30, kernel-score=0.80056611, grad=0.0024697, 8.9s
  Profil=1, epoha=22/30, kernel-score=0.80050257, grad=0.002202, 9.0s
  Profil=1, epoha=23/30, kernel-score=0.80042880, grad=0.0020522, 8.9s
  Profil=1, epoha=24/30, kernel-score=0.80039187, grad=0.002822, 9.0s
  Profil=1, epoha=25/30, kernel-score=0.80033014, grad=0.0022048, 9.0s
  VALIDACIJA uslovni NLL=18.12528495
  MAP posle 25 epoha — isti train kontekst: 04 06 08 10 12 14 16 | PROVERITI: ARITMETIČKI NIZ +2
  Profil=1, epoha=26/30, kernel-score=0.80027286, grad=0.0025298, 9.0s
  Profil=1, epoha=27/30, kernel-score=0.80021305, grad=0.0022654, 9.1s
  Profil=1, epoha=28/30, kernel-score=0.80016624, grad=0.0025313, 8.9s
  Profil=1, epoha=29/30, kernel-score=0.80009238, grad=0.0029141, 9.0s
  Profil=1, epoha=30/30, kernel-score=0.80002329, grad=0.0023902, 8.9s
  VALIDACIJA uslovni NLL=18.10310681
  MAP posle 30 epoha — isti train kontekst: 04 06 08 10 12 14 16 | PROVERITI: ARITMETIČKI NIZ +2

IZABRANO: profil=0, epohe=25, validation NLL=17.83128673; najbolji početni NLL=18.72583036.
  TRAIN+VALIDATION: 1/25, kernel-score=0.80340357, grad=0.0059942, 10.1s
  TRAIN+VALIDATION: 2/25, kernel-score=0.80276047, grad=0.0046884, 10.0s
  TRAIN+VALIDATION: 3/25, kernel-score=0.80246518, grad=0.0036399, 10.0s
  TRAIN+VALIDATION: 4/25, kernel-score=0.80223409, grad=0.0032243, 10.1s
  TRAIN+VALIDATION: 5/25, kernel-score=0.80206646, grad=0.0030873, 10.0s
  TRAIN+VALIDATION: 6/25, kernel-score=0.80189673, grad=0.0030704, 10.1s
  TRAIN+VALIDATION: 7/25, kernel-score=0.80175408, grad=0.0030912, 10.0s
  TRAIN+VALIDATION: 8/25, kernel-score=0.80165541, grad=0.0029772, 10.1s
  TRAIN+VALIDATION: 9/25, kernel-score=0.80152295, grad=0.0027862, 10.1s
  TRAIN+VALIDATION: 10/25, kernel-score=0.80141010, grad=0.0023181, 10.1s
  TRAIN+VALIDATION: 11/25, kernel-score=0.80132634, grad=0.0029987, 10.1s
  TRAIN+VALIDATION: 12/25, kernel-score=0.80125335, grad=0.0022823, 10.0s
  TRAIN+VALIDATION: 13/25, kernel-score=0.80118271, grad=0.0023456, 10.0s
  TRAIN+VALIDATION: 14/25, kernel-score=0.80111362, grad=0.0023683, 10.0s
  TRAIN+VALIDATION: 15/25, kernel-score=0.80104576, grad=0.0025394, 10.1s
  TRAIN+VALIDATION: 16/25, kernel-score=0.80099387, grad=0.0019633, 10.1s
  TRAIN+VALIDATION: 17/25, kernel-score=0.80094273, grad=0.0022014, 10.0s
  TRAIN+VALIDATION: 18/25, kernel-score=0.80089033, grad=0.0019714, 10.1s
  TRAIN+VALIDATION: 19/25, kernel-score=0.80083550, grad=0.0019064, 10.0s
  TRAIN+VALIDATION: 20/25, kernel-score=0.80076167, grad=0.0016185, 10.2s
  TRAIN+VALIDATION: 21/25, kernel-score=0.80070184, grad=0.0019193, 10.0s
  TRAIN+VALIDATION: 22/25, kernel-score=0.80064135, grad=0.0019264, 10.1s
  TRAIN+VALIDATION: 23/25, kernel-score=0.80058928, grad=0.0021257, 10.0s
  TRAIN+VALIDATION: 24/25, kernel-score=0.80053200, grad=0.0019249, 10.2s
  TRAIN+VALIDATION: 25/25, kernel-score=0.80049366, grad=0.0026377, 10.1s
  TRAIN+VALIDATION: L2 promena parametara=2.60742462
ODVOJENI TEST: uslovni NLL=17.66402138; kernel-score=0.80239516.
TEST, uklonjen kontekst: NLL=17.69124819; razlika bez-sa=0.02722681.
Ablacija koristi isti model sa praznom memorijom; nije zasebno obučen kontrolni model.
  CELA ISTORIJA POČETNI MAP: 03 07 09 13 15 19 21
  CELA ISTORIJA: 1/25, kernel-score=0.80344216, grad=0.0057605, 11.3s
  CELA ISTORIJA: 2/25, kernel-score=0.80287042, grad=0.0047291, 11.2s
  CELA ISTORIJA: 3/25, kernel-score=0.80254988, grad=0.0036856, 11.1s
  CELA ISTORIJA: 4/25, kernel-score=0.80227982, grad=0.003406, 11.2s
  CELA ISTORIJA: 5/25, kernel-score=0.80210259, grad=0.0028948, 11.2s
  CELA ISTORIJA MAP epoha 5: 07 13 19 23 25 29 31
  CELA ISTORIJA: 6/25, kernel-score=0.80199009, grad=0.0031771, 11.3s
  CELA ISTORIJA: 7/25, kernel-score=0.80184200, grad=0.0033295, 11.4s
  CELA ISTORIJA: 8/25, kernel-score=0.80169586, grad=0.0028992, 11.6s
  CELA ISTORIJA: 9/25, kernel-score=0.80161445, grad=0.0030399, 11.5s
  CELA ISTORIJA: 10/25, kernel-score=0.80148972, grad=0.0024678, 11.5s
  CELA ISTORIJA MAP epoha 10: 07 13 19 23 25 29 31
  CELA ISTORIJA: 11/25, kernel-score=0.80138529, grad=0.0034397, 11.4s
  CELA ISTORIJA: 12/25, kernel-score=0.80130573, grad=0.0025368, 11.5s
  CELA ISTORIJA: 13/25, kernel-score=0.80122267, grad=0.0023869, 11.6s
  CELA ISTORIJA: 14/25, kernel-score=0.80113873, grad=0.0022401, 11.6s
  CELA ISTORIJA: 15/25, kernel-score=0.80108441, grad=0.0026594, 11.5s
  CELA ISTORIJA MAP epoha 15: 07 11 17 23 27 29 33
  CELA ISTORIJA: 16/25, kernel-score=0.80103412, grad=0.0022486, 11.6s
  CELA ISTORIJA: 17/25, kernel-score=0.80099049, grad=0.0021699, 11.6s
  CELA ISTORIJA: 18/25, kernel-score=0.80092592, grad=0.0022438, 11.7s
  CELA ISTORIJA: 19/25, kernel-score=0.80088266, grad=0.0028296, 11.5s
  CELA ISTORIJA: 20/25, kernel-score=0.80083102, grad=0.0026011, 11.5s
  CELA ISTORIJA MAP epoha 20: 07 11 17 21 25 27 33
  CELA ISTORIJA: 21/25, kernel-score=0.80077514, grad=0.002371, 11.5s
  CELA ISTORIJA: 22/25, kernel-score=0.80073142, grad=0.0030085, 11.4s
  CELA ISTORIJA: 23/25, kernel-score=0.80068289, grad=0.0031745, 11.4s
  CELA ISTORIJA: 24/25, kernel-score=0.80064873, grad=0.00481, 11.4s
  CELA ISTORIJA: 25/25, kernel-score=0.80063347, grad=0.0028212, 11.4s
  CELA ISTORIJA MAP epoha 25: 03 05 21 23 26 28 33
  CELA ISTORIJA: L2 promena parametara=2.57116893
  ZAVRŠNI DIJAGNOSTIČKI MAP: 03 05 21 23 26 28 33
  Z=5.97889702531e-06; zbir marginala=7.000000000000; MAP čvorovi=798,679.
  Promena marginala od inicijalizacije: 0.0359393885
  Promena marginala pri promeni konteksta=0.208196557; isti maksimum u 2/4 dodatnih konteksta.
  Navedene dijagnostike nisu označile problem. To samo po sebi ne potvrđuje prediktivnu vrednost.

NEXT [loto7_4706_k84_loto_plus_1731.csv]: 03 05 21 23 26 28 33
Interna modelska masa=7.03284607795e-06. Nije potvrđena verovatnoća budućeg izvlačenja.

Obrada sva tri CSV-a završena.
"""



"""
V1 propusti:
- Provera istih kombinacija samo ispisuje upozorenje, a rezultat ipak ostaje predstavljen kao završen NEXT.
- Provera obrazaca prepoznaje uzastopne brojeve, ali ne proverava aritmetički niz sa korakom +3.
- Nije provereno da li učenje zaista nadjačava determinističku inicijalizaciju i da li model koristi prethodno kolo. 
  Jedan memorijski kubit je dodatno ograničenje, ali još nije dokazan uzrok.

Neću uklanjati taj niz filterom niti menjati brojeve radi različitosti. 
Treba utvrditi da li obrazac postoji već pre učenja, koliko se parametri i raspodele promene i da li NEXT zavisi od konteksta.

Ispis pokazuje da su učena tri različita modela, ali su sva zadržala isti sumnjivi maksimum. 
Problem nije prosto učitavanje istog CSV-a.

CSV	    Test MMD²	Test NLL	Osetljivost na kontekst
4706	0,000901	19,6811	    0,1975
2975	0,001554	19,6203	    0,1749
1731	0,002492	20,1373	    0,1794

Različiti potpisi ulaza i modela, različite marginale i promena raspodele pri promeni konteksta opovrgavaju potpuno ignorisanje ulaza. 
Međutim, ne potvrđuju korisno učenje.
Konkretan propust: izbor modela koristi samo MMD zajedničke raspodele parova kola. 
NEXT dolazi iz uslovne raspodele, a njen NLL se računa tek na testu i ne učestvuje u izboru. 
Zato mali MMD nije sprečio problematičan uslovni maksimum.
Dodatno:
- Nisam proverio maksimum pre učenja, pa ne znamo da li je obrazac +3 već ugrađen inicijalizacijom i dinamikom kola.
- Provera osetljivosti meri samo promenu marginala; maksimum može ostati isti.
- Matematičke provere koje su prošle ne proveravaju ove nedostatke.
Tačan uzrok perioda +3 još nije utvrđen. 
Nije opravdano pripisati ga dubini tri samo zato što se brojevi poklapaju.
Ispravka treba da zadrži HQCS i MMD učenje, ali da proveri početni maksimum, prati maksimum kroz učenje i bira parametre prema uslovnoj hronološkoj validaciji. 
Bez zabrane razmaka +3 i bez forsiranja različitih kombinacija.



V2 će učiti uslovnu raspodelu sledećeg kola pomoću MMD-a, a parametre birati prema uslovnom NLL-u na validaciji. 
Uklanjam inicijalizaciju koja zavisi od broja pozicije, proširujem kvantnu memoriju i dodajem praćenje maksimuma pre i tokom učenja. 
Obrazac +3 i isti NEXT biće označeni za proveru, bez menjanja rezultata filterom.


V2 menja učenje i izbor modela:
- Uči uslovnu raspodelu sledećeg kola, uz MMD kernel na tri skale.
- Koristi dva memorijska kubita i inicijalizaciju bez zavisnosti od rednog broja pozicije.
- Parametre bira prema uslovnom NLL-u validacije.
- Prati maksimum pre i tokom učenja; proverava pravilne razmake i iste rezultate, bez filtriranja kombinacija.
Ugrađene provere uključuju potpuno nabrajanje malog problema i proveru gradijenata. 
V2 je računski zahtevnija.
"""
