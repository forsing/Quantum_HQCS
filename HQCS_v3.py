#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
HQCS Loto 7/39 — V3 

Deterministička klasična simulacija kvantnog modela.
Kvantne kapije: RY, RZ, CNOT.
Klasični optimizator: Adam.
Klasična pretraga: grananje sa gornjim granicama.

Prethodno kolo uslovljava kvantnu memoriju.
Radni kubit zadržava prethodni opaženi bit.

Legalne putanje zadržavaju stvarne kvantne operatore i na granicama.
Posle sedmog izabranog broja preostale nule nisu identitet:
njihove amplitude učestvuju u masi cele putanje.

P(y | kontekst, legalnost) = ||A_y psi(kontekst)||² / Z(kontekst).

Isti Z koristi se u:
- uslovnom NLL;
- uslovnom MMD;
- marginalama;
- izboru maksimuma.

Nema slučajnih inicijalizacija, uzorkovanja, mešanja redova,
klasičnog ML regresora, rangiranja učestalosti ili filtera kombinacija.

Parametri i broj epoha biraju se hronološkom validacijom.
Test se koristi samo za izveštaj.
Završno učenje koristi sve istorijske prelaze.

NEXT je maksimum naučene raspodele, a ne potvrđena buduća kombinacija.
"""


import csv
import hashlib
import itertools
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
    from qiskit.quantum_info import Operator
except ImportError as exc:
    raise SystemExit(
        "Potrebni su numpy, torch, numba i qiskit.\n"
        f"{exc}"
    )

PATHS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_2975.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_plus_1731.csv",
)

N, K = 39, 7
MEMORY, DEPTH = 2, 4
BATCH = 64

# Unapred definisani kandidati; bira ih samo validacija.
LEARNING_RATES = (0.01, 0.003)
MAX_EPOCHS = 120
CHECK_EVERY = 5
MIN_EPOCHS = 20
PATIENCE = 5
MMD_WEIGHT = 1.0
KERNEL_SCALES = (1.0, 4.0, 16.0)

REAL = torch.float64
COMPLEX = torch.complex128

torch.set_default_dtype(REAL)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def fingerprint(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()
    ).hexdigest()[:16]


def read_draws(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not x.strip() for x in fields):
                continue
            if len(fields) != K:
                raise ValueError(f"{path}, red {line}: očekivano 7 polja.")
            try:
                row = [int(x.strip()) for x in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc
            require(
                len(set(row)) == K and all(1 <= x <= N for x in row),
                f"{path}, red {line}: nelegalna kombinacija {row}.",
            )
            rows.append(sorted(row))

    require(len(rows) >= 100, f"{path}: potrebno najmanje 100 kola.")
    draws = np.asarray(rows, dtype=np.int64)
    bits = np.zeros((len(draws), N), dtype=np.int64)
    bits[np.arange(len(draws))[:, None], draws - 1] = 1
    return draws, bits


def embedded_pauli(single, qubit, nq):
    a = np.ones((1, 1), dtype=np.complex128)
    for q in reversed(range(nq)):
        a = np.kron(a, single if q == qubit else np.eye(2))
    return torch.tensor(a, dtype=COMPLEX)


def cnot(nq, control, target):
    a = torch.zeros((1 << nq, 1 << nq), dtype=COMPLEX)
    for col in range(1 << nq):
        row = col ^ ((1 << target) if ((col >> control) & 1) else 0)
        a[row, col] = 1
    return a


class HQCS(torch.nn.Module):
    def __init__(self, n=N, k=K, memory=MEMORY, depth=DEPTH):
        super().__init__()
        self.n, self.k = n, k
        self.memory, self.depth = memory, depth
        self.nq = memory + 1
        self.d = 1 << memory

        # Deterministička inicijalizacija; nema slučajnog šuma.
        theta = torch.empty((2*n, depth, self.nq, 2), dtype=REAL)
        for layer in range(depth):
            for q in range(self.nq):
                theta[:, layer, q, 0] = 0.70 + 0.11*q + 0.07*layer
                theta[:, layer, q, 1] = 0.23 + 0.09*q - 0.04*layer

        self.theta = torch.nn.Parameter(theta)
        self.count_theta = torch.nn.Parameter(
            torch.zeros((k+1, depth, self.nq, 2), dtype=REAL)
        )

        self.register_buffer(
            "eye", torch.eye(1 << self.nq, dtype=COMPLEX)
        )
        y = np.array([[0, -1j], [1j, 0]])
        z = np.array([[1, 0], [0, -1]])
        self.register_buffer(
            "ys", torch.stack([
                embedded_pauli(y, q, self.nq) for q in range(self.nq)
            ])
        )
        self.register_buffer(
            "zs", torch.stack([
                embedded_pauli(z, q, self.nq) for q in range(self.nq)
            ])
        )

        self.edges = []
        entanglers = []
        for layer in range(depth):
            edges = [
                (q, q+1)
                for q in range(layer % 2, self.nq-1, 2)
            ]
            self.edges.append(edges)
            u = self.eye.clone()
            for control, target in edges:
                u = cnot(self.nq, control, target) @ u
            entanglers.append(u)
        self.register_buffer("entanglers", torch.stack(entanglers))

        t = torch.arange(n)[:, None, None]
        r = torch.arange(k+1)[None, :, None]
        x = torch.arange(2)[None, None, :]
        # Samo legalnost, bez zamene dozvoljenih operatora identitetom.
        allowed = (r >= x) & (r-x <= n-t-1)
        self.register_buffer("allowed", allowed)

    def rotation(self, angle, pauli):
        return (
            torch.cos(angle/2)[:, None, None]*self.eye
            - 1j*torch.sin(angle/2)[:, None, None]*pauli
        )

    def unitaries(self, angles):
        u = self.eye[None].expand(len(angles), -1, -1)
        for layer in range(self.depth):
            for q in range(self.nq):
                u = self.rotation(angles[:, layer, q, 0], self.ys[q]) @ u
                u = self.rotation(angles[:, layer, q, 1], self.zs[q]) @ u
            u = self.entanglers[layer] @ u
        return u

    def split(self, u):
        return torch.stack([
            torch.stack([
                u[:, x*self.d:(x+1)*self.d,
                  b*self.d:(b+1)*self.d]
                for b in (0, 1)
            ], dim=1)
            for x in (0, 1)
        ], dim=1)

    def operators(self):
        encoder = self.split(self.unitaries(self.theta[:self.n]))
        angles = (
            self.theta[self.n:, None] + self.count_theta[None]
        ).reshape(self.n*(self.k+1), self.depth, self.nq, 2)
        raw = self.split(self.unitaries(angles)).reshape(
            self.n, self.k+1, 2, 2, self.d, self.d
        )
        decoder = raw*self.allowed[:, :, :, None, None, None]
        return encoder, decoder

    def circuit(self, t, remaining=None):
        qc = QuantumCircuit(self.nq)
        if remaining is None:
            angles = self.theta[t]
        else:
            angles = self.theta[self.n+t] + self.count_theta[remaining]
        angles = angles.detach().numpy()
        for layer in range(self.depth):
            for q in range(self.nq):
                qc.ry(float(angles[layer, q, 0]), q)
                qc.rz(float(angles[layer, q, 1]), q)
            for control, target in self.edges[layer]:
                qc.cx(control, target)
        return qc


def encode(encoder, context, empty=False):
    state = torch.zeros((len(context), encoder.shape[-1]), dtype=COMPLEX)
    state[:, 0] = 1
    previous = torch.zeros(len(context), dtype=torch.long)
    if empty:
        return state, previous

    for t in range(context.shape[1]):
        x = context[:, t]
        state = torch.einsum("bij,bj->bi", encoder[t, x, previous], state)
        norm = state.abs().square().sum(1)
        require(
            torch.all(torch.isfinite(norm) & (norm > 0)).item(),
            "Nulta ili neispravna masa konteksta.",
        )
        state = state/torch.sqrt(norm)[:, None]
        previous = x
    return state, previous


def quadratic(state, effect):
    return torch.einsum(
        "bi,bij,bj->b", state.conj(), effect, state
    ).real


def single_root(decoder, k):
    d = decoder.shape[-1]
    following = torch.zeros((2, k+1, d, d), dtype=COMPLEX)
    following[:, 0] = torch.eye(d, dtype=COMPLEX)

    for t in reversed(range(len(decoder))):
        outputs = []
        for b in (0, 1):
            value = torch.zeros_like(following[0])
            for x in (0, 1):
                tail = following[x]
                if x:
                    tail = torch.cat(
                        (torch.zeros_like(tail[:1]), tail[:-1]), dim=0
                    )
                a = decoder[t, :, x, b]
                value = value + a.mH @ tail @ a
            outputs.append(value)
        following = torch.stack(outputs)
    return following[:, k]


def pair_pullback(effect, a, b):
    count, d = a.shape[0], a.shape[-1]
    h = effect.reshape(count, count, d, d, d, d)
    h = torch.einsum("rai,rsabcd->rsibcd", a.conj(), h)
    h = torch.einsum("sbj,rsibcd->rsijcd", b.conj(), h)
    h = torch.einsum("rck,rsijcd->rsijkd", a, h)
    h = torch.einsum("sdl,rsijkd->rsijkl", b, h)
    return h.reshape(count, count, d*d, d*d)


def pair_step(following, operators, gamma):
    outputs = []
    for b in (0, 1):
        for c in (0, 1):
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
                    value = value + math.exp(-gamma*(x-y)**2)*pair_pullback(
                        tail, operators[:, x, b], operators[:, y, c]
                    )
            outputs.append(value)
    return torch.stack(outputs)


def pair_root(decoder, k, gamma):
    d = decoder.shape[-1]
    following = torch.zeros(
        (4, k+1, k+1, d*d, d*d), dtype=COMPLEX
    )
    following[:, 0, 0] = torch.eye(d*d, dtype=COMPLEX)
    for t in reversed(range(len(decoder))):
        if torch.is_grad_enabled() and decoder.requires_grad:
            following = checkpoint(
                pair_step, following, decoder[t], gamma,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            following = pair_step(following, decoder[t], gamma)
    return following[:, k, k]


def cross_moment(decoder, state, previous, target, k, gamma):
    density = state[:, :, None]*state.conj()[:, None, :]
    work = torch.nn.functional.one_hot(previous, 2).to(REAL)
    count = torch.zeros(k+1, dtype=REAL)
    count[k] = 1
    rho = (
        work[:, :, None, None, None]
        * count[None, None, :, None, None]
        * density[:, None, None]
    )

    for t in range(len(decoder)):
        outputs = []
        for x in (0, 1):
            value = sum(
                decoder[t, :, x, b] @ rho[:, b]
                @ decoder[t, :, x, b].mH
                for b in (0, 1)
            )
            if x:
                value = torch.cat(
                    (value[:, 1:], torch.zeros_like(value[:, :1])), dim=1
                )
            weight = torch.exp(-gamma*(target[:, t].to(REAL)-x).square())
            outputs.append(value*weight[:, None, None, None])
        rho = torch.stack(outputs, dim=1)
    return rho[:, :, 0].diagonal(dim1=-2, dim2=-1).sum((-1, -2)).real


def raw_nll(decoder, state, previous, target, k):
    remaining = torch.full((len(target),), k, dtype=torch.long)
    result = torch.zeros(len(target), dtype=REAL)
    for t in range(target.shape[1]):
        x = target[:, t]
        state = torch.einsum(
            "bij,bj->bi", decoder[t, remaining, x, previous], state
        )
        norm = state.abs().square().sum(1)
        require(
            torch.all(torch.isfinite(norm) & (norm > 0)).item(),
            "Nulta ili neispravna masa ciljne putanje.",
        )
        result = result-torch.log(norm)
        state = state/torch.sqrt(norm)[:, None]
        remaining = remaining-x
        previous = x
    require(torch.all(remaining == 0).item(), "Nelegalna ciljna putanja.")
    return result


def normalizer(state, previous, root):
    z = quadratic(state, root[previous])
    require(
        torch.all(torch.isfinite(z) & (z > 1e-250) & (z <= 1+1e-8)).item(),
        "Neispravan ili numerički premali Z; rezultat se ne zamenjuje.",
    )
    return z


def losses(e, d, root, pair, contexts, targets, k, gammas):
    state, previous = encode(e, contexts)
    z = normalizer(state, previous, root)

    # -log(masa/Z) = -log(masa) + log(Z).
    nll = raw_nll(d, state, previous, targets, k)+torch.log(z)

    joint_state = (
        state[:, :, None]*state[:, None, :]
    ).reshape(len(state), -1)
    qq = quadratic(joint_state, pair[3*previous])/z.square()
    qy = sum(
        cross_moment(d, state, previous, targets, k, g) for g in gammas
    )/(len(gammas)*z)
    mmd = qq-2*qy+1

    require(
        torch.all(torch.isfinite(nll) & torch.isfinite(mmd)).item(),
        "Nefinitan gubitak.",
    )
    require(float(nll.detach().min()) >= -1e-7, "Negativan uslovni NLL.")
    require(float(mmd.detach().min()) >= -1e-7, "Negativan MMD.")
    return nll, mmd


def full_gradient(model, contexts, targets, gammas):
    model.zero_grad(set_to_none=True)
    graph_e, graph_d = model.operators()
    e = graph_e.detach().requires_grad_(True)
    d = graph_d.detach().requires_grad_(True)

    graph_root = single_root(d, model.k)
    graph_pair = sum(
        pair_root(d, model.k, g) for g in gammas
    )/len(gammas)
    root = graph_root.detach().requires_grad_(True)
    pair = graph_pair.detach().requires_grad_(True)

    total_nll = total_mmd = 0.0
    count = len(contexts)
    for start in range(0, count, BATCH):
        stop = min(start+BATCH, count)
        nll, mmd = losses(
            e, d, root, pair, contexts[start:stop], targets[start:stop],
            model.k, gammas,
        )
        ((nll+MMD_WEIGHT*mmd).sum()/count).backward()
        total_nll += float(nll.detach().sum())/count
        total_mmd += float(mmd.detach().sum())/count

    require(root.grad is not None and pair.grad is not None,
            "Nedostaje gradijent normalizacije ili MMD-a.")
    graph_root.backward(root.grad)
    graph_pair.backward(pair.grad)
    require(e.grad is not None and d.grad is not None,
            "Nedostaje gradijent operatora.")
    torch.autograd.backward((graph_e, graph_d), (e.grad, d.grad))

    for parameter in model.parameters():
        require(
            parameter.grad is not None
            and torch.all(torch.isfinite(parameter.grad)).item(),
            "Neispravan gradijent parametara.",
        )
    return total_nll, total_mmd


@torch.no_grad()
def evaluate(model, contexts, targets, gammas=None, empty=False):
    e, d = model.operators()
    root = single_root(d, model.k)
    pair = None
    if gammas is not None:
        pair = sum(pair_root(d, model.k, g) for g in gammas)/len(gammas)

    nlls, scores = [], []
    for start in range(0, len(contexts), BATCH):
        xc = contexts[start:start+BATCH]
        yt = targets[start:start+BATCH]
        if gammas is None:
            state, previous = encode(e, xc, empty=empty)
            z = normalizer(state, previous, root)
            nll = raw_nll(d, state, previous, yt, model.k)+torch.log(z)
        else:
            require(not empty, "MMD ablation nije implementirana.")
            nll, mmd = losses(
                e, d, root, pair, xc, yt, model.k, gammas
            )
            scores.extend(mmd.tolist())
        nlls.extend(nll.tolist())
    return np.asarray(nlls), np.asarray(scores)


def choose_kernel(targets):
    x = np.asarray(targets, dtype=np.float64)
    histogram = np.zeros(x.shape[1]+1, dtype=np.int64)
    sums = x.sum(1)
    for start in range(0, len(x), 256):
        block = x[start:start+256]
        distances = (
            sums[start:start+256, None]+sums[None, :]
            - 2*block @ x.T
        )
        distances = np.rint(distances).astype(np.int64)
        require(np.all(distances >= 0), "Negativna Hamming udaljenost.")
        histogram += np.bincount(
            distances.ravel(), minlength=len(histogram)
        )
    median = int(np.searchsorted(
        np.cumsum(histogram), (int(histogram.sum())+1)//2
    ))
    base = math.log(2)/max(1, median)
    return tuple(base*s for s in KERNEL_SCALES)


def np_encode(e, context):
    state = np.zeros(e.shape[-1], dtype=np.complex128)
    state[0] = 1
    previous = 0
    for t, x in enumerate(context):
        x = int(x)
        state = e[t, x, previous] @ state
        norm = np.linalg.norm(state)
        require(norm > 0 and np.isfinite(norm), "Neispravan kontekst.")
        state /= norm
        previous = x
    return state, previous


def path_mass(d, bits, state, previous, k):
    state = state.copy()
    remaining = k
    for t, x in enumerate(bits):
        x = int(x)
        if remaining < 0 or remaining >= d.shape[1]:
            return 0.0
        state = d[t, remaining, x, previous] @ state
        remaining -= x
        previous = x
    return float(np.vdot(state, state).real) if remaining == 0 else 0.0


def suffix_effects(d, k):
    n, _, _, _, size, _ = d.shape
    suffix = np.zeros((n+1, k+1, 2, size, size), dtype=np.complex128)
    suffix[n, 0] = np.eye(size)
    for t in reversed(range(n)):
        for r in range(k+1):
            for b in (0, 1):
                value = np.zeros((size, size), dtype=np.complex128)
                for x in (0, 1):
                    if r >= x:
                        a = d[t, r, x, b]
                        value += a.conj().T @ suffix[t+1, r-x, x] @ a
                suffix[t, r, b] = (value+value.conj().T)/2
    return suffix


def audit(d, state, previous, k, suffix=None):
    if suffix is None:
        suffix = suffix_effects(d, k)
    n, _, _, _, size, _ = d.shape
    z = float(np.vdot(state, suffix[0, k, previous] @ state).real)
    require(np.isfinite(z) and 1e-250 < z <= 1+1e-8, "Neispravan Z.")

    rho = np.zeros((2, k+1, size, size), dtype=np.complex128)
    rho[previous, k] = np.outer(state, state.conj())
    marginals = np.zeros(n)
    for t in range(n):
        following = np.zeros_like(rho)
        for b in (0, 1):
            for r in range(k+1):
                for x in (0, 1):
                    if r < x:
                        continue
                    a = d[t, r, x, b]
                    density = a @ rho[b, r] @ a.conj().T
                    following[x, r-x] += density
                    if x:
                        marginals[t] += float(np.trace(
                            suffix[t+1, r-1, 1] @ density
                        ).real)
        rho = following

    forward_z = sum(float(np.trace(rho[b, 0]).real) for b in (0, 1))
    np.testing.assert_allclose(forward_z/z, 1.0, atol=2e-8, rtol=0)
    marginals /= z
    np.testing.assert_allclose(marginals.sum(), k, atol=2e-7, rtol=0)
    require(
        np.all(marginals >= -1e-8) and np.all(marginals <= 1+1e-8),
        "Neispravne marginale.",
    )
    return suffix, z, marginals


@njit(cache=False)
def matvec(a, v):
    out = np.zeros(len(v), dtype=np.complex128)
    for i in range(len(v)):
        for j in range(len(v)):
            out[i] += a[i, j]*v[j]
    return out


@njit(cache=False)
def effect_value(a, v):
    out = matvec(a, v)
    value = 0.0
    for i in range(len(v)):
        value += (v[i].conjugate()*out[i]).real
    return value


@njit(cache=False)
def find_map(d, suffix, state, previous, k):
    n, size = len(d), len(state)
    capacity = 2*n+4
    states = np.zeros((capacity, size), dtype=np.complex128)
    positions = np.zeros(capacity, dtype=np.int64)
    remaining = np.zeros(capacity, dtype=np.int64)
    prev = np.zeros(capacity, dtype=np.int64)
    masks = np.zeros(capacity, dtype=np.int64)
    bounds = np.zeros(capacity)

    states[0] = state
    remaining[0] = k
    prev[0] = previous
    bounds[0] = effect_value(suffix[0, k, previous], state)

    # Dodatak čini odsecanje konzervativnijim.
    tolerance = abs(bounds[0])*1e-11
    top, visited = 1, 0
    best, best_mask = -1.0, np.int64(0)

    while top:
        top -= 1
        t, r, b = positions[top], remaining[top], prev[top]
        mask, bound = masks[top], bounds[top]
        vector = states[top].copy()
        visited += 1

        if best >= 0 and bound+tolerance < best:
            continue
        if t == n:
            if r == 0:
                mass = 0.0
                for j in range(size):
                    mass += (vector[j].conjugate()*vector[j]).real
                if mass > best or (mass == best and mask < best_mask):
                    best, best_mask = mass, mask
            continue

        children = np.zeros((2, size), dtype=np.complex128)
        child_bounds = np.zeros(2)
        valid = np.zeros(2, dtype=np.int64)
        for x in (0, 1):
            if 0 <= r-x <= n-t-1:
                children[x] = matvec(d[t, r, x, b], vector)
                child_bounds[x] = effect_value(
                    suffix[t+1, r-x, x], children[x]
                )
                valid[x] = 1

        first = 0 if child_bounds[0] <= child_bounds[1] else 1
        for order in (0, 1):
            x = first if order == 0 else 1-first
            if not valid[x]:
                continue
            states[top] = children[x]
            positions[top] = t+1
            remaining[top] = r-x
            prev[top] = x
            masks[top] = mask | (np.int64(x) << t)
            bounds[top] = child_bounds[x]
            top += 1

    return best_mask, best, visited


def mask_bits(mask, n):
    return np.asarray([(int(mask) >> i) & 1 for i in range(n)], dtype=np.int64)


@torch.no_grad()
def check_qiskit(model):
    e, d = model.operators()
    e, d = e.numpy(), d.numpy()
    allowed = model.allowed.numpy()
    for t in range(model.n):
        u = np.asarray(Operator(model.circuit(t)).data)
        for x in (0, 1):
            for b in (0, 1):
                block = u[
                    x*model.d:(x+1)*model.d,
                    b*model.d:(b+1)*model.d,
                ]
                np.testing.assert_allclose(e[t, x, b], block, atol=5e-12)

        for r in range(model.k+1):
            u = np.asarray(Operator(model.circuit(t, r)).data)
            for x in (0, 1):
                for b in (0, 1):
                    block = u[
                        x*model.d:(x+1)*model.d,
                        b*model.d:(b+1)*model.d,
                    ]*allowed[t, r, x]
                    np.testing.assert_allclose(d[t, r, x, b], block, atol=5e-12)


def legal_outputs(n, k):
    rows = []
    for selected in itertools.combinations(range(n), k):
        row = np.zeros(n, dtype=np.int64)
        row[list(selected)] = 1
        rows.append(row)
    return np.asarray(rows)


def self_checks():
    print("Provere: Qiskit, uslovna normalizacija, NLL, MMD, gradijenti, MAP.",
          flush=True)
    gammas = (0.13, 0.52, 2.08)

    for k in range(5):
        model = HQCS(n=4, k=k, memory=1, depth=4)
        check_qiskit(model)
        with torch.no_grad():
            e, d = model.operators()
        en, dn = e.numpy(), d.numpy()
        rows = legal_outputs(4, k)
        context = rows[0]
        state, previous = np_encode(en, context)
        suffix, z, marginal = audit(dn, state, previous, k)
        masses = np.asarray([
            path_mass(dn, row, state, previous, k) for row in rows
        ])
        p = masses/z
        np.testing.assert_allclose(p.sum(), 1.0, atol=2e-10)
        np.testing.assert_allclose(p @ rows, marginal, atol=2e-10)

        mask, maximum, _ = find_map(dn, suffix, state, previous, k)
        np.testing.assert_allclose(maximum/z, p.max(), atol=2e-10)
        require(mask_bits(mask, 4).sum() == k, "Nelegalni MAP u proveri.")

        for illegal_mask in range(1 << 4):
            row = mask_bits(illegal_mask, 4)
            if row.sum() != k:
                require(path_mass(dn, row, state, previous, k) == 0,
                        "Nelegalna putanja ima masu.")

    model = HQCS(n=4, k=2, memory=1, depth=4)
    rows = legal_outputs(4, 2)
    xc = torch.tensor(rows[[0, 2]], dtype=torch.long)
    yt = torch.tensor(rows[[3, 5]], dtype=torch.long)
    e, d = model.operators()
    root = single_root(d, 2)
    pair = sum(pair_root(d, 2, g) for g in gammas)/len(gammas)
    nll, mmd = losses(e, d, root, pair, xc, yt, 2, gammas)

    en, dn = e.detach().numpy(), d.detach().numpy()
    distances = np.abs(rows[:, None]-rows[None, :]).sum(2)
    gram = sum(np.exp(-g*distances) for g in gammas)/len(gammas)

    expected_nll, expected_mmd = [], []
    for context, target in zip(xc.numpy(), yt.numpy()):
        state, previous = np_encode(en, context)
        masses = np.asarray([
            path_mass(dn, row, state, previous, 2) for row in rows
        ])
        p = masses/masses.sum()
        index = int(np.flatnonzero(np.all(rows == target, axis=1))[0])
        expected_nll.append(-math.log(p[index]))
        expected_mmd.append(p @ gram @ p-2*p @ gram[:, index]+1)

    np.testing.assert_allclose(nll.detach().numpy(), expected_nll, atol=2e-9)
    np.testing.assert_allclose(mmd.detach().numpy(), expected_mmd, atol=2e-9)

    (nll+MMD_WEIGHT*mmd).mean().backward()
    direct = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    full_gradient(model, xc, yt, gammas)
    for name, parameter in model.named_parameters():
        np.testing.assert_allclose(
            parameter.grad.numpy(), direct[name].numpy(),
            atol=2e-8, rtol=2e-5,
        )

    parameters = dict(model.named_parameters())
    for name, index in (
        ("theta", (1, 0, 0, 0)),
        ("theta", (5, 1, 1, 0)),
        ("count_theta", (0, 0, 1, 0)),
        ("count_theta", (2, 1, 0, 0)),
    ):
        parameter = parameters[name]
        original = float(parameter[index].detach())
        values = []
        eps = 1e-6
        try:
            for delta in (eps, -eps):
                with torch.no_grad():
                    parameter[index] = original+delta
                a, b = evaluate(model, xc, yt, gammas)
                values.append(a.mean()+MMD_WEIGHT*b.mean())
        finally:
            with torch.no_grad():
                parameter[index] = original
        np.testing.assert_allclose(
            float(direct[name][index]), (values[0]-values[1])/(2*eps),
            atol=2e-6, rtol=3e-3,
        )

    print("Ugrađene numeričke provere prošle.", flush=True)


def train_step(model, optimizer, xc, yt, gammas):
    nll, mmd = full_gradient(model, xc, yt, gammas)
    optimizer.step()
    for parameter in model.parameters():
        require(torch.all(torch.isfinite(parameter)).item(),
                "Nefinitni parametri nakon optimizacije.")
    return nll, mmd


def select_configuration(xc, yt, train_end, val_end, gammas):
    train_x, train_y = xc[:train_end], yt[:train_end]
    val_x, val_y = xc[train_end:val_end], yt[train_end:val_end]

    initial_model = HQCS()
    initial_score = evaluate(initial_model, val_x, val_y)[0].mean()
    best = {
        "lr": LEARNING_RATES[0],
        "epochs": 0,
        "score": float(initial_score),
        "boundary": False,
    }
    print(f"Početni validation NLL={initial_score:.8f}", flush=True)

    for lr in LEARNING_RATES:
        model = HQCS()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        local_best = float(initial_score)
        local_epoch = 0
        stale = 0

        for epoch in range(1, MAX_EPOCHS+1):
            started = time.perf_counter()
            train_nll, train_mmd = train_step(
                model, optimizer, train_x, train_y, gammas
            )
            elapsed = time.perf_counter()-started
            if epoch == 1 or epoch % CHECK_EVERY == 0:
                print(
                    f"lr={lr:g}, epoha={epoch}, train NLL={train_nll:.7f}, "
                    f"MMD={train_mmd:.7f}, {elapsed:.1f}s",
                    flush=True,
                )
            if epoch % CHECK_EVERY:
                continue

            score = float(evaluate(model, val_x, val_y)[0].mean())
            print(f"  validation NLL={score:.8f}", flush=True)
            if score < local_best-1e-6:
                local_best, local_epoch, stale = score, epoch, 0
            else:
                stale += 1

            if score < best["score"]-1e-6:
                best = {
                    "lr": lr, "epochs": epoch, "score": score,
                    "boundary": epoch == MAX_EPOCHS,
                }
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break

        print(
            f"lr={lr:g}: najbolja epoha={local_epoch}, "
            f"validation NLL={local_best:.8f}",
            flush=True,
        )

    require(
        best["epochs"] > 0,
        "Nijedan obučeni kandidat nije poboljšao početnu validaciju. "
        "Neobučen model neće biti izdat kao završna predikcija.",
    )
    print(
        f"IZBOR: lr={best['lr']:g}, epoha={best['epochs']}, "
        f"validation NLL={best['score']:.8f}",
        flush=True,
    )
    if best["boundary"]:
        print("UPOZORENJE: optimum je na granici budžeta; "
              "konvergencija nije potvrđena.", flush=True)
    return best


def fit_all(xc, yt, gammas, configuration, label):
    model = HQCS()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configuration["lr"]
    )
    for epoch in range(1, configuration["epochs"]+1):
        nll, mmd = train_step(model, optimizer, xc, yt, gammas)
        if epoch == 1 or epoch % CHECK_EVERY == 0 or epoch == configuration["epochs"]:
            print(
                f"{label}: {epoch}/{configuration['epochs']}, "
                f"NLL={nll:.7f}, MMD={mmd:.7f}",
                flush=True,
            )
    return model


def longest_arithmetic(numbers):
    dp = [dict() for _ in numbers]
    longest = ()
    for i in range(len(numbers)):
        for j in range(i):
            step = numbers[i]-numbers[j]
            candidate = dp[j].get(step, (numbers[j],))+(numbers[i],)
            if len(candidate) > len(dp[i].get(step, ())):
                dp[i][step] = candidate
            if len(candidate) > len(longest):
                longest = candidate
    return longest


@torch.no_grad()
def predict(model, bits, draws):
    e, d = model.operators()
    en, dn = e.numpy(), d.numpy()
    state, previous = np_encode(en, bits[-1])
    suffix, z, marginals = audit(dn, state, previous, K)

    print("Pretraga maksimuma cele legalne raspodele...", flush=True)
    mask, mass, visited = find_map(dn, suffix, state, previous, K)
    selected = mask_bits(mask, N)
    numbers = tuple((np.flatnonzero(selected)+1).tolist())
    require(len(numbers) == K, "MAP nema sedam brojeva.")

    independent_mass = path_mass(dn, selected, state, previous, K)
    np.testing.assert_allclose(
        mass/z, independent_mass/z, atol=1e-12, rtol=2e-8
    )
    require(0 < mass/z <= 1+1e-8, "Neispravna uslovna masa maksimuma.")

    messages = []
    arithmetic = longest_arithmetic(numbers)
    if len(arithmetic) >= 5:
        messages.append(f"Aritmetički podniz: {arithmetic}.")
    if all(x <= 19 for x in numbers):
        messages.append("Svih sedam brojeva je u 1–19.")
    if all(x >= 20 for x in numbers):
        messages.append("Svih sedam brojeva je u 20–39.")
    span = numbers[-1]-numbers[0]
    if span <= float(np.quantile(draws[:, -1]-draws[:, 0], 0.01)):
        messages.append("Raspon je u donjih 1% istorijskih raspona.")
    repeats = int(np.all(draws == np.asarray(numbers), axis=1).sum())
    if repeats:
        messages.append(f"Kombinacija se pojavila {repeats} puta u istoriji.")

    sensitivity = 0.0
    for row in bits[-5:-1]:
        other_state, other_previous = np_encode(en, row)
        _, _, other_marginals = audit(
            dn, other_state, other_previous, K, suffix
        )
        sensitivity = max(
            sensitivity, float(np.max(np.abs(other_marginals-marginals)))
        )
    if sensitivity < 1e-6:
        messages.append("Raspodela gotovo ne reaguje na proverene kontekste.")

    print(
        f"Sirova legalna masa Z={z:.12g}; "
        f"normalizovana legalna masa=1; "
        f"zbir marginala={marginals.sum():.10f}; "
        f"MAP čvorovi={visited:,}.",
        flush=True,
    )
    print(f"Osetljivost marginala={sensitivity:.8g}", flush=True)
    for message in messages:
        print("DIJAGNOSTIKA: "+message, flush=True)

    return {
        "numbers": numbers,
        "mass": mass/z,
        "marginals": marginals,
        "messages": messages,
        "model_hash": fingerprint(dn),
    }


def process(path):
    print("\n"+Path(path).name, flush=True)
    draws, bits = read_draws(path)
    xc = torch.tensor(bits[:-1], dtype=torch.long)
    yt = torch.tensor(bits[1:], dtype=torch.long)
    total = len(xc)
    train_end, val_end = int(total*0.70), int(total*0.85)
    print(
        f"Kola={len(bits)}; prelazi={total}; train={train_end}; "
        f"validation={val_end-train_end}; test={total-val_end}.",
        flush=True,
    )

    # Kernel koristi samo trening deo.
    gammas = choose_kernel(bits[1:train_end+1])
    configuration = select_configuration(
        xc, yt, train_end, val_end, gammas
    )

    test_model = fit_all(
        xc[:val_end], yt[:val_end], gammas, configuration, "TRAIN+VALIDATION"
    )
    test_nll, test_mmd = evaluate(
        test_model, xc[val_end:], yt[val_end:], gammas
    )
    empty_nll, _ = evaluate(
        test_model, xc[val_end:], yt[val_end:], empty=True
    )
    print(
        f"TEST: NLL={test_nll.mean():.8f}; MMD={test_mmd.mean():.8f}",
        flush=True,
    )
    print(
        f"ABLACIJA ISTOG MODELA SA PRAZNOM MEMORIJOM: "
        f"NLL={empty_nll.mean():.8f}. "
        "Ovo nije zasebno obučen kontrolni model.",
        flush=True,
    )
    del test_model

    # Test rezultat ne menja konfiguraciju.
    final_model = fit_all(
        xc, yt, gammas, configuration, "CELA ISTORIJA"
    )
    check_qiskit(final_model)
    result = predict(final_model, bits, draws)
    result["file"] = Path(path).name
    result["input_hash"] = fingerprint(bits)

    if configuration["boundary"]:
        result["messages"].append("Konvergencija nije potvrđena.")
    if empty_nll.mean() <= test_nll.mean():
        result["messages"].append(
            "Test ablation ne potvrđuje korist konteksta."
        )

    print(
        "NEXT ["+result["file"]+"]: "
        +" ".join(f"{x:02d}" for x in result["numbers"]),
        flush=True,
    )
    print(f"Interna uslovna masa={result['mass']:.12g}", flush=True)
    if result["messages"]:
        print(
            "STATUS: model ima dijagnostička upozorenja; "
            "NEXT nije potvrđen kao kvalitetna predikcija.",
            flush=True,
        )
    else:
        print(
            "STATUS: navedene dijagnostike nisu označile problem; "
            "to nije dokaz prediktivne vrednosti.",
            flush=True,
        )
    return result


def main():
    print(
        "HQCS Loto 7/39 — V3 ispravka\n"
        "Kvantni model, deterministička klasična simulacija, Adam.\n"
        "Uslovni NLL + MMD; puna masa legalne putanje.\n"
        "Bez slučajnih kombinacija i kozmetičkih filtera.",
        flush=True,
    )
    for path in PATHS:
        require(Path(path).is_file(), f"Nedostaje CSV: {path}")

    self_checks()
    results = []
    failures = []

    for path in PATHS:
        try:
            results.append(process(path))
        except Exception as exc:
            failures.append((Path(path).name, str(exc)))
            print(
                f"NEUSPEH [{Path(path).name}]: {type(exc).__name__}: {exc}\n"
                "Zamenska kombinacija nije generisana.",
                file=sys.stderr, flush=True,
            )

    for i, left in enumerate(results):
        for right in results[i+1:]:
            if left["numbers"] == right["numbers"]:
                difference = float(np.max(np.abs(
                    left["marginals"]-right["marginals"]
                )))
                print(
                    "\nISTI NEXT — POTREBNA PROVERA MODELA:\n"
                    f"{left['file']} / {right['file']}\n"
                    f"Ulazi: {left['input_hash']} / {right['input_hash']}\n"
                    f"Modeli: {left['model_hash']} / {right['model_hash']}\n"
                    f"Maksimalna razlika marginala={difference:.9g}\n"
                    "Rezultati nisu menjani da bi bili različiti.",
                    flush=True,
                )

    if failures:
        print("\nObrada nije uspešno završena za sve CSV-ove.", flush=True)
        return 1
    print("\nObrada sva tri CSV-a završena.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena pretraga ne potvrđuje maksimum.",
            file=sys.stderr,
        )
        sys.exit(130)



"""
HQCS Loto 7/39 — V3 
Kvantni model, deterministička klasična simulacija, Adam.
Uslovni NLL + MMD; puna masa legalne putanje.
Bez slučajnih kombinacija i kozmetičkih filtera.
Provere: Qiskit, uslovna normalizacija, NLL, MMD, gradijenti, MAP.
Ugrađene numeričke provere prošle.


loto7_4706_k84.csv
Kola=4706; prelazi=4705; train=3293; validation=706; test=706.
Početni validation NLL=18.61513054
lr=0.01, epoha=1, train NLL=18.5051136, MMD=0.8034079, 14.6s
lr=0.01, epoha=5, train NLL=17.3404710, MMD=0.8026718, 14.2s
  validation NLL=17.40327498
lr=0.01, epoha=10, train NLL=16.9265151, MMD=0.8019300, 14.1s
  validation NLL=17.00221720
lr=0.01, epoha=15, train NLL=16.7072421, MMD=0.8016626, 15.0s
  validation NLL=16.78004993
lr=0.01, epoha=20, train NLL=16.6112532, MMD=0.8015180, 14.2s
  validation NLL=16.72734043
lr=0.01, epoha=25, train NLL=16.5661357, MMD=0.8014161, 14.3s
  validation NLL=16.69788893
lr=0.01, epoha=30, train NLL=16.5306722, MMD=0.8013368, 14.3s
  validation NLL=16.66275319
lr=0.01, epoha=35, train NLL=16.5041229, MMD=0.8012452, 14.2s
  validation NLL=16.64641845
lr=0.01, epoha=40, train NLL=16.4817735, MMD=0.8011733, 14.3s
  validation NLL=16.65136749
lr=0.01, epoha=45, train NLL=16.4647050, MMD=0.8011292, 14.2s
  validation NLL=16.66595056
lr=0.01, epoha=50, train NLL=16.4496783, MMD=0.8010935, 14.1s
  validation NLL=16.68267602
lr=0.01, epoha=55, train NLL=16.4368311, MMD=0.8010770, 14.5s
  validation NLL=16.69970021
lr=0.01, epoha=60, train NLL=16.4274320, MMD=0.8010738, 13.8s
  validation NLL=16.69711563
lr=0.01: najbolja epoha=35, validation NLL=16.64641845
lr=0.003, epoha=1, train NLL=18.5051136, MMD=0.8034079, 14.0s
lr=0.003, epoha=5, train NLL=17.8755181, MMD=0.8028624, 14.0s
  validation NLL=17.88029507
lr=0.003, epoha=10, train NLL=17.3939450, MMD=0.8024386, 13.8s
  validation NLL=17.49642738
lr=0.003, epoha=15, train NLL=17.2178826, MMD=0.8022114, 13.9s
  validation NLL=17.34927706
lr=0.003, epoha=20, train NLL=17.0558103, MMD=0.8020735, 13.7s
  validation NLL=17.20628061
lr=0.003, epoha=25, train NLL=16.9212131, MMD=0.8018948, 13.7s
  validation NLL=17.09416680
lr=0.003, epoha=30, train NLL=16.8239010, MMD=0.8017550, 13.7s
  validation NLL=16.99057171
lr=0.003, epoha=35, train NLL=16.7420783, MMD=0.8016422, 13.7s
  validation NLL=16.90923840
lr=0.003, epoha=40, train NLL=16.6794902, MMD=0.8015584, 13.8s
  validation NLL=16.85756347
lr=0.003, epoha=45, train NLL=16.6344413, MMD=0.8015010, 13.8s
  validation NLL=16.82034515
lr=0.003, epoha=50, train NLL=16.6002342, MMD=0.8014564, 13.8s
  validation NLL=16.78958438
lr=0.003, epoha=55, train NLL=16.5730745, MMD=0.8014123, 13.7s
  validation NLL=16.76771596
lr=0.003, epoha=60, train NLL=16.5520985, MMD=0.8013723, 13.6s
  validation NLL=16.75328763
lr=0.003, epoha=65, train NLL=16.5357167, MMD=0.8013503, 13.7s
  validation NLL=16.73859506
lr=0.003, epoha=70, train NLL=16.5213925, MMD=0.8013135, 13.7s
  validation NLL=16.72592712
lr=0.003, epoha=75, train NLL=16.5078908, MMD=0.8012774, 13.7s
  validation NLL=16.71960380
lr=0.003, epoha=80, train NLL=16.4971866, MMD=0.8012585, 13.7s
  validation NLL=16.71492955
lr=0.003, epoha=85, train NLL=16.4873315, MMD=0.8012281, 13.8s
  validation NLL=16.70894101
lr=0.003, epoha=90, train NLL=16.4797146, MMD=0.8012252, 13.7s
  validation NLL=16.70375134
lr=0.003, epoha=95, train NLL=16.4723506, MMD=0.8012120, 13.7s
  validation NLL=16.70034488
lr=0.003, epoha=100, train NLL=16.4652962, MMD=0.8011953, 13.6s
  validation NLL=16.69767843
lr=0.003, epoha=105, train NLL=16.4580183, MMD=0.8011823, 13.8s
  validation NLL=16.69687538
lr=0.003, epoha=110, train NLL=16.4529841, MMD=0.8011787, 13.7s
  validation NLL=16.69918368
lr=0.003, epoha=115, train NLL=16.4468145, MMD=0.8011555, 13.8s
  validation NLL=16.69407123
lr=0.003, epoha=120, train NLL=16.4405551, MMD=0.8011405, 13.8s
  validation NLL=16.69461768
lr=0.003: najbolja epoha=115, validation NLL=16.69407123
IZBOR: lr=0.01, epoha=35, validation NLL=16.64641845
TRAIN+VALIDATION: 1/35, NLL=18.5245364, MMD=0.8034005
TRAIN+VALIDATION: 5/35, NLL=17.3271281, MMD=0.8026111
TRAIN+VALIDATION: 10/35, NLL=16.9351714, MMD=0.8019273
TRAIN+VALIDATION: 15/35, NLL=16.7133432, MMD=0.8016615
TRAIN+VALIDATION: 20/35, NLL=16.6348946, MMD=0.8015159
TRAIN+VALIDATION: 25/35, NLL=16.5955223, MMD=0.8014282
TRAIN+VALIDATION: 30/35, NLL=16.5589435, MMD=0.8013618
TRAIN+VALIDATION: 35/35, NLL=16.5327123, MMD=0.8012720
TEST: NLL=16.69186498; MMD=0.80180076
ABLACIJA ISTOG MODELA SA PRAZNOM MEMORIJOM: NLL=16.82537340. Ovo nije zasebno obučen kontrolni model.
CELA ISTORIJA: 1/35, NLL=18.5187930, MMD=0.8034506
CELA ISTORIJA: 5/35, NLL=17.3223463, MMD=0.8025602
CELA ISTORIJA: 10/35, NLL=16.9461953, MMD=0.8019686
CELA ISTORIJA: 15/35, NLL=16.6947944, MMD=0.8016887
CELA ISTORIJA: 20/35, NLL=16.6260442, MMD=0.8015816
CELA ISTORIJA: 25/35, NLL=16.5906255, MMD=0.8014819
CELA ISTORIJA: 30/35, NLL=16.5615094, MMD=0.8013973
CELA ISTORIJA: 35/35, NLL=16.5373079, MMD=0.8013369
Pretraga maksimuma cele legalne raspodele...
Sirova legalna masa Z=1.47782247186e-05; normalizovana legalna masa=1; zbir marginala=7.0000000000; MAP čvorovi=2,332,522.
Osetljivost marginala=0.086902477
DIJAGNOSTIKA: Aritmetički podniz: (28, 30, 32, 34, 36, 38).
DIJAGNOSTIKA: Svih sedam brojeva je u 20–39.
DIJAGNOSTIKA: Raspon je u donjih 1% istorijskih raspona.
NEXT [loto7_4706_k84.csv]: 27 28 30 32 34 36 38
Interna uslovna masa=1.71689725691e-06
STATUS: model ima dijagnostička upozorenja; NEXT nije potvrđen kao kvalitetna predikcija.



loto7_4706_k84_loto_2975.csv
Kola=2975; prelazi=2974; train=2081; validation=446; test=447.
Početni validation NLL=18.66095240
lr=0.01, epoha=1, train NLL=18.5705745, MMD=0.8035034, 10.9s
lr=0.01, epoha=5, train NLL=17.2784093, MMD=0.8024547, 10.8s
  validation NLL=17.43658840
lr=0.01, epoha=10, train NLL=16.9078482, MMD=0.8019302, 10.8s
  validation NLL=17.10544255
lr=0.01, epoha=15, train NLL=16.6798799, MMD=0.8015420, 10.8s
  validation NLL=16.88948200
lr=0.01, epoha=20, train NLL=16.5954196, MMD=0.8014279, 10.9s
  validation NLL=16.81360897
lr=0.01, epoha=25, train NLL=16.5454360, MMD=0.8013416, 10.9s
  validation NLL=16.75272614
lr=0.01, epoha=30, train NLL=16.5146278, MMD=0.8013173, 10.8s
  validation NLL=16.76073121
lr=0.01, epoha=35, train NLL=16.4837321, MMD=0.8012100, 10.8s
  validation NLL=16.73706064
lr=0.01, epoha=40, train NLL=16.4535276, MMD=0.8011144, 10.9s
  validation NLL=16.72002394
lr=0.01, epoha=45, train NLL=16.4273119, MMD=0.8010507, 10.8s
  validation NLL=16.70668644
lr=0.01, epoha=50, train NLL=16.4026213, MMD=0.8009729, 10.7s
  validation NLL=16.70800495
lr=0.01, epoha=55, train NLL=16.3792262, MMD=0.8008738, 10.8s
  validation NLL=16.70792859
lr=0.01, epoha=60, train NLL=16.3569293, MMD=0.8008169, 10.8s
  validation NLL=16.72114750
lr=0.01, epoha=65, train NLL=16.3357309, MMD=0.8007738, 10.8s
  validation NLL=16.74958228
lr=0.01, epoha=70, train NLL=16.3151509, MMD=0.8007551, 10.8s
  validation NLL=16.76588232
lr=0.01: najbolja epoha=45, validation NLL=16.70668644
lr=0.003, epoha=1, train NLL=18.5705745, MMD=0.8035034, 10.8s
lr=0.003, epoha=5, train NLL=17.8973363, MMD=0.8028407, 10.9s
  validation NLL=17.95118122
lr=0.003, epoha=10, train NLL=17.3649983, MMD=0.8023671, 10.8s
  validation NLL=17.54116801
lr=0.003, epoha=15, train NLL=17.1608868, MMD=0.8021464, 10.8s
  validation NLL=17.44368279
lr=0.003, epoha=20, train NLL=17.0025598, MMD=0.8019991, 10.8s
  validation NLL=17.32801695
lr=0.003, epoha=25, train NLL=16.8738676, MMD=0.8017665, 10.8s
  validation NLL=17.18212165
lr=0.003, epoha=30, train NLL=16.7738070, MMD=0.8016160, 10.8s
  validation NLL=17.07380408
lr=0.003, epoha=35, train NLL=16.6935938, MMD=0.8015215, 10.8s
  validation NLL=16.98079242
lr=0.003, epoha=40, train NLL=16.6325750, MMD=0.8014138, 10.8s
  validation NLL=16.90534287
lr=0.003, epoha=45, train NLL=16.5886560, MMD=0.8013453, 10.8s
  validation NLL=16.85250646
lr=0.003, epoha=50, train NLL=16.5521868, MMD=0.8012702, 10.8s
  validation NLL=16.82327812
lr=0.003, epoha=55, train NLL=16.5219459, MMD=0.8011924, 10.9s
  validation NLL=16.79240230
lr=0.003, epoha=60, train NLL=16.4979133, MMD=0.8011433, 10.9s
  validation NLL=16.79019431
lr=0.003, epoha=65, train NLL=16.4762620, MMD=0.8010786, 10.8s
  validation NLL=16.78441773
lr=0.003, epoha=70, train NLL=16.4585653, MMD=0.8010493, 10.8s
  validation NLL=16.78397356
lr=0.003, epoha=75, train NLL=16.4422318, MMD=0.8010166, 10.8s
  validation NLL=16.78384537
lr=0.003, epoha=80, train NLL=16.4284107, MMD=0.8009694, 10.9s
  validation NLL=16.77685186
lr=0.003, epoha=85, train NLL=16.4151156, MMD=0.8009350, 10.8s
  validation NLL=16.77920522
lr=0.003, epoha=90, train NLL=16.4029802, MMD=0.8009047, 10.8s
  validation NLL=16.78119625
lr=0.003, epoha=95, train NLL=16.3914838, MMD=0.8008756, 10.8s
  validation NLL=16.78682606
lr=0.003, epoha=100, train NLL=16.3808818, MMD=0.8008472, 10.8s
  validation NLL=16.79089214
lr=0.003, epoha=105, train NLL=16.3713761, MMD=0.8008153, 10.8s
  validation NLL=16.79449898
lr=0.003: najbolja epoha=80, validation NLL=16.77685186
IZBOR: lr=0.01, epoha=45, validation NLL=16.70668644
TRAIN+VALIDATION: 1/45, NLL=18.5865257, MMD=0.8035059
TRAIN+VALIDATION: 5/45, NLL=17.3539683, MMD=0.8025856
TRAIN+VALIDATION: 10/45, NLL=16.9390517, MMD=0.8019306
TRAIN+VALIDATION: 15/45, NLL=16.7118880, MMD=0.8016883
TRAIN+VALIDATION: 20/45, NLL=16.6232769, MMD=0.8014835
TRAIN+VALIDATION: 25/45, NLL=16.5654555, MMD=0.8013920
TRAIN+VALIDATION: 30/45, NLL=16.5194966, MMD=0.8012876
TRAIN+VALIDATION: 35/45, NLL=16.4880888, MMD=0.8012064
TRAIN+VALIDATION: 40/45, NLL=16.4571367, MMD=0.8011267
TRAIN+VALIDATION: 45/45, NLL=16.4349292, MMD=0.8010908
TEST: NLL=16.76168881; MMD=0.80191502
ABLACIJA ISTOG MODELA SA PRAZNOM MEMORIJOM: NLL=16.89557243. Ovo nije zasebno obučen kontrolni model.
CELA ISTORIJA: 1/45, NLL=18.5783127, MMD=0.8035494
CELA ISTORIJA: 5/45, NLL=17.3386939, MMD=0.8025692
CELA ISTORIJA: 10/45, NLL=16.9436495, MMD=0.8018877
CELA ISTORIJA: 15/45, NLL=16.7126647, MMD=0.8015866
CELA ISTORIJA: 20/45, NLL=16.6188097, MMD=0.8014330
CELA ISTORIJA: 25/45, NLL=16.5657410, MMD=0.8013028
CELA ISTORIJA: 30/45, NLL=16.5263498, MMD=0.8012459
CELA ISTORIJA: 35/45, NLL=16.4951860, MMD=0.8011820
CELA ISTORIJA: 40/45, NLL=16.4767830, MMD=0.8011410
CELA ISTORIJA: 45/45, NLL=16.4590263, MMD=0.8011107
Pretraga maksimuma cele legalne raspodele...
Sirova legalna masa Z=1.44859954623e-05; normalizovana legalna masa=1; zbir marginala=7.0000000000; MAP čvorovi=2,768,987.
Osetljivost marginala=0.1148029
DIJAGNOSTIKA: Aritmetički podniz: (17, 19, 21, 23, 25, 27).
NEXT [loto7_4706_k84_loto_2975.csv]: 17 19 21 23 25 27 34
Interna uslovna masa=1.39907668842e-06
STATUS: model ima dijagnostička upozorenja; NEXT nije potvrđen kao kvalitetna predikcija.



loto7_4706_k84_loto_plus_1731.csv
Kola=1731; prelazi=1730; train=1211; validation=259; test=260.
Početni validation NLL=18.72583036
lr=0.01, epoha=1, train NLL=18.4408079, MMD=0.8033288, 8.3s
lr=0.01, epoha=5, train NLL=17.0971293, MMD=0.8020044, 8.2s
  validation NLL=17.34431003
lr=0.01, epoha=10, train NLL=16.7461675, MMD=0.8014396, 8.2s
  validation NLL=17.22220479
lr=0.01, epoha=15, train NLL=16.5247662, MMD=0.8011515, 8.2s
  validation NLL=16.95632757
lr=0.01, epoha=20, train NLL=16.4461629, MMD=0.8011117, 8.3s
  validation NLL=16.90560186
lr=0.01, epoha=25, train NLL=16.3765168, MMD=0.8010291, 8.0s
  validation NLL=16.90136066
lr=0.01, epoha=30, train NLL=16.3179661, MMD=0.8009227, 7.9s
  validation NLL=16.88583337
lr=0.01, epoha=35, train NLL=16.2643528, MMD=0.8008742, 8.0s
  validation NLL=16.86494311
lr=0.01, epoha=40, train NLL=16.2187379, MMD=0.8008397, 7.9s
  validation NLL=16.89873469
lr=0.01, epoha=45, train NLL=16.1794074, MMD=0.8007462, 7.9s
  validation NLL=16.93845425
lr=0.01, epoha=50, train NLL=16.1466970, MMD=0.8006738, 7.9s
  validation NLL=16.96124046
lr=0.01, epoha=55, train NLL=16.1179943, MMD=0.8006185, 8.0s
  validation NLL=16.97792334
lr=0.01, epoha=60, train NLL=16.0948269, MMD=0.8005679, 8.1s
  validation NLL=16.97170841
lr=0.01: najbolja epoha=35, validation NLL=16.86494311
lr=0.003, epoha=1, train NLL=18.4408079, MMD=0.8033288, 8.0s
lr=0.003, epoha=5, train NLL=17.7913958, MMD=0.8026935, 8.0s
  validation NLL=17.98332136
lr=0.003, epoha=10, train NLL=17.2544905, MMD=0.8021477, 8.0s
  validation NLL=17.52767612
lr=0.003, epoha=15, train NLL=16.9770092, MMD=0.8017865, 7.9s
  validation NLL=17.32313286
lr=0.003, epoha=20, train NLL=16.8061653, MMD=0.8016077, 7.9s
  validation NLL=17.23352906
lr=0.003, epoha=25, train NLL=16.6765811, MMD=0.8014229, 7.9s
  validation NLL=17.17402271
lr=0.003, epoha=30, train NLL=16.5772110, MMD=0.8012881, 8.0s
  validation NLL=17.10901653
lr=0.003, epoha=35, train NLL=16.5058354, MMD=0.8012030, 7.9s
  validation NLL=17.05369091
lr=0.003, epoha=40, train NLL=16.4521496, MMD=0.8011414, 7.9s
  validation NLL=17.02124936
lr=0.003, epoha=45, train NLL=16.4089844, MMD=0.8010789, 7.9s
  validation NLL=17.00077048
lr=0.003, epoha=50, train NLL=16.3724150, MMD=0.8010299, 7.9s
  validation NLL=16.99842743
lr=0.003, epoha=55, train NLL=16.3438619, MMD=0.8010170, 7.9s
  validation NLL=16.99395736
lr=0.003, epoha=60, train NLL=16.3174498, MMD=0.8009762, 7.9s
  validation NLL=16.99486182
lr=0.003, epoha=65, train NLL=16.2947918, MMD=0.8009449, 7.9s
  validation NLL=16.99946300
lr=0.003, epoha=70, train NLL=16.2741207, MMD=0.8009134, 7.9s
  validation NLL=17.01232237
lr=0.003, epoha=75, train NLL=16.2562619, MMD=0.8008837, 7.9s
  validation NLL=17.00074361
lr=0.003, epoha=80, train NLL=16.2436922, MMD=0.8008694, 8.0s
  validation NLL=17.00423129
lr=0.003: najbolja epoha=55, validation NLL=16.99395736
IZBOR: lr=0.01, epoha=35, validation NLL=16.86494311
TRAIN+VALIDATION: 1/35, NLL=18.4910261, MMD=0.8034036
TRAIN+VALIDATION: 5/35, NLL=17.1856527, MMD=0.8022313
TRAIN+VALIDATION: 10/35, NLL=16.8072761, MMD=0.8018110
TRAIN+VALIDATION: 15/35, NLL=16.5980881, MMD=0.8014728
TRAIN+VALIDATION: 20/35, NLL=16.5179498, MMD=0.8013586
TRAIN+VALIDATION: 25/35, NLL=16.4548660, MMD=0.8012169
TRAIN+VALIDATION: 30/35, NLL=16.4117133, MMD=0.8011337
TRAIN+VALIDATION: 35/35, NLL=16.3623353, MMD=0.8010174
TEST: NLL=16.74099810; MMD=0.80182540
ABLACIJA ISTOG MODELA SA PRAZNOM MEMORIJOM: NLL=17.24030154. Ovo nije zasebno obučen kontrolni model.
CELA ISTORIJA: 1/35, NLL=18.4795177, MMD=0.8034422
CELA ISTORIJA: 5/35, NLL=17.1836817, MMD=0.8022392
CELA ISTORIJA: 10/35, NLL=16.8210587, MMD=0.8018015
CELA ISTORIJA: 15/35, NLL=16.6117358, MMD=0.8014254
CELA ISTORIJA: 20/35, NLL=16.5272900, MMD=0.8012994
CELA ISTORIJA: 25/35, NLL=16.4657021, MMD=0.8012449
CELA ISTORIJA: 30/35, NLL=16.4207322, MMD=0.8011827
CELA ISTORIJA: 35/35, NLL=16.3801568, MMD=0.8010427
Pretraga maksimuma cele legalne raspodele...
Sirova legalna masa Z=1.17389809056e-05; normalizovana legalna masa=1; zbir marginala=7.0000000000; MAP čvorovi=2,591,598.
Osetljivost marginala=0.20046729
DIJAGNOSTIKA: Aritmetički podniz: (16, 18, 20, 22, 24, 26).
NEXT [loto7_4706_k84_loto_plus_1731.csv]: 16 18 20 22 24 26 39
Interna uslovna masa=1.5527542485e-06
STATUS: model ima dijagnostička upozorenja; NEXT nije potvrđen kao kvalitetna predikcija.


Obrada sva tri CSV-a završena.
"""



"""
V3: preostale pozicije više ne dobijaju automatski masu 1 nakon sedmog broja. 
Cela legalna putanja ulazi u račun, a ista normalizacija koristi se u učenju, proveri i izboru NEXT-a. 
Početni model učestvuje u validacionom poređenju.
"""
