import streamlit as st
import numpy as np
import pandas as pd
import itertools, random, json, io, zipfile, hashlib, datetime

# ---------------- UI ----------------
st.set_page_config(page_title="Phase U — Gluing Sanity", layout="wide")

with st.sidebar:
    st.header("Settings")
    dim = st.selectbox("Dimension", ["2D", "3D"], index=1)
    vmax = st.slider("Max vertices (vmax)", 3, 8, 6)
    mode = st.radio("Mode", ["Exhaustive", "Sampled"], index=0)
    sample_size = st.number_input("Sample size (per (vA,vB) bucket)", 1, 1000, 50)
    max_top_cells = st.number_input("Max top cells per complex", 1, 6, 3)
    pairs_cap = st.number_input("Pairs cap (global per run)", 10, 100000, 200)
    seed = st.number_input("Random seed", 0, 10**9, 42)
    allow_fusion = st.checkbox("Allow Fusion (A0 off if checked)", value=False)
    run = st.button("Run")
    reset = st.button("Reset")

# ---------------- session state ----------------
if "results" not in st.session_state: st.session_state.results = []
if "witnesses" not in st.session_state: st.session_state.witnesses = []
if "metadata"  not in st.session_state: st.session_state.metadata = {}

if reset:
    st.session_state.results = []
    st.session_state.witnesses = []
    st.session_state.metadata = {}
    st.success("Cleared.")

# ---------------- helpers ----------------
def faces_of_simplex(simplex):
    s = list(simplex)
    return [tuple(sorted(s[:i] + s[i+1:])) for i in range(len(s))]

def all_d_faces(top_simplices, d):
    S = set()
    for t in top_simplices:
        for f in itertools.combinations(t, d):
            S.add(tuple(sorted(f)))
    return sorted(S)

def enumerate_small_complexes(vcount, top_simplex_size, max_top_cells):
    verts = list(range(vcount))
    all_possible = list(itertools.combinations(verts, top_simplex_size))
    # up to max_top_cells top cells
    for r in range(1, min(len(all_possible), max_top_cells)+1):
        for comb in itertools.combinations(all_possible, r):
            yield [tuple(sorted(s)) for s in comb]

def build_D(all_tops, d_faces):
    D = np.zeros((len(all_tops), len(d_faces)), dtype=np.uint8)
    for i, t in enumerate(all_tops):
        T = set(t)
        for j, f in enumerate(d_faces):
            if set(f).issubset(T):
                D[i, j] = 1
    return D

def solve_mod2(A, b):
    A = A.copy() % 2; b = b.copy() % 2
    rows, cols = A.shape
    M = np.concatenate([A, b.reshape(-1,1)], axis=1).astype(np.uint8)
    r = 0
    piv_cols = []
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] == 1:
                pivot = i; break
        if pivot is None: continue
        M[[r, pivot]] = M[[pivot, r]]
        piv_cols.append(c)
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i, :] ^= M[r, :]
        r += 1
        if r == rows: break
    # inconsistent?
    for i in range(r, rows):
        if M[i, -1] == 1: return None
    x = np.zeros(cols, dtype=np.uint8)
    # back-sub not needed for one solution in F2; take pivot rows
    for i, c in enumerate(piv_cols):
        x[c] = M[i, -1]
    return x % 2

def run_phaseU(dim, vmax, mode, sample_size, max_top_cells, pairs_cap, seed):
    random.seed(seed)
    d = 2 if dim=="2D" else 3
    top_size = d+1

    # build complexes registry keyed by vertex count
    reg = {v: list(enumerate_small_complexes(v, top_size, max_top_cells))
           for v in range(top_size, vmax+1)}

    # make (vA,vB) buckets
    vbuckets = []
    for vA in reg:
        for vB in reg:
            if len(reg[vA]) and len(reg[vB]):
                vbuckets.append((vA, vB))

    # choose pairs
    pairs = []
    if mode == "Exhaustive":
        for vA, vB in vbuckets:
            for A in reg[vA]:
                for B in reg[vB]:
                    pairs.append((A, B, vA, vB))
                    if len(pairs) >= pairs_cap: break
                if len(pairs) >= pairs_cap: break
            if len(pairs) >= pairs_cap: break
    else:
        # Sampled: up to sample_size per bucket, but respect pairs_cap
        for vA, vB in vbuckets:
            pool = list(itertools.product(reg[vA], reg[vB]))
            random.shuffle(pool)
            take = min(sample_size, len(pool))
            for A, B in pool[:take]:
                pairs.append((A, B, vA, vB))
                if len(pairs) >= pairs_cap: break
            if len(pairs) >= pairs_cap: break

    results = []
    witnesses = []
    id_counter = 0

    for topA, topB, vA, vB in pairs:
        id_counter += 1
        # discover interfaces (label-based)
        facesA = set(f for t in topA for f in faces_of_simplex(t))
        facesB = set(f for t in topB for f in faces_of_simplex(t))
        vertsA = set(v for t in topA for v in t)
        vertsB = set(v for t in topB for v in t)
        common_vertices = sorted(vertsA & vertsB)

        iface_map = {
            "face": [f for f in facesA if f in facesB and len(f)==d],
            "edge": [f for f in facesA if f in facesB and len(f)==2],
            "vertex": [(v,) for v in common_vertices]
        }

        all_tops = list(topA) + list(topB)
        d_faces_global = all_d_faces(all_tops, d)  # G0: global φ allowed
        D = build_D(all_tops, d_faces_global)
        b = np.ones(len(all_tops), dtype=np.uint8)

        for iface_type, cands in iface_map.items():
            if not cands:  # still record a 'no interface' scenario?
                continue
            # take up to 5 candidates per type to keep runs quick
            sample_cands = cands if len(cands) <= 5 else random.sample(cands, 5)
            for cand in sample_cands:
                sol = solve_mod2(D, b)
                result = "SUCCESS" if sol is not None else "FAIL"

                notes = ""
                # anomaly G0: (a) not actually shared (shouldn't happen given construction)
                if iface_type == "face" and (cand not in facesA or cand not in facesB):
                    notes = "SUSPICIOUS: interface not shared"
                # (b) success but D empty
                if result=="SUCCESS" and (D.size == 0 or D.shape[1]==0):
                    notes = "SUSPICIOUS: success with empty D"
                # (c) face=no though both sides use that face
                if iface_type=="face" and result=="FAIL" and (cand in facesA and cand in facesB):
                    notes = "SUSPICIOUS: face=no with common face"

                witness_json = None
                if result=="SUCCESS":
                    phi_cols = [j for j, v in enumerate(sol) if v==1]
                    phi_faces = [d_faces_global[j] for j in phi_cols]
                    # parity check: for each top cell, odd count
                    parity_ok = True
                    for i, t in enumerate(all_tops):
                        cnt = sum(1 for f in phi_faces if set(f).issubset(set(t)))
                        if cnt % 2 != 1:
                            parity_ok = False; break
                    if not parity_ok:
                        notes = (notes+" | " if notes else "") + "SUSPICIOUS: parity check failed"
                    witness_json = {
                        "topsA": topA, "topsB": topB,
                        "interface_type": iface_type, "interface": cand,
                        "phi_faces": phi_faces,
                        "parity_ok": parity_ok,
                        "timestamp": datetime.datetime.utcnow().isoformat()+"Z"
                    }
                    witnesses.append(("UG-%d" % id_counter, witness_json))

                row = {
                    "ID": f"UG-{id_counter}",
                    "vA": len(vertsA),
                    "vB": len(vertsB),
                    "topsA_count": len(topA),
                    "topsB_count": len(topB),
                    "interface_type": iface_type,
                    "interface": str(cand),
                    "result": result,
                    "notes": notes
                }
                results.append(row)

    # metadata
    success_count = sum(r["result"]=="SUCCESS" for r in results)
    fail_count    = sum(r["result"]=="FAIL" for r in results)
    meta = {
        "dimension": dim, "vmax": vmax, "mode": mode,
        "sample_size": int(sample_size), "max_top_cells": int(max_top_cells),
        "pairs_cap": int(pairs_cap), "seed": int(seed),
        "allow_fusion": bool(allow_fusion),
        "admissibility": "G0 (global φ)",
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "row_count": len(results),
        "success_count": int(success_count),
        "fail_count": int(fail_count)
    }
    sha_payload = json.dumps({"meta": meta, "results": results}, sort_keys=True).encode()
    meta["sha1_hash"] = hashlib.sha1(sha_payload).hexdigest()
    return results, witnesses, meta

# ---------------- run ----------------
if run:
    results, witnesses, meta = run_phaseU(dim, vmax, mode, sample_size, max_top_cells, pairs_cap, seed)
    st.session_state.results = results
    st.session_state.witnesses = witnesses
    st.session_state.metadata = meta
    st.success(f"Done. Rows={len(results)} | SUCCESS={meta['success_count']} | FAIL={meta['fail_count']}")

# ---------------- show table & downloads ----------------
left, right = st.columns([3,2])

with left:
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df, use_container_width=True, height=500)

with right:
    if st.session_state.results:
        # CSV
        csv_buf = io.StringIO()
        pd.DataFrame(st.session_state.results).to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), "phaseU_G0_results.csv", mime="text/csv")

        # ZIP (witnesses + metadata + summary)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # witnesses
            for wid, wj in st.session_state.witnesses:
                zf.writestr(f"{wid}_witness.json", json.dumps(wj, indent=2))
            # metadata
            zf.writestr("run_metadata.json", json.dumps(st.session_state.metadata, indent=2))
            # summary csv
            zf.writestr("results_summary.csv", csv_buf.getvalue())
        st.download_button("Download Witnesses (ZIP)", zip_buf.getvalue(), "witnesses_bundle.zip", mime="application/zip")
    else:
        st.info("No results yet. Set parameters and press Run.")
