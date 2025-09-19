import streamlit as st
import numpy as np
import pandas as pd
import itertools, random, json, io, zipfile, hashlib, datetime

st.set_page_config(page_title="Phase U — Hybrid + Newman Tests", layout="wide")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    dim = st.selectbox("Dimension", ["1D","2D","3D"], index=1)
    vmax = st.slider("Max vertices (vmax)", 3, 8, 6)
    mode = st.radio("Mode", ["Exhaustive","Sampled"], index=0)
    sample_size = st.number_input("Sample size PER (vA,vB)", 1, 500, 50)
    max_top_cells = st.number_input("Max top cells per complex", 1, 6, 3)
    pairs_cap = st.number_input("Pairs cap (global)", 10, 100000, 200)
    seed = st.number_input("Random seed", 0, 10**9, 42)

    st.markdown("---")
    st.subheader("Special seeds")
    use_seeds = st.checkbox("Protected seeds (S², S³)", value=True)
    use_unions = st.checkbox("Union seeds (S²∨S²)", value=True)
    use_towers = st.checkbox("Tower seeds (S²∨...∨S²)", value=True)
    use_adversarial = st.checkbox("Adversarial thin/dense", value=False)

    st.markdown("---")
    st.subheader("Newman tests")
    test_core = st.checkbox("Core normal form (odd simplex)")
    test_codim = st.checkbox("Codim-1 gluing (local star)")
    test_compression = st.checkbox("Certificate compression minimality")
    test_functorial = st.checkbox("Functorial persistence")
    test_oddcycle = st.checkbox("1D odd cycle reduction")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: run = st.button("Run")
    with col2: reset = st.button("Reset")

# ---------------- state ----------------
if "results" not in st.session_state: st.session_state.results = []
if "witnesses" not in st.session_state: st.session_state.witnesses = []
if "metadata" not in st.session_state: st.session_state.metadata = {}

if reset:
    st.session_state.results, st.session_state.witnesses, st.session_state.metadata = [], [], {}
    st.success("Cleared state.")

# ---------------- helpers ----------------
def faces_of_simplex(simplex):
    return [tuple(sorted(simplex[:i] + simplex[i+1:])) for i in range(len(simplex))]

def all_d_faces(top_simplices, d):
    S=set()
    for t in top_simplices:
        for f in itertools.combinations(t, d):
            S.add(tuple(sorted(f)))
    return sorted(S)

def enumerate_small_complexes(vcount, top_size, max_top_cells):
    verts = list(range(vcount))
    all_possible = list(itertools.combinations(verts, top_size))
    for r in range(1, min(len(all_possible), max_top_cells)+1):
        for comb in itertools.combinations(all_possible, r):
            yield [tuple(sorted(s)) for s in comb]

def build_D(all_tops, d_faces):
    D = np.zeros((len(all_tops), len(d_faces)), dtype=np.uint8)
    for i,t in enumerate(all_tops):
        T=set(t)
        for j,f in enumerate(d_faces):
            if set(f).issubset(T): D[i,j]=1
    return D

def rref_rank_mod2(A):
    M=(A.copy()%2).astype(np.uint8); rows,cols=M.shape; r=0
    for c in range(cols):
        pivot=None
        for i in range(r,rows):
            if M[i,c]==1: pivot=i; break
        if pivot is None: continue
        if pivot!=r: M[[r,pivot]]=M[[pivot,r]]
        for i in range(rows):
            if i!=r and M[i,c]==1: M[i,:]^=M[r,:]
        r+=1
        if r==rows: break
    return r,M

def solve_mod2(A,b):
    A=(A.copy()%2).astype(np.uint8); b=(b.copy()%2).astype(np.uint8)
    rows,cols=A.shape
    M=np.concatenate([A,b.reshape(-1,1)],axis=1).astype(np.uint8)
    r=0; piv_cols=[]
    for c in range(cols):
        pivot=None
        for i in range(r,rows):
            if M[i,c]==1: pivot=i; break
        if pivot is None: continue
        if pivot!=r: M[[r,pivot]]=M[[pivot,r]]
        piv_cols.append(c)
        for i in range(rows):
            if i!=r and M[i,c]==1: M[i,:]^=M[r,:]
        r+=1
        if r==rows: break
    for i in range(r,rows):
        if M[i,-1]==1: return None,r,False
    x=np.zeros(cols,dtype=np.uint8)
    for i,c in enumerate(piv_cols):
        x[c]=M[i,-1]
    return x%2,r,True

# ---- seeds ----
def seed_S2_tetra_boundary(): return [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
def seed_S3_4simplex_boundary():
    V=[0,1,2,3,4]; return [tuple(sorted([v for v in V if v!=omit])) for omit in V]
def seed_union_S2S2():
    A=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    B=[(0,4,5),(0,4,6),(0,5,6),(4,5,6)]
    return A+B
def seed_tower_S2(n):
    complexes=[]
    for k in range(n):
        offset=3*k
        T=[(offset,offset+1,offset+2),(offset,offset+1,offset+3),
           (offset,offset+2,offset+3),(offset+1,offset+2,offset+3)]
        complexes+=T
    return complexes
def seed_odd_cycle():
    return [(0,1),(1,2),(2,0)]  # 3-cycle edges

# ---------------- runner ----------------
def run_phaseU(...):
    # [keep your hybrid solver loop from before]
    # inject Newman test seeds if toggled:
    # - test_core: add single odd (d+1)-simplex
    # - test_codim: add two tops glued along a face
    # - test_compression: rerun seeds, compress faces not used in witness (log separately)
    # - test_functorial: add refinements of S²/S³ (subdivide, attach trees)
    # - test_oddcycle: for dim=1, add odd cycle edges
    ...
