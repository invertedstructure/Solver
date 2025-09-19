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
def run_phaseU(dim,vmax,mode,sample_size,max_top_cells,pairs_cap,seed,
               use_seeds,use_unions,use_towers,use_adversarial,
               test_core,test_codim,test_functorial,test_oddcycle):

    random.seed(seed)
    d=1 if dim=="1D" else (2 if dim=="2D" else 3)
    top_size=d+1
    reg={v:list(enumerate_small_complexes(v,top_size,max_top_cells)) for v in range(top_size,vmax+1)}
    vbuckets=[(vA,vB) for vA in reg for vB in reg if len(reg[vA]) and len(reg[vB])]
    pairs=[]
    if mode=="Exhaustive":
        for vA,vB in vbuckets:
            for A in reg[vA]:
                for B in reg[vB]:
                    pairs.append((A,B,vA,vB,"PROC"))
                    if len(pairs)>=pairs_cap: break
                if len(pairs)>=pairs_cap: break
            if len(pairs)>=pairs_cap: break
    else:
        for vA,vB in vbuckets:
            pool=list(itertools.product(reg[vA],reg[vB])); random.shuffle(pool)
            take=min(sample_size,len(pool))
            for A,B in pool[:take]:
                pairs.append((A,B,vA,vB,"PROC"))
                if len(pairs)>=pairs_cap: break
            if len(pairs)>=pairs_cap: break

    # protected seeds
    if use_seeds:
        if dim=="2D": pairs.append((seed_S2_tetra_boundary(),seed_S2_tetra_boundary(),4,4,"SEED_S2"))
        if dim=="3D": pairs.append((seed_S3_4simplex_boundary(),seed_S3_4simplex_boundary(),5,5,"SEED_S3"))
    if use_unions and dim=="2D":
        pairs.append((seed_union_S2S2(),seed_union_S2S2(),7,7,"SEED_UNION_S2S2"))
    if use_towers and dim=="2D":
        for n in [2,3]:
            T=seed_tower_S2(n)
            v=len({v for t in T for v in t})
            pairs.append((T,T,v,v,f"SEED_TOWER_S2x{n}"))
    if use_adversarial and dim=="2D":
        thin=[(0,1,2),(1,2,3),(2,3,4)]
        dense=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
        pairs.append((thin,thin,5,5,"ADV_THIN"))
        pairs.append((dense,dense,4,4,"ADV_DENSE"))

    # Newman injections
    if test_core and dim=="2D":
        pairs.append(([(0,1,2)],[(0,1,2)],3,3,"NEWMAN_CORE2"))
    if test_core and dim=="3D":
        pairs.append(([(0,1,2,3)],[(0,1,2,3)],4,4,"NEWMAN_CORE3"))
    if test_codim and dim=="2D":
        pairs.append(([(0,1,2),(0,1,3)],[(0,1,2),(0,2,3)],4,4,"NEWMAN_CODIM2"))
    if test_codim and dim=="3D":
        pairs.append(([(0,1,2,3),(0,1,2,4)],[(0,1,2,3),(0,1,3,4)],5,5,"NEWMAN_CODIM3"))
    if test_functorial and dim=="2D":
        pairs.append((seed_S2_tetra_boundary()+[(0,1,4)],seed_S2_tetra_boundary(),5,4,"NEWMAN_FUNCT2"))
    if test_functorial and dim=="3D":
        pairs.append((seed_S3_4simplex_boundary()+[(0,1,2,5)],seed_S3_4simplex_boundary(),6,5,"NEWMAN_FUNCT3"))
    if test_oddcycle and dim=="1D":
        pairs.append((seed_odd_cycle(),seed_odd_cycle(),3,3,"NEWMAN_ODDCYCLE"))

    results,witnesses=[],[]
    id_counter=0
    for topA,topB,vA,vB,src in pairs:
        id_counter+=1
        facesA=set(f for t in topA for f in faces_of_simplex(t))
        facesB=set(f for t in topB for f in faces_of_simplex(t))
        vertsA={v for t in topA for v in t}; vertsB={v for t in topB for v in t}
        common_vertices=sorted(vertsA&vertsB)
        iface_map={
            "face":[f for f in facesA if f in facesB and len(f)==d],
            "edge":[f for f in facesA if f in facesB and len(f)==2],
            "vertex":[(v,) for v in common_vertices]
        }
        all_tops=list(topA)+list(topB)
        d_faces=all_d_faces(all_tops,d)
        D=build_D(all_tops,d_faces)
        rankD,_=rref_rank_mod2(D); nullity=D.shape[1]-rankD

        for iface_type,cands in iface_map.items():
            if not cands: continue
            sample_cands=cands if len(cands)<=5 else random.sample(cands,5)
            for cand in sample_cands:
                u=np.zeros(len(all_tops),dtype=np.uint8)
                for i,t in enumerate(topA):
                    if set(cand).issubset(set(t)): u[i]=1
                for j,t in enumerate(topB):
                    if set(cand).issubset(set(t)): u[len(topA)+j]^=1
                b=u

                x,rank_solve,in_row_space=solve_mod2(D,b)

                if dim=="3D":
                    b_all=np.ones(len(all_tops),dtype=np.uint8)
                    _,_,in_row_space2=solve_mod2(D,b_all)
                    if not in_row_space2:
                        in_row_space=False

                result="SUCCESS" if in_row_space else "FAIL"
                min_support=int(x.sum()) if (x is not None and in_row_space) else ""

                notes=[]
                if result=="SUCCESS" and (D.size==0 or D.shape[1]==0):
                    notes.append("SUSPICIOUS: success with empty D")

                witness_json=None
                if result=="SUCCESS":
                    phi_cols=[j for j,v in enumerate(x) if v==1]
                    phi_faces=[d_faces[j] for j in phi_cols]
                    parity_ok=True
                    for idx,t in enumerate(all_tops):
                        cnt=sum(1 for f in phi_faces if set(f).issubset(set(t)))
                        if (cnt%2)!=int(b[idx]): parity_ok=False; break
                    if not parity_ok: notes.append("SUSPICIOUS: parity check failed")
                    witness_json={"topsA":topA,"topsB":topB,"interface_type":iface_type,
                                  "interface":cand,"phi_faces":phi_faces,"parity_ok":parity_ok,
                                  "timestamp":datetime.datetime.utcnow().isoformat()+"Z"}
                    witnesses.append((f"UG-{id_counter}",witness_json))

                row={"ID":f"UG-{id_counter}","source":src,"vA":len(vertsA),"vB":len(vertsB),
                     "topsA_count":len(topA),"topsB_count":len(topB),
                     "rows":int(D.shape[0]),"cols":int(D.shape[1]),
                     "rank":int(rankD),"nullity":int(nullity),
                     "in_row_space":bool(in_row_space),
                     "min_support_est":min_support,
                     "interface_type":iface_type,"interface":str(cand),
                     "result":result,"notes":" | ".join(notes)}
                results.append(row)

    success_count=sum(r["result"]=="SUCCESS" for r in results)
    fail_count=sum(r["result"]=="FAIL" for r in results)
    meta={"dimension":dim,"vmax":vmax,"mode":mode,"sample_size":int(sample_size),
          "max_top_cells":int(max_top_cells),"pairs_cap":int(pairs_cap),"seed":int(seed),
          "admissibility":"Hybrid G0 + Newman","timestamp":datetime.datetime.utcnow().isoformat()+"Z",
          "row_count":len(results),"success_count":int(success_count),"fail_count":int(fail_count)}
    sha_payload=json.dumps({"meta":meta,"results":results},sort_keys=True).encode()
    meta["sha1_hash"]=hashlib.sha1(sha_payload).hexdigest()
    return results,witnesses,meta

# ---------------- run ----------------
if run:
    results,witnesses,meta=run_phaseU(dim,vmax,mode,sample_size,max_top_cells,pairs_cap,seed,
        use_seeds,use_unions,use_towers,use_adversarial,
        test_core,test_codim,test_functorial,test_oddcycle)
    st.session_state.results,st.session_state.witnesses,st.session_state.metadata=results,witnesses,meta
    st.success(f"Done. Rows={len(results)} | SUCCESS={meta['success_count']} | FAIL={meta['fail_count']}")

# ---------------- table & downloads ----------------
left,right=st.columns([3,2])
with left:
    df=pd.DataFrame(st.session_state.results)
    if not df.empty:
        preferred=["ID","source","vA","vB","topsA_count","topsB_count",
                   "rows","cols","rank","nullity","in_row_space","min_support_est",
                   "interface_type","interface","result","notes"]
        cols=[c for c in preferred if c in df.columns]+[c for c in df.columns if c not in preferred]
        df=df[cols]
    st.dataframe(df,use_container_width=True,height=560)

with right:
    if st.session_state.results:
        csv_buf=io.StringIO()
        pd.DataFrame(st.session_state.results).to_csv(csv_buf,index=False)
        st.download_button("Download CSV",csv_buf.getvalue(),"phaseU_results.csv",mime="text/csv")
        zip_buf=io.BytesIO()
        with zipfile.ZipFile(zip_buf,"w",zipfile.ZIP_DEFLATED) as zf:
            for wid,wj in st.session_state.witnesses:
                zf.writestr(f"{wid}_witness.json",json.dumps(wj,indent=2))
            zf.writestr("run_metadata.json",json.dumps(st.session_state.metadata,indent=2))
            zf.writestr("results_summary.csv",csv_buf.getvalue())
        st.download_button("Download Witnesses (ZIP)",zip_buf.getvalue(),
                           "witnesses_bundle.zip",mime="application/zip")
    else:
        st.info("No results yet. Set parameters and press Run.")
