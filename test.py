def db_conn():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, charset="utf8mb4"
    )
def fetch_completed_order_ids(conn) -> Set[int]:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT MaDDH
            FROM DonDatHang
            WHERE MaTTDH IN ({','.join(['%s']*len(ORDER_STATUS_OK))})
        """, ORDER_STATUS_OK)
        return {int(row[0]) for row in cur.fetchall()}
def fetch_ctsp_to_masp(conn) -> Tuple[Dict[int,int], Dict[int, List[int]]]:
    with conn.cursor() as cur:
        cur.execute("SELECT MaCTSP, MaSP FROM ChiTietSanPham")
        ct2sp = {int(r[0]): int(r[1]) for r in cur.fetchall()}
    sp2ct = defaultdict(list)
    for ct, sp in ct2sp.items():
        sp2ct[sp].append(ct)
    return ct2sp, sp2ct
def fetch_transactions_masp(conn, order_ids: Set[int], ct2sp: Dict[int,int]) -> List[List[int]]:
    if not order_ids: return []
    by_order = defaultdict(set)
    with conn.cursor() as cur:
        ids = list(order_ids)
        CHUNK = 1000
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i+CHUNK]
            ph = ",".join(["%s"]*len(chunk))
            cur.execute(f"""
                SELECT MaDDH, MaCTSP, SoLuong, COALESCE(SoLuongTra,0) AS SLTra
                FROM CT_DonDatHang
                WHERE MaDDH IN ({ph})
            """, chunk)
            for MaDDH, MaCTSP, SoLuong, SLTra in cur.fetchall():
                if SoLuong is None: 
                    continue
                if (SLTra or 0) >= SoLuong:
                    continue  # dòng đã trả hết → bỏ
                masp = ct2sp.get(int(MaCTSP))
                if masp is None:
                    continue
                by_order[int(MaDDH)].add(masp)
    tx = [sorted(list(s)) for _, s in sorted(by_order.items(), key=lambda x: x[0])]
    return tx
class Node:
    __slots__=("item","count","parent","children","link")
    def __init__(self,item,count,parent):
        self.item=item
        self.count=count
        self.parent=parent
        self.children={}
        self.link=None
def build_fp_tree(transactions, min_sup_count, debug=False):
    freq = Counter(chain.from_iterable(transactions))
    freq = {i:c for i,c in freq.items() if c >= min_sup_count}
    header = {i:[freq[i], None] for i in sorted(freq, key=lambda x:(-freq[x], x))}
    root = Node(None,0,None)

    for idx, t in enumerate(transactions, 1):
        ordered = [i for i in sorted(t, key=lambda x:(-freq.get(x,0), x)) if i in freq]
        if not ordered: 
            continue
        cur = root
        for it in ordered:
            if it not in cur.children:
                n = Node(it,0,cur)
                cur.children[it]=n
                if header[it][1] is None: 
                    header[it][1]=n
                else:
                    p=header[it][1]
                    while p.link: 
                        p=p.link
                    p.link = n
            cur = cur.children[it]
            cur.count += 1
    return root, header
def cond_pattern_base(item, header):
    res=[]
    node=header[item][1]
    while node:
        path=ascend(node)
        if path: 
            res.append((path, node.count))
        node=node.link
    return res
def build_conditional_tree(paths, min_sup_count, order_hint=None, debug=False):
    freq = Counter()
    for items, cnt in paths:
        for it in set(items):
            freq[it] += cnt
    freq = {i:c for i,c in freq.items() if c >= min_sup_count}
    order = (sorted(freq, key=lambda x:(-freq[x], x))
             if order_hint is None else [i for i in order_hint if i in freq])
    header = {i:[freq[i], None] for i in order}
    root = Node(None,0,None)
    for items, cnt in paths:
        filt = [i for i in items if i in freq]
        filt.sort(key=lambda x: order.index(x))
        if not filt: continue
        cur = root
        for it in filt:
            if it not in cur.children:
                n = Node(it,0,cur)
                cur.children[it]=n
                if header[it][1] is None: header[it][1]=n
                else:
                    p=header[it][1]
                    while p.link: p=p.link
                    p.link = n
            cur = cur.children[it]
            cur.count += cnt
    return root, header, order
def mine_patterns(header, min_sup_count, debug=False):
    patterns={}
    items = sorted(header, key=lambda x:(header[x][0], x))  # duyệt từ ít phổ biến
    for item in items:
        base = cond_pattern_base(item, header)
        cond_root, cond_header, order = build_conditional_tree(base, min_sup_count, debug=debug)
        sub_patterns = {}
        if cond_header:
            sub_patterns = mine_patterns(cond_header, min_sup_count, debug=debug)
        patterns[frozenset([item])] = header[item][0]
        for P, cnt in sub_patterns.items():
            patterns[frozenset(set(P)|{item})] = cnt
    return patterns

def generate_rules(freq_counts: Dict[frozenset,int], N: int, max_rule_size=MAX_RULE_SIZE):
    sup_rel = {X:c/N for X,c in freq_counts.items()}
    rules_dict = {}
    for X,cnt in freq_counts.items():
        if len(X)<2 or len(X)>max_rule_size:
            continue
        X_list=list(X)
        for r in range(1,len(X)):
            for A in combinations(X_list, r):
                A=frozenset(A)
                B=frozenset(X - set(A))
                conf = sup_rel[X] / max(sup_rel.get(A,1e-12), 1e-12)
                if conf < MIN_CONF:
                    continue
                for b in B:
                    bset=frozenset([b])
                    lift = (sup_rel[X] / max(sup_rel.get(A,1e-12)*sup_rel.get(bset,1e-12), 1e-12))
                    rule_key = (A, b)
                    if rule_key not in rules_dict:
                        rules_dict[rule_key] = {
                            "A": A, "b": b, "X": X,
                            "support": sup_rel[X],
                            "confidence": conf,
                            "lift": lift
                        }
    rules = list(rules_dict.values())
    rules.sort(key=lambda r:(r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules, sup_rel
def init_cache_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS FP_ModelMetadata (...)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS FP_Rules (...)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS FP_FrequentItemsets (...)""")
        conn.commit()

def save_model_to_db(conn, cache_data, min_sup, min_conf):
    ...
    cur.execute("""
        INSERT INTO FP_ModelMetadata (N, min_sup, min_conf, total_rules, total_freq_items)
        VALUES (%s, %s, %s, %s, %s)
    """, (...))
    ...
    cur.executemany("""INSERT INTO FP_Rules (...) VALUES (...)""", chunk)
    ...
    cur.executemany("""INSERT INTO FP_FrequentItemsets (...) VALUES (...)""", chunk)
    conn.commit()
def load_model_from_db(conn, min_sup, min_conf):
    cur.execute("""
        SELECT id, N, total_rules, total_freq_items, created_at
        FROM FP_ModelMetadata
        WHERE min_sup = %s AND min_conf = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (min_sup, min_conf))
    ...

@app.get("/health")
def health():
    return {
        "ok": True,
        "transactions": CACHE["N"],
        "rules": len(CACHE["rules"]),
        "min_sup": MIN_SUP,
        "min_conf": MIN_CONF,
    }


@app.post("/refresh")
def refresh(force: bool = True):
    rebuild_model(debug=True, force_rebuild=force)
    return {"ok": True, "transactions": CACHE["N"], "rules": len(CACHE["rules"])}

from pydantic import BaseModel
from typing import List, Dict, Set
from fastapi import FastAPI
import math
from collections import defaultdict

app = FastAPI()

class RecRequest(BaseModel):
    items: List[int]       # danh sách MaCTSP trong giỏ
    k: int = 8
    exclude_incart: bool = True
    require_instock: bool = False
    group_by_antecedent: bool = True
    per_group_k: int | None = None

def _score_rule(r, N: int) -> float:
    return (r["confidence"] * math.log1p(N * r["support"]) * (r["lift"] ** 0.5))

@app.post("/recommend")
def recommend(req: RecRequest):
    if CACHE["N"] == 0: return {"items": [], "groups": []}
    conn = db_conn()
    try: ct2sp, _ = fetch_ctsp_to_masp(conn)
    finally: conn.close()
    cart_ct = [int(x) for x in req.items]
    cart_sp: Set[int] = set()
    for ct in cart_ct:
        sp = ct2sp.get(ct)
        if sp is not None: cart_sp.add(sp)
    rules_by_A: Dict[frozenset, list] = CACHE.get("rules_by_A", {})
    N = CACHE["N"]
    per_group_k = req.per_group_k or req.k
    subsets = []
    for sz in range(len(cart_sp), 0, -1):
        for A in combinations(sorted(cart_sp), sz): subsets.append(frozenset(A))
    instock = None
    if req.require_instock:
        conn = db_conn()
        try: instock = fetch_instock_set_masp(conn)
        finally: conn.close()
    groups = []
    for A in subsets:
        rules = rules_by_A.get(A, [])
        agg: Dict[int, Dict] = {}
        for r in rules:
            b_sp = r["b"]
            if req.exclude_incart and b_sp in cart_sp: continue
            if instock is not None and b_sp not in instock: continue
            score = _score_rule(r, N)
            cur = agg.get(b_sp)
            if cur is None or score > cur["score"]:
                agg[b_sp] = {
                    "MaSP": b_sp, "score": round(score, 6),
                    "confidence": round(r["confidence"], 6),
                    "support": round(r["support"], 6), "lift": round(r["lift"], 6),
                    "antecedent": sorted(list(r["A"])), "rule_size": len(r["X"]),
                }
        items = sorted(
            agg.values(),
            key=lambda x: (x["score"], x["confidence"], x["support"]),
            reverse=True
        )[:per_group_k]
        if items:
            groups.append({
                "antecedent": sorted(list(A)),
                "items": items
            })
    merged: Dict[int, Dict] = {}
    for g in groups:
        for it in g["items"]:
            b = it["MaSP"]
            cur = merged.get(b)
            if cur is None or it["score"] > cur["score"]: merged[b] = it
    merged_items = sorted(
        merged.values(),
        key=lambda x: (x["score"], x["confidence"], x["support"]),
        reverse=True
    )[:req.k]
    return {"items": merged_items, "groups": groups}
