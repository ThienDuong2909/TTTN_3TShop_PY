# fp_rec_api.py
# pip install fastapi uvicorn pymysql pydantic
import os, math
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from itertools import combinations, chain

import pymysql
from fastapi import FastAPI
from pydantic import BaseModel

# ------------ Config ------------
DB_HOST = os.getenv("DB_HOST", "45.252.248.106")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "thienduong_3tshop")
DB_PASS = os.getenv("DB_PASS", "SmP!8l4HpSe(%7l!")
DB_NAME = os.getenv("DB_NAME", "thienduong_3tshop_test")

REQUIRE_INVOICE = False       # True nếu muốn bắt buộc có hóa đơn
ORDER_STATUS_OK = (4, 7)      # 4=hoàn tất; (thêm 7 nếu bạn có trạng thái giao thành công)
MIN_SUP = float(os.getenv("MIN_SUP", "0.4"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.8"))  # hơi hạ để có nhiều gợi ý
MAX_RULE_SIZE = 4             # tối đa kích cỡ itemset để sinh luật

# ------------ DB utilities ------------
def db_conn():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, charset="utf8mb4"
    )

def fetch_completed_order_ids(conn) -> Set[int]:
    with conn.cursor() as cur:
        if REQUIRE_INVOICE:
            cur.execute(f"""
                SELECT ddh.MaDDH
                FROM DonDatHang ddh
                JOIN HoaDon hd ON hd.MaDDH = ddh.MaDDH
                WHERE ddh.MaTTDH IN ({','.join(['%s']*len(ORDER_STATUS_OK))})
            """, ORDER_STATUS_OK)
        else:
            cur.execute(f"""
                SELECT MaDDH
                FROM DonDatHang
                WHERE MaTTDH IN ({','.join(['%s']*len(ORDER_STATUS_OK))})
            """, ORDER_STATUS_OK)
        return {row[0] for row in cur.fetchall()}

def fetch_transactions(conn, order_ids: Set[int]) -> List[List[int]]:
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
                if SoLuong is None: continue
                if (SLTra or 0) < SoLuong:
                    by_order[MaDDH].add(int(MaCTSP))
    return [sorted(list(s)) for _, s in sorted(by_order.items(), key=lambda x: x[0])]

# ------------ FP-Tree core ------------
class Node:
    __slots__=("item","count","parent","children","link")
    def __init__(self,item,count,parent):
        self.item=item; self.count=count; self.parent=parent
        self.children={}; self.link=None

def build_fp_tree(transactions, min_sup_count):
    freq = Counter(chain.from_iterable(transactions))
    freq = {i:c for i,c in freq.items() if c >= min_sup_count}
    header = {i:[freq[i], None] for i in sorted(freq, key=lambda x:(-freq[x], x))}
    root = Node(None,0,None)
    for t in transactions:
        ordered = [i for i in sorted(t, key=lambda x:(-freq.get(x,0), x)) if i in freq]
        cur = root
        for it in ordered:
            if it not in cur.children:
                n = Node(it,0,cur); cur.children[it]=n
                if header[it][1] is None: header[it][1]=n
                else:
                    p=header[it][1]
                    while p.link: p=p.link
                    p.link = n
            cur = cur.children[it]
            cur.count += 1
    return root, header

def ascend(node):
    path=[]
    while node and node.parent and node.parent.item is not None:
        node=node.parent; path.append(node.item)
    return list(reversed(path))

def cond_pattern_base(item, header):
    res=[]; node=header[item][1]
    while node:
        path=ascend(node)
        if path: res.append((path, node.count))
        node=node.link
    return res

def build_conditional_tree(paths, min_sup_count, order_hint=None):
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
                n = Node(it,0,cur); cur.children[it]=n
                if header[it][1] is None: header[it][1]=n
                else:
                    p=header[it][1]
                    while p.link: p=p.link
                    p.link = n
            cur = cur.children[it]
            cur.count += cnt
    return root, header, order

def mine_patterns(header, min_sup_count):
    patterns={}
    items = sorted(header, key=lambda x:(header[x][0], x))  # ascending support
    for item in items:
        base = cond_pattern_base(item, header)
        cond_root, cond_header, order = build_conditional_tree(base, min_sup_count)
        sub_patterns = {}
        if cond_header:
            sub_patterns = mine_patterns(cond_header, min_sup_count)
        patterns[frozenset([item])] = header[item][0]
        for P, cnt in sub_patterns.items():
            patterns[frozenset(set(P)|{item})] = cnt
    return patterns

# ------------ Rules & recommender ------------
def generate_rules(freq_counts: Dict[frozenset,int], N: int, max_rule_size=MAX_RULE_SIZE):
    # relative supports
    sup_rel = {X:c/N for X,c in freq_counts.items()}
    rules=[]  # A -> B
    for X,cnt in freq_counts.items():
        if len(X)<2 or len(X)>max_rule_size: 
            continue
        X_list=list(X)
        for r in range(1,len(X)):
            for A in combinations(X_list, r):
                A=frozenset(A); B=frozenset(X - set(A))
                conf = sup_rel[X] / max(sup_rel.get(A,1e-12), 1e-12)
                if conf < MIN_CONF: 
                    continue
                # explode consequent into singletons for ranking clarity
                for b in B:
                    bset=frozenset([b])
                    # lift = P(X)/[P(A)*P({b})] but more standard is P(Ab)/[P(A)P(b)]
                    lift = (sup_rel[X] / max(sup_rel.get(A,1e-12)*sup_rel.get(bset,1e-12), 1e-12))
                    rules.append({
                        "A": A, "b": b, "X": X,
                        "support": sup_rel[X],
                        "confidence": conf,
                        "lift": lift
                    })
    # sort: by confidence desc, then lift desc, then support desc
    rules.sort(key=lambda r:(r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules, sup_rel

# in-memory cache
CACHE = {
    "N": 0,
    "freq": {},         # {frozenset: count}
    "rules": [],        # list of dicts (A->b)
    "sup_rel": {},      # relative supports for lift
    "rules_by_A": {},
}

def rebuild_model():
    conn = db_conn()
    try:
        orders = fetch_completed_order_ids(conn)
        tx = fetch_transactions(conn, orders)
    finally:
        conn.close()
    if not tx:
        CACHE.update({"N":0,"freq":{},"rules":[],"sup_rel":{}})
        return
    min_sup_count = max(1, math.ceil(MIN_SUP * len(tx)))
    # FP-tree
    _, header = build_fp_tree(tx, min_sup_count)
    pats = mine_patterns(header, min_sup_count)
    # exact count by scan (an toàn)
    abs_cnt = Counter()
    for X in pats:
        cnt=0
        for t in tx:
            if set(X).issubset(t): cnt+=1
        abs_cnt[X]=cnt
    freq = {X:c for X,c in abs_cnt.items() if c>=min_sup_count}
    rules, sup_rel = generate_rules(freq, len(tx))
    rules_by_A = defaultdict(list)
    for r in rules:
        rules_by_A[r["A"]].append(r)

    CACHE.update({"N":len(tx), "freq":freq, "rules":rules, "sup_rel":sup_rel, "rules_by_A":rules_by_A})

# ------------ API ------------
app = FastAPI(title="FP-Growth Recommender")

class RecRequest(BaseModel):
    items: List[int]       # các MaCTSP đang có trong giỏ
    k: int = 8             # số gợi ý tối đa
    exclude_incart: bool = True
    require_instock: bool = False  # nếu muốn lọc tồn kho
    group_by_antecedent: bool = True      # NEW: trả về theo nhóm
    per_group_k: int | None = None

@app.on_event("startup")
def _startup():
    rebuild_model()

@app.get("/health")
def health():
    return {"ok": True, "transactions": CACHE["N"], "rules": len(CACHE["rules"])}

@app.post("/refresh")
def refresh():
    rebuild_model()
    return {"ok": True, "transactions": CACHE["N"], "rules": len(CACHE["rules"])}

def fetch_instock_set(conn) -> Set[int]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MaCTSP FROM ChiTietSanPham WHERE SoLuongTon > 0")
            return {row[0] for row in cur.fetchall()}
    except Exception:
        return set()  # nếu không có bảng/field, bỏ qua lọc tồn


# def recommend(req: RecRequest):
#     cart = set(int(x) for x in req.items)
#     if CACHE["N"] == 0:
#         return {"items": []}

#     # lọc rule có tiền đề A ⊆ cart và hậu quả b ∉ cart
#     cands = []
#     for r in CACHE["rules"]:
#         A = r["A"]; b = r["b"]
#         if not A.issubset(cart): 
#             continue
#         if req.exclude_incart and b in cart: 
#             continue
#         cands.append(r)
#         print(f"Rule matched: {set(A)} -> {b}")
    
#     print(f"Found {len(cands)} candidate rules for cart {cart}")

#     # optional: lọc tồn kho
#     if req.require_instock:
#         conn = db_conn()
#         try:
#             instock = fetch_instock_set(conn)
#         finally:
#             conn.close()
#         cands = [r for r in cands if r["b"] in instock]

#     # gộp theo b, chọn score cao nhất (conf * log(1+N*support) * lift^0.5)
#     N = CACHE["N"]
#     aggregated: Dict[int, Dict] = {}
#     for r in cands:
#         score = (r["confidence"] * math.log1p(N * r["support"]) * (r["lift"] ** 0.5))
#         cur = aggregated.get(r["b"])
#         if (cur is None) or (score > cur["score"]):
#             aggregated[r["b"]] = {
#                 "MaCTSP": r["b"],
#                 "score": round(score, 6),
#                 "confidence": round(r["confidence"], 6),
#                 "support": round(r["support"], 6),
#                 "lift": round(r["lift"], 6),
#                 "antecedent": sorted(list(r["A"])),
#                 "rule_size": len(r["X"]),
#             }
#             print("aggregating:", aggregated[r["b"]])
#             # print(f"Aggregated recommendation: {r['b']} with score {score:.6f}")
#     # print(f"Aggregated to {len(aggregated)} unique recommendations")

#     # sort & top-k
#     out = sorted(aggregated.values(), key=lambda x:(x["score"], x["confidence"], x["support"]), reverse=True)
#     print(f"Final recommendations: {[item['MaCTSP'] for item in out[:req.k]]}")
#     return {"items": out[:req.k]}
@app.post("/recommend")
def recommend(req: RecRequest):
    cart = set(int(x) for x in req.items)
    if CACHE["N"] == 0:
        return {"items": [], "groups": []}

    rules_by_A: Dict[frozenset, list] = CACHE.get("rules_by_A", {})
    N = CACHE["N"]
    per_group_k = req.per_group_k or req.k

    # 1) Xây danh sách các tiền đề EXACT cần xét: toàn bộ tập con không rỗng của giỏ
    #    (để tạo từng nhóm riêng: {2,3}, {2}, {3}…)
    subsets = []
    for sz in range(len(cart), 0, -1):   # to {2,3} rồi mới {2} {3}
        from itertools import combinations
        for A in combinations(sorted(cart), sz):
            subsets.append(frozenset(A))

    # 2) Lấy rules theo từng nhóm A (exact match), lọc instock, lọc trùng trong giỏ
    groups = []
    instock = None
    if req.require_instock:
        conn = db_conn()
        try:
            instock = fetch_instock_set(conn)
        finally:
            conn.close()

    def score_of(r):
        # cùng công thức như trước
        return (r["confidence"] * math.log1p(N * r["support"]) * (r["lift"] ** 0.5))

    for A in subsets:
        rules = rules_by_A.get(A, [])
        # filter hậu quả
        cand = []
        for r in rules:
            b = r["b"]
            if req.exclude_incart and b in cart:
                continue
            if instock is not None and b not in instock:
                continue
            cand.append(r)

        # gộp theo b trong NHÓM A (chọn rule có điểm cao nhất cho mỗi b)
        agg: Dict[int, Dict] = {}
        for r in cand:
            s = score_of(r)
            cur = agg.get(r["b"])
            if cur is None or s > cur["score"]:
                agg[r["b"]] = {
                    "MaCTSP": r["b"],
                    "score": round(s, 6),
                    "confidence": round(r["confidence"], 6),
                    "support": round(r["support"], 6),
                    "lift": round(r["lift"], 6),
                    "antecedent": sorted(list(r["A"])),
                    "rule_size": len(r["X"]),
                }

        # sort & top-k cho nhóm này
        items = sorted(agg.values(), key=lambda x: (x["score"], x["confidence"], x["support"]), reverse=True)[:per_group_k]
        if items:
            groups.append({
                "antecedent": sorted(list(A)),
                "items": items
            })
        print(f"Group for antecedent {set(A)} has {len(items)} items.")
        print(f"  Items: {[item['MaCTSP'] for item in items]}")

    # 3) Tạo danh sách TỔNG HỢP giống hành vi cũ (giữ backward-compat)
    #    = hợp các nhóm, giữ best score per b, rồi top-k
    merged: Dict[int, Dict] = {}
    for g in groups:
        for it in g["items"]:
            b = it["MaCTSP"]
            cur = merged.get(b)
            if cur is None or it["score"] > cur["score"]:
                merged[b] = it
    merged_items = sorted(merged.values(), key=lambda x: (x["score"], x["confidence"], x["support"]), reverse=True)[:req.k]
    print("grouped recommendation results merged:")
    #print(groups)
    print('Final merged recommendations:', [item['MaCTSP'] for item in merged_items])
    

    return {
        "items": merged_items,  # tổng hợp (cũ)
        "groups": groups        # NHÓM THEO TIỀN ĐỀ (mới)
    }

