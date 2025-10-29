# fp_rec_api.py
# pip install fastapi uvicorn pymysql pydantic
import os, math
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from itertools import combinations, chain

import pymysql
from fastapi import FastAPI
from pydantic import BaseModel

# ======================= Config =======================
DB_HOST = os.getenv("DB_HOST", "45.252.248.106")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "thienduong_3tshop")
DB_PASS = os.getenv("DB_PASS", "SmP!8l4HpSe(%7l!")
DB_NAME = os.getenv("DB_NAME", "thienduong_3tshop_test")

REQUIRE_INVOICE = False       # True: chỉ lấy đơn có hóa đơn
ORDER_STATUS_OK = (4, 7)      # 4=đã hoàn tất; (7 nếu bạn có trạng thái giao thành công)
MIN_SUP = float(os.getenv("MIN_SUP", "0.4"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.8"))
MAX_RULE_SIZE = 4             # kích thước itemset tối đa để sinh luật

# ======================= DB utils ======================
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
        return {int(row[0]) for row in cur.fetchall()}

def fetch_ctsp_to_masp(conn) -> Tuple[Dict[int,int], Dict[int, List[int]]]:
    """Trả về 2 map: (ct2sp, sp2ctlist) và in DEBUG mapping."""
    with conn.cursor() as cur:
        cur.execute("SELECT MaCTSP, MaSP FROM ChiTietSanPham")
        ct2sp = {int(r[0]): int(r[1]) for r in cur.fetchall()}

    sp2ct = defaultdict(list)
    for ct, sp in ct2sp.items():
        sp2ct[sp].append(ct)

    # DEBUG mapping
    print("\n" + "="*90)
    print("DEBUG — ÁNH XẠ MaCTSP → MaSP")
    print(f"- Tổng MaCTSP: {len(ct2sp)}")
    print(f"- Tổng MaSP  : {len(sp2ct)}")
    for sp in sorted(sp2ct):
        print(f"  MaSP {sp}: {sorted(sp2ct[sp])}")
    return ct2sp, sp2ct

def fetch_transactions_masp(conn, order_ids: Set[int], ct2sp: Dict[int,int]) -> List[List[int]]:
    """
    Lấy giao dịch theo cấp MaSP:
      - từ CT_DonDatHang (MaCTSP)
      - loại các dòng trả hết (SoLuongTra >= SoLuong)
      - map sang MaSP, loại trùng biến thể trong cùng 1 đơn
    """
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
                    continue  # trả hết → bỏ
                masp = ct2sp.get(int(MaCTSP))
                if masp is None:
                    continue
                by_order[int(MaDDH)].add(masp)
    # Chuẩn hóa theo MaSP tăng dần
    tx = [sorted(list(s)) for _, s in sorted(by_order.items(), key=lambda x: x[0])]
    # DEBUG transactions
    print("\n" + "="*90)
    print("DEBUG — GIAO DỊCH (đã quy về MaSP, loại trùng biến thể)")
    for i,t in enumerate(tx,1):
        print(f"  T{i:03d}:", t)
    return tx

def fetch_instock_set_masp(conn) -> Set[int]:
    """
    Lọc tồn kho theo MaSP. Ở đây suy theo tổng SoLuongTon của các MaCTSP con.
    Nếu dự án bạn có bảng tồn kho riêng cho MaSP, sửa lại query này cho đúng.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ctp.MaSP
                FROM ChiTietSanPham ctp
                GROUP BY ctp.MaSP
                HAVING SUM(COALESCE(ctp.SoLuongTon,0)) > 0
            """)
            return {int(row[0]) for row in cur.fetchall()}
    except Exception:
        return set()

# ======================= FP-Tree core =======================
class Node:
    __slots__=("item","count","parent","children","link")
    def __init__(self,item,count,parent):
        self.item=item; self.count=count; self.parent=parent
        self.children={}; self.link=None

def build_fp_tree(transactions, min_sup_count, debug=False):
    freq = Counter(chain.from_iterable(transactions))
    freq = {i:c for i,c in freq.items() if c >= min_sup_count}
    header = {i:[freq[i], None] for i in sorted(freq, key=lambda x:(-freq[x], x))}
    root = Node(None,0,None)
    if debug:
        print("\n" + "="*90)
        print("DEBUG — BUILD FP-TREE (theo MaSP)")
        print(f"- min_sup_count = {min_sup_count}")
        print("- Header init (MaSP → support count):")
        for i in sorted(header, key=lambda x:(-header[x][0], x)):
            print(f"  {i} → {header[i][0]}")
    for idx, t in enumerate(transactions, 1):
        ordered = [i for i in sorted(t, key=lambda x:(-freq.get(x,0), x)) if i in freq]
        if debug:
            print(f"  Insert T{idx:04d}:", ordered)
        if not ordered: 
            continue
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
    if debug:
        print("  Conditional Header:", {i:freq[i] for i in order})
    for items, cnt in paths:
        filt = [i for i in items if i in freq]
        filt.sort(key=lambda x: order.index(x))
        if not filt: continue
        if debug:
            print("  Insert prefix:", filt, "x", cnt)
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

def mine_patterns(header, min_sup_count, debug=False):
    patterns={}
    items = sorted(header, key=lambda x:(header[x][0], x))  # ascending support
    for item in items:
        if debug:
            print(f"\n# Item {item}: conditional pattern base")
        base = cond_pattern_base(item, header)
        if debug:
            for path,cnt in base:
                print("   ", path, ":", cnt)
        cond_root, cond_header, order = build_conditional_tree(base, min_sup_count, debug=debug)
        sub_patterns = {}
        if cond_header:
            sub_patterns = mine_patterns(cond_header, min_sup_count, debug=debug)
        patterns[frozenset([item])] = header[item][0]
        for P, cnt in sub_patterns.items():
            patterns[frozenset(set(P)|{item})] = cnt
    return patterns

# ======================= Rules & cache =======================
def generate_rules(freq_counts: Dict[frozenset,int], N: int, max_rule_size=MAX_RULE_SIZE):
    sup_rel = {X:c/N for X,c in freq_counts.items()}
    rules=[]
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
                for b in B:
                    bset=frozenset([b])
                    lift = (sup_rel[X] / max(sup_rel.get(A,1e-12)*sup_rel.get(bset,1e-12), 1e-12))
                    rules.append({
                        "A": A, "b": b, "X": X,
                        "support": sup_rel[X],
                        "confidence": conf,
                        "lift": lift
                    })
    rules.sort(key=lambda r:(r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules, sup_rel

CACHE = {
    "N": 0,
    "freq": {},         # {frozenset(MaSP): abs_count}
    "rules": [],        # list of dicts (A->b) theo MaSP
    "sup_rel": {},
    "rules_by_A": {},   # {frozenset(MaSP): [rules...] }
}

def rebuild_model(debug=True):
    conn = db_conn()
    try:
        orders = fetch_completed_order_ids(conn)
        if debug:
            print("\n" + "="*90)
            print(f"REBUILD MODEL — số đơn hoàn tất: {len(orders)}")
        ct2sp, sp2ct = fetch_ctsp_to_masp(conn)
        tx = fetch_transactions_masp(conn, orders, ct2sp)
    finally:
        conn.close()

    if not tx:
        CACHE.update({"N":0,"freq":{},"rules":[],"sup_rel":{},"rules_by_A":{}})
        if debug: print("No transactions → cache cleared.")
        return

    min_sup_count = max(1, math.ceil(MIN_SUP * len(tx)))
    if debug:
        print(f"- MIN_SUP={MIN_SUP:.2f} → min_sup_count={min_sup_count} trên {len(tx)} giao dịch")

    # FP-tree & patterns
    _, header = build_fp_tree(tx, min_sup_count, debug=False)
    pats = mine_patterns(header, min_sup_count, debug=False)

    # exact count by re-scan for clarity
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

    CACHE.update({
        "N": len(tx),
        "freq": freq,
        "rules": rules,
        "sup_rel": sup_rel,
        "rules_by_A": dict(rules_by_A),
    })

    if debug:
        print(f"- Frequent itemsets: {len(freq)} | Rules (pass MIN_CONF={MIN_CONF:.2f}): {len(rules)}")
        # một vài frequent set tiêu biểu
        top = sorted(freq.items(), key=lambda kv:(-kv[1], -len(kv[0]), sorted(kv[0])))[:10]
        print("  TOP frequent (MaSP):")
        for X,c in top:
            print(f"   • {sorted(list(X))}  sup={c}/{len(tx)}={c/len(tx):.1%}")

# ======================= API =======================
app = FastAPI(title="FP-Growth Recommender (mine by MaSP)")

class RecRequest(BaseModel):
    items: List[int]       # INPUT: MaCTSP trong giỏ
    k: int = 8
    exclude_incart: bool = True
    require_instock: bool = False
    group_by_antecedent: bool = True
    per_group_k: int | None = None

@app.on_event("startup")
def _startup():
    rebuild_model(debug=True)

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
def refresh():
    rebuild_model(debug=True)
    return {"ok": True, "transactions": CACHE["N"], "rules": len(CACHE["rules"])}

def _score_rule(r, N: int) -> float:
    # score = conf * log(1 + N*support) * sqrt(lift)
    return (r["confidence"] * math.log1p(N * r["support"]) * (r["lift"] ** 0.5))

@app.post("/recommend")
def recommend(req: RecRequest):
    """
    Nhận items = MaCTSP[], nhưng tra cứu & trả về theo MaSP
    - groups: gom theo từng antecedent (MaSP) exact match: {full cart} rồi tới các tập con
    - items : danh sách tổng hợp (giữ tương thích), theo điểm số
    """
    if CACHE["N"] == 0:
        return {"items": [], "groups": []}

    # Map input MaCTSP -> MaSP
    conn = db_conn()
    try:
        ct2sp, _ = fetch_ctsp_to_masp(conn)
    finally:
        conn.close()

    cart_ct = [int(x) for x in req.items]
    cart_sp = set()
    for ct in cart_ct:
        sp = ct2sp.get(ct)
        if sp is not None:
            cart_sp.add(sp)

    print("\n" + "="*90)
    print("RECOMMEND DEBUG")
    print(f"- Input cart (MaCTSP): {cart_ct}")
    print(f"- Mapped cart (MaSP) : {sorted(list(cart_sp))}")

    rules_by_A: Dict[frozenset, list] = CACHE.get("rules_by_A", {})
    N = CACHE["N"]
    per_group_k = req.per_group_k or req.k

    # Tạo các tập con KHÔNG RỖNG của giỏ (theo MaSP) để nhóm riêng
    subsets = []
    for sz in range(len(cart_sp), 0, -1):
        for A in combinations(sorted(cart_sp), sz):
            subsets.append(frozenset(A))
    print(f"- Antecedent subsets (MaSP): {['{' + ','.join(map(str,sorted(a))) + '}' for a in subsets]}")

    instock = None
    if req.require_instock:
        conn = db_conn()
        try:
            instock = fetch_instock_set_masp(conn)
        finally:
            conn.close()
        print(f"- Instock MaSP count: {len(instock)}")

    groups = []
    total_candidates = 0

    for A in subsets:
        rules = rules_by_A.get(A, [])
        cand = []
        for r in rules:
            b_sp = r["b"]  # đã là MaSP vì training theo MaSP
            if req.exclude_incart and b_sp in cart_sp:
                continue
            if instock is not None and b_sp not in instock:
                continue
            cand.append(r)

        total_candidates += len(cand)

        # Gộp theo b_sp trong nhóm A, lấy rule có score cao nhất
        agg: Dict[int, Dict] = {}
        for r in cand:
            s = _score_rule(r, N)
            cur = agg.get(r["b"])
            if cur is None or s > cur["score"]:
                agg[r["b"]] = {
                    "MaSP": r["b"],
                    "score": round(s, 6),
                    "confidence": round(r["confidence"], 6),
                    "support": round(r["support"], 6),
                    "lift": round(r["lift"], 6),
                    "antecedent": sorted(list(r["A"])),  # theo MaSP
                    "rule_size": len(r["X"]),
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

        print(f"  • Group antecedent {sorted(list(A))} → {len(items)} items")
        if items:
            print("    Items:", [it["MaSP"] for it in items])

    # Hợp các nhóm thành danh sách tổng hợp (giữ tương thích)
    merged: Dict[int, Dict] = {}
    for g in groups:
        for it in g["items"]:
            b = it["MaSP"]
            cur = merged.get(b)
            if cur is None or it["score"] > cur["score"]:
                merged[b] = it

    merged_items = sorted(
        merged.values(),
        key=lambda x: (x["score"], x["confidence"], x["support"]),
        reverse=True
    )[:req.k]

    print(f"- Candidate rules considered: {total_candidates}")
    print(f"- Final merged top-{req.k} (MaSP): {[it['MaSP'] for it in merged_items]}")

    return {
        "items": merged_items,  # danh sách tổng hợp (MaSP)
        "groups": groups        # nhóm theo antecedent (MaSP)
    }
