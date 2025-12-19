import re
from datetime import datetime

def extract_insert_data(sql_content, table_name):
    """Extract INSERT statements for a specific table"""
    pattern = rf"INSERT INTO `{table_name}`.*?VALUES\s*(.*?);"
    matches = re.findall(pattern, sql_content, re.DOTALL | re.IGNORECASE)
    return matches

def parse_insert_values(values_str):
    """Parse VALUES clause into list of tuples"""
    rows = []
    # Split by ),( to get individual rows
    row_pattern = r'\(([^)]+)\)'
    matches = re.findall(row_pattern, values_str)
    return matches

def get_max_id(insert_data, id_position=0):
    """Get maximum ID from insert data"""
    if not insert_data:
        return 0
    
    max_id = 0
    for row in insert_data:
        values = parse_insert_values(row)
        for val_str in values:
            parts = [v.strip() for v in val_str.split(',')]
            try:
                current_id = int(parts[id_position])
                max_id = max(max_id, current_id)
            except (ValueError, IndexError):
                continue
    return max_id

def adjust_insert_values(values_str, id_mappings):
    """Adjust IDs in INSERT values based on mappings"""
    rows = parse_insert_values(values_str)
    adjusted_rows = []
    
    for row in rows:
        parts = [v.strip() for v in row.split(',')]
        
        # Apply ID mappings
        for table, (offset, position) in id_mappings.items():
            if position < len(parts):
                try:
                    old_id = int(parts[position])
                    parts[position] = str(old_id + offset)
                except ValueError:
                    pass
        
        adjusted_rows.append('(' + ','.join(parts) + ')')
    
    return ',\n'.join(adjusted_rows)

# Read SQL files
with open('thienduong_3tshop_tttn(1).sql', 'r', encoding='utf-8') as f:
    sql_tttn = f.read()

with open('thienduong_3tshop_test(4).sql', 'r', encoding='utf-8') as f:
    sql_test = f.read()

# Tables to merge (with ID tracking)
merge_tables = {
    'DonDatHang': ('MaDDH', 0),
    'CT_DonDatHang': ('MaCTDDH', 0),
    'HoaDon': ('SoHD', 0),
    'PhieuTraHang': ('MaPhieuTra', 0),
    'PhieuChi': ('MaPhieuChi', 0),
    'FP_ModelMetadata': ('id', 0),
    'FP_FrequentItemsets': ('id', 0),
    'FP_Rules': ('id', 0)
}

# Calculate offsets
offsets = {}
for table, (id_col, id_pos) in merge_tables.items():
    tttn_data = extract_insert_data(sql_tttn, table)
    max_id = get_max_id(tttn_data, id_pos)
    offsets[table] = max_id
    print(f"{table}: Max ID in TTTN = {max_id}")

# Start building merged SQL
merged_sql = sql_test

# Replace data for merge tables
for table, (id_col, id_pos) in merge_tables.items():
    # Get data from both files
    tttn_inserts = extract_insert_data(sql_tttn, table)
    test_inserts = extract_insert_data(sql_test, table)
    
    if not tttn_inserts and not test_inserts:
        continue
    
    # Build ID mappings for this table
    id_mappings = {
        'DonDatHang': (offsets['DonDatHang'], 1),  # MaDDH position
        'CT_DonDatHang': (offsets['CT_DonDatHang'], 0),  # MaCTDDH position
        'PhieuTraHang': (offsets['PhieuTraHang'], 0),  # MaPhieuTra position
        'PhieuChi': (offsets['PhieuChi'], 0),  # MaPhieuChi position
        'FP_ModelMetadata': (offsets['FP_ModelMetadata'], 0),  # id position
        'FP_FrequentItemsets': (offsets['FP_FrequentItemsets'], 0),  # id position
        'FP_Rules': (offsets['FP_Rules'], 0)  # id position
    }
    
    # Adjust test data
    adjusted_test_inserts = []
    for insert in test_inserts:
        if table == 'DonDatHang':
            adjusted = adjust_insert_values(insert, {
                'DonDatHang': (offsets['DonDatHang'], 0)
            })
        elif table == 'CT_DonDatHang':
            adjusted = adjust_insert_values(insert, {
                'CT_DonDatHang': (offsets['CT_DonDatHang'], 0),
                'DonDatHang': (offsets['DonDatHang'], 1),
                'PhieuTraHang': (offsets['PhieuTraHang'], 5)
            })
        elif table == 'HoaDon':
            # HoaDon needs special handling for composite key
            adjusted = adjust_insert_values(insert, {
                'DonDatHang': (offsets['DonDatHang'], 1)
            })
        elif table == 'PhieuTraHang':
            adjusted = adjust_insert_values(insert, {
                'PhieuTraHang': (offsets['PhieuTraHang'], 0)
            })
        elif table == 'PhieuChi':
            adjusted = adjust_insert_values(insert, {
                'PhieuChi': (offsets['PhieuChi'], 0),
                'PhieuTraHang': (offsets['PhieuTraHang'], 3)
            })
        elif table in ['FP_ModelMetadata', 'FP_FrequentItemsets', 'FP_Rules']:
            adjusted = adjust_insert_values(insert, {
                table: (offsets[table], 0),
                'FP_ModelMetadata': (offsets['FP_ModelMetadata'], 1)
            })
        else:
            adjusted = insert
        
        adjusted_test_inserts.append(adjusted)
    
    # Combine TTTN + adjusted TEST data
    combined_values = []
    for insert in tttn_inserts:
        combined_values.extend(parse_insert_values(insert))
    for insert in adjusted_test_inserts:
        rows = insert.split('),\n')
        combined_values.extend([r.strip('()') for r in rows if r.strip()])
    
    # Find and replace INSERT statement for this table
    pattern = rf"(INSERT INTO `{table}`[^;]+VALUES\s*).*?;"
    
    if combined_values:
        new_insert = f"INSERT INTO `{table}` VALUES\n"
        formatted_values = []
        for i, val in enumerate(combined_values):
            if not val.startswith('('):
                val = '(' + val
            if not val.endswith(')'):
                val = val + ')'
            formatted_values.append(val)
        
        new_insert += ',\n'.join(formatted_values) + ';'
        
        merged_sql = re.sub(pattern, new_insert, merged_sql, flags=re.DOTALL)

# Write merged SQL
with open('thienduong_3tshop_test_merged.sql', 'w', encoding='utf-8') as f:
    f.write(merged_sql)

print("\nMerge completed! Output: thienduong_3tshop_test_merged.sql")
print(f"\nOffsets applied:")
for table, offset in offsets.items():
    print(f"  {table}: +{offset}")