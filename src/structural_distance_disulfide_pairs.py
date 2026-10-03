from src.structural_distance import build_distance_matrix, classify_pair

dist_matrix, pos_to_idx = build_distance_matrix()

disulfide_pairs = [
    (171, 193),  # A171C/S193C
    (233, 282),  # N233C/S282C
]

print("Checking disulfide-pair geometry (real Cys-Cys bonds typically need Cβ-Cβ ~4-8Å apart):\n")
for pos_a, pos_b in disulfide_pairs:
    relationship, distance = classify_pair(pos_a, pos_b, dist_matrix, pos_to_idx)
    if distance is None:
        print(f"  {pos_a}-{pos_b}: position not found in resolved structure")
    else:
        feasible = "PLAUSIBLE" if distance <= 10.0 else "UNLIKELY -- too far apart for a real disulfide bond"
        print(f"  {pos_a}-{pos_b}: {distance:.2f} Å apart ({relationship}) -- {feasible}")