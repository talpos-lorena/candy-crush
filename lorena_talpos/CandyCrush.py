import argparse
import csv
import os
import random
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any, Set

# ---------- constante (scoruri) ----------
PTS_LINE_3 = 5
PTS_LINE_4 = 10
PTS_LINE_5 = 50
PTS_L = 20
PTS_T = 30

# ---------- patternuri relative (L și T) ----------
L_PATTERNS = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    [(-2, 0), (-1, 0), (0, 0), (0, 1), (0, 2)],
    [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0)],
    [(0, -2), (0, -1), (0, 0), (1, 0), (2, 0)],
]

T_PATTERNS = [
    [(-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0)],
    [(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (2, 0)],
]


# ---------- utilitare pentru tabla de joc ----------
def inside(board: List[List[int]], r: int, c: int) -> bool:
    return 0 <= r < len(board) and 0 <= c < len(board[0])


def make_random_board(rows: int, cols: int) -> List[List[int]]:
    return [[random.randint(1, 4) for _ in range(cols)] for _ in range(rows)]


def deep_copy_board(board: List[List[int]]) -> List[List[int]]:
    return deepcopy(board)


# ---------- detectare linii (orizontal + vertical) ----------
def find_lines(board: List[List[int]]) -> List[Dict[str, Any]]:
    rows, cols = len(board), len(board[0])
    found = []

    # orizontal
    for r in range(rows):
        c = 0
        while c < cols:
            val = board[r][c]
            if val == 0:
                c += 1
                continue
            start_c = c
            # extindem spre dreapta
            while c + 1 < cols and board[r][c + 1] == val:
                c += 1
            length = c - start_c + 1
            if length >= 3:
                eff_len = 5 if length >= 5 else length
                if eff_len == 3:
                    sc = PTS_LINE_3
                elif eff_len == 4:
                    sc = PTS_LINE_4
                else:
                    sc = PTS_LINE_5
                cells = [(r, start_c + i) for i in range(eff_len)]
                found.append({"type": "LINE", "score": sc, "cells": cells})
            c += 1

    # vertical
    for c in range(cols):
        r = 0
        while r < rows:
            val = board[r][c]
            if val == 0:
                r += 1
                continue
            start_r = r
            while r + 1 < rows and board[r + 1][c] == val:
                r += 1
            length = r - start_r + 1
            if length >= 3:
                eff_len = 5 if length >= 5 else length
                if eff_len == 3:
                    sc = PTS_LINE_3
                elif eff_len == 4:
                    sc = PTS_LINE_4
                else:
                    sc = PTS_LINE_5
                cells = [(start_r + i, c) for i in range(eff_len)]
                found.append({"type": "LINE", "score": sc, "cells": cells})
            r += 1

    return found


# ---------- detectare L și T ----------
def find_L_and_T(board: List[List[int]]) -> List[Dict[str, Any]]:
    rows, cols = len(board), len(board[0])
    results = []

    for r in range(rows):
        for c in range(cols):
            center_val = board[r][c]
            if center_val == 0:
                continue

            # calculăm lungimea liniei orizontale ce trece prin centru
            horiz = 1
            cc = c - 1
            while cc >= 0 and board[r][cc] == center_val:
                horiz += 1
                cc -= 1
            cc = c + 1
            while cc < cols and board[r][cc] == center_val:
                horiz += 1
                cc += 1

            # lungimea liniei verticale
            vert = 1
            rr = r - 1
            while rr >= 0 and board[rr][c] == center_val:
                vert += 1
                rr -= 1
            rr = r + 1
            while rr < rows and board[rr][c] == center_val:
                vert += 1
                rr += 1

            # dacă nu avem minim 3 pe orizontală sau verticală, skip
            if horiz < 3 or vert < 3:
                continue

            # verifică toate pattern-urile L
            for pattern in L_PATTERNS:
                cells = []
                ok = True
                for dr, dc in pattern:
                    rr, cc = r + dr, c + dc
                    if not inside(board, rr, cc) or board[rr][cc] != center_val:
                        ok = False
                        break
                    cells.append((rr, cc))
                if ok:
                    results.append({"type": "L", "score": PTS_L, "cells": cells})

            # verifică pattern-urile T
            for pattern in T_PATTERNS:
                cells = []
                ok = True
                for dr, dc in pattern:
                    rr, cc = r + dr, c + dc
                    if not inside(board, rr, cc) or board[rr][cc] != center_val:
                        ok = False
                        break
                    cells.append((rr, cc))
                if ok:
                    results.append({"type": "T", "score": PTS_T, "cells": cells})

    return results


# ---------- eliminare formatiuni (fără suprapuneri) ----------
def remove_formations(board: List[List[int]], formations: List[Dict[str, Any]]) -> int:
    """
    Sortează formațiunile descrescător după scor.
    Aplică regula anti-dublare: dacă o celulă e deja folosită, ignoră formațiunea.
    Returnează suma punctelor obținute și marchează celulele ca 0.
    """
    formations_sorted = sorted(formations, key=lambda x: x["score"], reverse=True)
    used: Set[Tuple[int, int]] = set()
    to_zero: Set[Tuple[int, int]] = set()
    total = 0

    for f in formations_sorted:
        cells = f["cells"]
        if any(cell in used for cell in cells):
            continue
        total += f["score"]
        for cell in cells:
            used.add(cell)
            to_zero.add(cell)

    for (r, c) in to_zero:
        board[r][c] = 0

    return total


# ---------- gravitație și reumplere ----------
def apply_gravity(board: List[List[int]]) -> None:
    rows, cols = len(board), len(board[0])
    for c in range(cols):
        write_row = rows - 1
        for r in range(rows - 1, -1, -1):
            if board[r][c] != 0:
                board[write_row][c] = board[r][c]
                write_row -= 1
        # rest punem 0
        for r in range(write_row, -1, -1):
            board[r][c] = 0


def refill_board(board: List[List[int]]) -> None:
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 0:
                board[r][c] = random.randint(1, 4)


# ---------- rezolvare cascade (până nu mai apar formațiuni) ----------
def resolve_cascades(board: List[List[int]]) -> Tuple[int, int]:
    """
    Returnează (număr_de_cascade, puncte_obținute) după aplicarea repetată:
    detectare -> eliminare -> gravitație -> reumplere
    se oprește când nu mai există linii sau L/T.
    """
    cascades = 0
    total_points = 0
    while True:
        lines = find_lines(board)
        if not lines:
            break
        l_t = find_L_and_T(board)
        all_form = lines + l_t
        gained = remove_formations(board, all_form)
        total_points += gained
        apply_gravity(board)
        refill_board(board)
        cascades += 1
    return cascades, total_points


# ---------- utilitar pentru verificare linie în jurul unei celule ----------
def has_line_at(board: List[List[int]], r: int, c: int) -> bool:
    if not inside(board, r, c):
        return False
    val = board[r][c]
    if val == 0:
        return False

    rows, cols = len(board), len(board[0])

    # orizontal
    cnt = 1
    cc = c - 1
    while cc >= 0 and board[r][cc] == val:
        cnt += 1
        cc -= 1
    cc = c + 1
    while cc < cols and board[r][cc] == val:
        cnt += 1
        cc += 1
    if cnt >= 3:
        return True

    # vertical
    cnt = 1
    rr = r - 1
    while rr >= 0 and board[rr][c] == val:
        cnt += 1
        rr -= 1
    rr = r + 1
    while rr < rows and board[rr][c] == val:
        cnt += 1
        rr += 1
    return cnt >= 3


# ---------- găsire mutare validă (swap) ----------
def find_valid_swap(board: List[List[int]]) -> Optional[Tuple[int, int, int, int]]:
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            # swap cu dreapta
            if c + 1 < cols and board[r][c] != board[r][c + 1]:
                board[r][c], board[r][c + 1] = board[r][c + 1], board[r][c]
                if has_line_at(board, r, c) or has_line_at(board, r, c + 1):
                    board[r][c], board[r][c + 1] = board[r][c + 1], board[r][c]
                    return (r, c, r, c + 1)
                board[r][c], board[r][c + 1] = board[r][c + 1], board[r][c]

            # swap cu jos
            if r + 1 < rows and board[r][c] != board[r + 1][c]:
                board[r][c], board[r + 1][c] = board[r + 1][c], board[r][c]
                if has_line_at(board, r, c) or has_line_at(board, r + 1, c):
                    board[r][c], board[r + 1][c] = board[r + 1][c], board[r][c]
                    return (r, c, r + 1, c)
                board[r][c], board[r + 1][c] = board[r + 1][c], board[r][c]
    return None


# ---------- încărcare table predefinite ----------
def load_predefined(path: str, rows: int, cols: int) -> List[List[List[int]]]:
    boards = []
    if not os.path.exists(path):
        return boards
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if len(current) == rows:
                    boards.append(current)
                current = []
                continue
            parts = line.split()
            if len(parts) != cols:
                continue
            try:
                row = [int(x) for x in parts]
            except ValueError:
                continue
            current.append(row)
            if len(current) == rows:
                boards.append(current)
                current = []
    if len(current) == rows:
        boards.append(current)
    return boards


# ---------- rulare unui singur joc ----------
def play_one_game(game_id: int, rows: int, cols: int, target: int,
                  initial_board: Optional[List[List[int]]] = None) -> Dict[str, Any]:
    board = deep_copy_board(initial_board) if initial_board is not None else make_random_board(rows, cols)

    total_points = 0
    swaps_done = 0
    total_cascades = 0
    reached = False
    moves_to_target: Optional[int] = None

    # rezolvăm posibilele formațiuni inițiale
    casc_count, gained = resolve_cascades(board)
    total_cascades += casc_count
    total_points += gained

    if total_points >= target:
        reached = True
        moves_to_target = 0
        return {
            "game_id": game_id,
            "points": total_points,
            "swaps": swaps_done,
            "total_cascades": total_cascades,
            "reached_target": reached,
            "stopping_reason": "REACHED_TARGET",
            "moves_to_10000": moves_to_target,
        }

    while True:
        if total_points >= target:
            reached = True
            reason = "REACHED_TARGET"
            break

        swap = find_valid_swap(board)
        if swap is None:
            reached = total_points >= target
            reason = "NO_MOVES"
            break

        r1, c1, r2, c2 = swap
        board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
        swaps_done += 1

        casc_count, gained = resolve_cascades(board)
        total_cascades += casc_count
        total_points += gained

        if total_points >= target and moves_to_target is None:
            moves_to_target = swaps_done

    return {
        "game_id": game_id,
        "points": total_points,
        "swaps": swaps_done,
        "total_cascades": total_cascades,
        "reached_target": reached,
        "stopping_reason": reason,
        "moves_to_10000": moves_to_target,
    }


# ---------- rulare mai multor jocuri + salvare CSV ----------
def run_many_games(num_games: int, rows: int, cols: int, target: int,
                   use_predefined: bool, out_path: str) -> None:
    predefined_boards = []
    if use_predefined:
        predefined_boards = load_predefined(os.path.join("data", "predefined_boards.txt"), rows, cols)

    results = []
    for gid in range(num_games):
        print(f"-> Running game {gid + 1}/{num_games} ...")
        if predefined_boards:
            initial = predefined_boards[gid % len(predefined_boards)]
        else:
            initial = None
        summary = play_one_game(gid, rows, cols, target, initial_board=initial)
        results.append(summary)

    # asigurăm directorul output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["game_id", "points", "swaps", "total_cascades",
                         "reached_target", "stopping_reason", "moves_to_10000"])
        for r in results:
            writer.writerow([
                r["game_id"],
                r["points"],
                r["swaps"],
                r["total_cascades"],
                str(r["reached_target"]),
                r["stopping_reason"],
                "" if r["moves_to_10000"] is None else r["moves_to_10000"],
            ])

    total_points = sum(r["points"] for r in results)
    total_swaps = sum(r["swaps"] for r in results)
    avg_points = total_points / num_games if num_games else 0
    avg_swaps = total_swaps / num_games if num_games else 0

    games_reached = [r for r in results if r["reached_target"] and r["moves_to_10000"] is not None]
    if games_reached:
        avg_swaps_to_target = sum(r["moves_to_10000"] for r in games_reached) / len(games_reached)
    else:
        avg_swaps_to_target = 0

    print(f"Games run: {num_games}")
    print(f"Average points: {avg_points:.2f}")
    print(f"Average swaps: {avg_swaps:.2f}")
    if games_reached:
        print(f"Average swaps to 10000 (only those who reached): {avg_swaps_to_target:.2f}")
    else:
        print("No game reached the target of 10000 points.")
    print(f"\nResults saved to: {out_path}\n")


# ---------- parsare argumente CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Candy Crush automatic simulator (refactored)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--rows", type=int, default=11)
    p.add_argument("--cols", type=int, default=11)
    p.add_argument("--target", type=int, default=10000)
    p.add_argument("--input_predefined", action="store_true")
    p.add_argument("--out", type=str, default=os.path.join("../data/results", "summary.csv"))
    return p.parse_args()


# ---------- entrypoint ----------
def main() -> None:
    args = parse_args()
    random.seed(42)
    run_many_games(args.games, args.rows, args.cols, args.target, args.input_predefined, args.out)


if __name__ == "__main__":
    main()
