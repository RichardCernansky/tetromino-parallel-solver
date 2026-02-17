#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <string>
#include <climits>
#include <algorithm>
#include <iomanip>
#include <chrono>

// ─────────────────────────────────────────────
//  Board size limit
// ─────────────────────────────────────────────
static const int MAXR = 50;
static const int MAXC = 50;

// ─────────────────────────────────────────────
//  Cell states in board[][]
//    UNDECIDED (0)  — not yet decided
//    UNCOVERED (-1) — explicitly left bare → adds to cost
//    N > 0          — covered by piece id N
// ─────────────────────────────────────────────
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

// ─────────────────────────────────────────────
//  Piece shape templates
//  Each variant stores the 4 {dr,dc} offsets of

const int T_VAR[4][4][2] = {
    {{0,0},{0,1},{0,2},{1,1}},   // T0
    {{0,0},{1,0},{2,0},{1,1}},   // T1
    {{0,1},{1,0},{1,1},{1,2}},   // T2
    {{0,0},{0,1},{1,1},{2,1}},   // T3
};
const int Z_VAR[4][4][2] = {
    {{0,0},{0,1},{1,1},{1,2}},   // Z0
    {{0,1},{1,0},{1,1},{2,0}},   // Z1
    {{0,1},{0,2},{1,0},{1,1}},   // Z2 (S horiz)
    {{0,0},{1,0},{1,1},{2,1}},   // Z3 (S vert)
};

// ─────────────────────────────────────────────
//  A concrete placement: 4 board cells + type
// ─────────────────────────────────────────────
struct Placement {
    int  rows[4], cols[4];
    char type;          // 'T' or 'Z'
};

// ─────────────────────────────────────────────
//  Full board state threaded through the DFS
// ─────────────────────────────────────────────
struct State {
    int  board[MAXR][MAXC];       // cell ownership
    int  weights[MAXR][MAXC];     // fixed input weights
    char piece_type[MAXR*MAXC];   // piece_type[id-1] → 'T' or 'Z'
    int  rows, cols;

    int  cost;           // sum of weights of UNCOVERED cells so far
    int  undecided_sum;  // sum of weights of UNDECIDED cells

    int  t_count;        // T pieces placed so far
    int  z_count;        // Z pieces placed so far
    int  next_id;        // next piece id (1-based)
};

// ─────────────────────────────────────────────
//  Best solution snapshot
// ─────────────────────────────────────────────
struct Best {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  t_count, z_count;
    int  next_id;        // how many pieces were placed
};

// ─────────────────────────────────────────────
//  Statistics
// ─────────────────────────────────────────────
static long long g_calls = 0;   // total recursive calls

// ═════════════════════════════════════════════
//  Find first UNDECIDED cell in row-major order.
//  Returns {-1,-1} when the board is fully decided.
// ═════════════════════════════════════════════
std::pair<int,int> first_undecided(const State& s)
{
    for (int r = 0; r < s.rows; r++)
        for (int c = 0; c < s.cols; c++)
            if (s.board[r][c] == UNDECIDED)
                return {r, c};
    return {-1, -1};
}

// ═════════════════════════════════════════════
//  Generate all valid placements of one piece
//  type that cover cell (r, c).
//
//  Strategy:
//    For every variant, treat each of its 4
//    cells as the "anchor" landing on (r,c).
//    Compute the 4 actual board positions and
//    accept if all are in-bounds and UNDECIDED.
//    Deduplicate via a sorted-cell set.
// ═════════════════════════════════════════════
std::vector<Placement> get_placements(
    const State& s, int r, int c,
    const int VAR[4][4][2], char type)
{
    std::vector<Placement> result;
    std::set<std::array<std::pair<int,int>,4>> seen;

    for (int v = 0; v < 4; v++) {
        for (int anchor = 0; anchor < 4; anchor++) {
            int ar = VAR[v][anchor][0];
            int ac = VAR[v][anchor][1];

            Placement p;
            p.type = type;
            bool valid = true;

            for (int i = 0; i < 4; i++) {
                int row = r + VAR[v][i][0] - ar;
                int col = c + VAR[v][i][1] - ac;
                if (row < 0 || row >= s.rows ||
                    col < 0 || col >= s.cols  ||
                    s.board[row][col] != UNDECIDED) {
                    valid = false; break;
                }
                p.rows[i] = row;
                p.cols[i] = col;
            }
            if (!valid) continue;

            std::array<std::pair<int,int>,4> key;
            for (int i = 0; i < 4; i++) key[i] = {p.rows[i], p.cols[i]};
            std::sort(key.begin(), key.end());
            if (seen.count(key)) continue;
            seen.insert(key);
            result.push_back(p);
        }
    }
    return result;
}

// ═════════════════════════════════════════════
//  Apply / undo helpers
//  These are exact mirrors of each other — if
//  you change one, change the other.
// ═════════════════════════════════════════════
void apply_piece(State& s, const Placement& p)
{
    int id = s.next_id++;
    s.piece_type[id-1] = p.type;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum        -= s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = id;
    }
    if (p.type == 'T') s.t_count++;
    else               s.z_count++;
}

void undo_piece(State& s, const Placement& p)
{
    s.next_id--;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum        += s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = UNDECIDED;
    }
    if (p.type == 'T') s.t_count--;
    else               s.z_count--;
}

void apply_uncover(State& s, int r, int c)
{
    s.cost          += s.weights[r][c];
    s.undecided_sum -= s.weights[r][c];
    s.board[r][c]    = UNCOVERED;
}

void undo_uncover(State& s, int r, int c)
{
    s.cost          -= s.weights[r][c];
    s.undecided_sum += s.weights[r][c];
    s.board[r][c]    = UNDECIDED;
}

// ═════════════════════════════════════════════
//  Parity pruning helper
//
//  At the end, |t_count - z_count| <= 1.
//  At any point during the search, we know:
//    - current counts: t_count, z_count
//    - max extra pieces we could ever place:
//      floor(undecided_cells / 4)
//      (conservative: undecided_sum / min_weight
//       would be tighter but expensive to compute)
//
//  We use undecided cell COUNT, not sum, so we
//  need to track it.  We pass it in as a param
//  (it's cheaply maintained alongside undecided_sum).
//
//  Returns true  → this branch CANNOT satisfy
//                  the parity constraint → prune.
//  Returns false → parity is still satisfiable.
// ═════════════════════════════════════════════
bool parity_prune(int t_count, int z_count, int undecided_cells)
{
    int diff = t_count - z_count;   // positive: more T placed
    if (diff < 0) diff = -diff;     // |diff|

    // Maximum additional pieces we could place
    int max_more = undecided_cells / 4;

    // After placing up to max_more pieces (all of the minority type),
    // the minimum achievable |diff| is:
    //   if diff <= max_more: we can close the gap → min_diff = 0 or 1 depending on parity
    //   if diff >  max_more: we cannot close the gap at all → prune
    //
    // Actually the exact condition is:
    //   We need to place at least (diff - 1) more of the minority type
    //   to get |final_diff| <= 1.  That requires max_more >= diff - 1.
    //   i.e. prune when diff - 1 > max_more  →  diff > max_more + 1
    return (diff > max_more + 1);
}

// ═════════════════════════════════════════════
//  Main recursive BB-DFS with full pruning
//
//  Pruning rules (applied at the TOP of each call,
//  before any branching):
//
//  P1 — Cost bound:
//       If cost >= best.cost, this branch is
//       already at least as expensive as the
//       best known solution.  Prune.
//
//  P2 — Optimistic bound:
//       Even if we covered every remaining
//       undecided cell for free, the cost is
//       still s.cost.  So if cost >= best.cost
//       is already handled by P1.  But we also
//       check: if cost + 0 >= best.cost → same.
//       (P1 covers this; listed separately for
//       clarity in the assignment context.)
//
//  P3 — Parity:
//       If the T/Z balance cannot be restored
//       within the remaining undecided cells.
//
//  P4 — Trivial lower bound early exit:
//       If best.cost == trivial_lb, we have an
//       optimal solution — stop everything.
// ═════════════════════════════════════════════
void dfs(State& s, Best& best, int trivial_lb,
         int undecided_cells, bool& found_optimal)
{
    ++g_calls;

    // ── P4: already optimal ──────────────────────
    if (found_optimal) return;

    // ── P1: cost bound ───────────────────────────
    // current cost alone already meets or beats best
    if (s.cost >= best.cost) return;

    // ── P3: parity ───────────────────────────────
    if (parity_prune(s.t_count, s.z_count, undecided_cells)) return;

    // ── Find first undecided cell ────────────────
    auto [r, c] = first_undecided(s);

    // ── Base case: board fully decided ───────────
    if (r == -1) {
        // cost < best.cost is guaranteed by P1 above
        best.cost    = s.cost;
        best.t_count = s.t_count;
        best.z_count = s.z_count;
        best.next_id = s.next_id;
        for (int i = 0; i < s.rows; i++)
            for (int j = 0; j < s.cols; j++)
                best.board[i][j] = s.board[i][j];
        for (int k = 0; k < s.next_id - 1; k++)
            best.piece_type[k] = s.piece_type[k];

        if (best.cost == trivial_lb) found_optimal = true;
        return;
    }

    // ── Collect all piece placements covering (r,c) ─
    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    // ── Move ordering (Note 1) ───────────────────
    //
    //  We merge all piece placements into one list
    //  and sort by the SUM OF COVERED WEIGHTS,
    //  descending.  Covering heavier cells first
    //  produces a better (lower) cost early, which
    //  tightens the bound and prunes more branches.
    //
    //  The "uncover" branch always comes last: it
    //  adds cost immediately so it is the least
    //  promising move.

    // Compute coverage weight for each placement
    auto coverage = [&](const Placement& p) {
        int sum = 0;
        for (int i = 0; i < 4; i++)
            sum += s.weights[p.rows[i]][p.cols[i]];
        return sum;
    };

    // Merge T and Z moves into one sorted list
    std::vector<Placement> all_moves;
    all_moves.reserve(t_moves.size() + z_moves.size());
    for (auto& p : t_moves) all_moves.push_back(p);
    for (auto& p : z_moves) all_moves.push_back(p);

    std::sort(all_moves.begin(), all_moves.end(),
        [&](const Placement& a, const Placement& b) {
            return coverage(a) > coverage(b);  // heaviest first
        });

    // ── Branch: all piece placements (sorted) ────
    for (auto& p : all_moves) {
        if (found_optimal) return;
        apply_piece(s, p);
        dfs(s, best, trivial_lb, undecided_cells - 4, found_optimal);
        undo_piece(s, p);
    }

    // ── Branch: mark (r,c) as uncovered ──────────
    //
    //  Pre-check: if uncovering this cell alone
    //  already reaches best.cost, skip entirely.
    if (found_optimal) return;
    if (s.cost + s.weights[r][c] < best.cost) {
        apply_uncover(s, r, c);
        dfs(s, best, trivial_lb, undecided_cells - 1, found_optimal);
        undo_uncover(s, r, c);
    }
}

// ═════════════════════════════════════════════
//  Compute trivial lower bound:
//    k = (rows * cols) mod 4
//    lb = sum of k smallest weights on the board
//  If k == 0, lb = 0.
// ═════════════════════════════════════════════
int trivial_lower_bound(const State& s)
{
    int k = (s.rows * s.cols) % 4;
    if (k == 0) return 0;

    std::vector<int> w;
    w.reserve(s.rows * s.cols);
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            w.push_back(s.weights[i][j]);
    std::sort(w.begin(), w.end());

    int lb = 0;
    for (int i = 0; i < k; i++) lb += w[i];
    return lb;
}

// ═════════════════════════════════════════════
//  Print the solution in the required format:
//    covered cells  → [T|Z]<id>
//    uncovered cells → their weight
// ═════════════════════════════════════════════
void print_solution(const State& s, const Best& best)
{
    std::cout << "\n=== Solution ===\n";
    std::cout << "Cost:    " << best.cost << "\n";
    std::cout << "T pieces: " << best.t_count << "\n";
    std::cout << "Z pieces: " << best.z_count << "\n\n";

    // column width: widest possible label
    // piece labels: "T" or "Z" + up to 4 digits for large boards
    // weight labels: up to 3 digits (max weight = 100)
    int max_id = best.next_id - 1;
    int w = std::max(3, (int)std::to_string(max_id).size() + 1);

    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            int v = best.board[i][j];
            std::string cell;
            if (v == UNCOVERED) {
                cell = std::to_string(s.weights[i][j]);
            } else {
                // v is the piece id (1-based)
                char t = best.piece_type[v - 1];
                cell = std::string(1, t) + std::to_string(v);
            }
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

// ═════════════════════════════════════════════
//  Main
// ═════════════════════════════════════════════
int main(int argc, char* argv[])
{
    // Check if user provided an input filename as a command-line argument
    // argc = argument count, argv = argument vector (array of strings)
    // argv[0] = program name, argv[1] = first argument (the filename)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;  // Exit with error code
    }

    // Open the input file
    // argv[1] is the filename the user typed (e.g., "mapa3_11.txt")
    std::ifstream fin(argv[1]);
    if (!fin) {
        // File doesn't exist or can't be opened
        std::cerr << "Cannot open: " << argv[1] << "\n";
        return 1;
    }

    // ── Read board dimensions and weights ────────

    // Create an empty State struct and zero-initialize it with {}
    State s{};

    // Read first line: "rows cols"
    // Example: "3 11" means 3 rows, 11 columns
    fin >> s.rows >> s.cols;

    // Initialize the counters for this state
    s.cost = 0;              // No cells uncovered yet
    s.undecided_sum = 0;     // Will accumulate as we read weights
    s.t_count = 0;           // No T pieces placed yet
    s.z_count = 0;           // No Z pieces placed yet
    s.next_id = 1;           // Piece IDs start from 1

    // Remember total cell count - we'll need it for parity pruning
    int total_cells = s.rows * s.cols;

    // Read the weight matrix row by row
    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            // Read one integer weight from the file
            fin >> s.weights[i][j];

            // Mark this cell as undecided (not yet covered or uncovered)
            s.board[i][j] = UNDECIDED;

            // Add this weight to the sum of all undecided cells
            // At the start, ALL cells are undecided, so this sums everything
            s.undecided_sum += s.weights[i][j];
        }
    }

    // ── Initialize the "best solution so far" ────

    // Start with the absolute WORST possible tiling: everything uncovered
    Best best{};
    best.cost = s.undecided_sum;  // Cost if we placed zero pieces
    best.t_count = 0;             // Zero T pieces
    best.z_count = 0;             // Zero Z pieces
    best.next_id = 1;             // No pieces at all

    // Mark every cell as uncovered in the initial "worst solution"
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            best.board[i][j] = UNCOVERED;

    // ── Compute the theoretical best possible cost ──

    // lb = lower bound = the cheapest this problem could EVER be
    // If we ever reach this cost, we can stop immediately (optimal)
    int lb = trivial_lower_bound(s);

    // Print some info before starting the search
    std::cout << "Board:              " << s.rows << " x " << s.cols << "\n";
    std::cout << "Trivial lower bound: " << lb << "\n";
    std::cout << "Starting search...\n";

    // This flag will be set to true if we reach the lower bound (optimal solution)
    // Start timing the search
    auto t0 = std::chrono::high_resolution_clock::now();
    // THE ACTUAL SEARCH HAPPENS HERE
    // Pass state by reference (modified in-place)
    // Pass best by reference (will be updated when better solutions are found)
    // Pass lb by value (never changes)
    // Pass total_cells by value (for parity pruning)
    // Pass found_optimal by reference (set to true if optimal reached)
    // ── Run the DFS search ───────────────────────
    bool found_optimal = false;
    dfs(s, best, lb, total_cells, found_optimal);

    // Stop timing
    auto t1 = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in seconds
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ── Print results ────────────────────────────

    // g_calls is a global counter incremented at the start of every DFS call
    std::cout << "Recursive calls:    " << g_calls << "\n";

    // Print time with 3 decimal places
    std::cout << "Wall time:          " << std::fixed
              << std::setprecision(3) << elapsed << " s\n";

    // Did we prove optimality?
    if (found_optimal)
        std::cout << "Result: OPTIMAL (reached trivial lower bound)\n";
    else
        std::cout << "Result: best found (lower bound not reached)\n";

    // Print the actual tiling (the board with T1, Z2, weights, etc.)
    // best now contains the best solution found during the search
    print_solution(s, best);

    return 0;  // Success
}