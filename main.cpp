#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <climits>
#include <algorithm>
#include <iomanip>

// ─────────────────────────────────────────────
//  Board size limits
// ─────────────────────────────────────────────
static const int MAXR = 50;
static const int MAXC = 50;

// ─────────────────────────────────────────────
//  Cell states stored in board[][]
//    0        = undecided
//   -1        = explicitly marked uncovered
//   N > 0     = covered by piece with id N
// ─────────────────────────────────────────────
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

// ─────────────────────────────────────────────
//  Piece shape definitions
//
//  Each variant is 4 pairs {dr, dc} — offsets
//  from the top-left corner of the piece's
//  bounding box.  Nothing more.
// ─────────────────────────────────────────────

// T-piece: 4 rotations
//
//  T0 (stem down)    T1 (stem right)   T2 (stem up)     T3 (stem left)
//  X X X             X .               . X .             . X
//  . X .             X X               X X X             X X
//                    X .                                 . X
//
const int T_VAR[4][4][2] = {
    {{0,0}, {0,1}, {0,2}, {1,1}},   // T0
    {{0,0}, {1,0}, {2,0}, {1,1}},   // T1
    {{0,1}, {1,0}, {1,1}, {1,2}},   // T2
    {{0,0}, {0,1}, {1,1}, {2,1}},   // T3
};

// Z-piece: 2 shapes (Z and S) × 2 orientations = 4 variants
//
//  Z0 (Z horiz)      Z1 (Z vert)      Z2 (S horiz)      Z3 (S vert)
//  X X .             . X              . X X              X .
//  . X X             X X              X X .              X X
//                    X .                                 . X
//
const int Z_VAR[4][4][2] = {
    {{0,0}, {0,1}, {1,1}, {1,2}},   // Z0
    {{0,1}, {1,0}, {1,1}, {2,0}},   // Z1
    {{0,1}, {0,2}, {1,0}, {1,1}},   // Z2
    {{0,0}, {1,0}, {1,1}, {2,1}},   // Z3
};

// ─────────────────────────────────────────────
//  A single placement: 4 board cells + type
// ─────────────────────────────────────────────
struct Placement {
    int rows[4], cols[4];   // the 4 board cells this piece occupies
    char type;              // 'T' or 'Z'
};

// ─────────────────────────────────────────────
//  Board state — everything the DFS needs
// ─────────────────────────────────────────────
struct State {
    int  board[MAXR][MAXC];     // cell ownership (0 / -1 / piece-id)
    int  weights[MAXR][MAXC];   // input weights, never modified
    int  rows, cols;

    int  cost;                  // sum of weights of UNCOVERED cells so far
    int  undecided_sum;         // sum of weights of UNDECIDED cells

    int  t_count, z_count;      // how many T / Z pieces placed so far
    int  next_id;               // piece id to assign next (starts at 1)
};

// ─────────────────────────────────────────────
//  Best solution found so far
// ─────────────────────────────────────────────
struct Best {
    int  board[MAXR][MAXC];
    int  cost;
    int  t_count, z_count;
};

// ═════════════════════════════════════════════
//  Helper: find first undecided cell (row-major)
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
//  Helper: generate all valid placements of one
//  piece type that cover cell (r, c).
//
//  For each variant and each of its 4 cells as
//  the "anchor" on (r,c), compute the 4 board
//  positions.  Keep the placement if:
//    • all 4 cells are within the board, AND
//    • all 4 cells are currently UNDECIDED.
//  Deduplicate by sorting the 4 cells and using
//  a set.
// ═════════════════════════════════════════════
std::vector<Placement> get_placements(
    const State& s, int r, int c,
    const int VAR[4][4][2], char type)
{
    std::vector<Placement> result;
    // Use a set of sorted cell-tuples to avoid duplicates
    std::set<std::array<std::pair<int,int>, 4>> seen;

    for (int v = 0; v < 4; v++) {
        for (int anchor = 0; anchor < 4; anchor++) {
            // offset of the anchor cell within this variant
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
                    valid = false;
                    break;
                }
                p.rows[i] = row;
                p.cols[i] = col;
            }

            if (!valid) continue;

            // Build a sorted key for deduplication
            std::array<std::pair<int,int>, 4> key;
            for (int i = 0; i < 4; i++)
                key[i] = {p.rows[i], p.cols[i]};
            std::sort(key.begin(), key.end());

            if (seen.count(key)) continue;
            seen.insert(key);
            result.push_back(p);
        }
    }
    return result;
}

// ═════════════════════════════════════════════
//  Apply / undo: place a piece on the board
// ═════════════════════════════════════════════
void apply_piece(State& s, const Placement& p)
{
    int id = s.next_id++;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum -= s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = id;
    }
    if (p.type == 'T') s.t_count++;
    else               s.z_count++;
}

void undo_piece(State& s, const Placement& p)
{
    s.next_id--;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum += s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = UNDECIDED;
    }
    if (p.type == 'T') s.t_count--;
    else               s.z_count--;
}

// ═════════════════════════════════════════════
//  Apply / undo: mark a single cell as uncovered
// ═════════════════════════════════════════════
void apply_uncover(State& s, int r, int c)
{
    s.cost         += s.weights[r][c];
    s.undecided_sum -= s.weights[r][c];
    s.board[r][c]   = UNCOVERED;
}

void undo_uncover(State& s, int r, int c)
{
    s.cost         -= s.weights[r][c];
    s.undecided_sum += s.weights[r][c];
    s.board[r][c]   = UNDECIDED;
}

// ═════════════════════════════════════════════
//  Recursive BB-DFS  (no pruning yet)
// ═════════════════════════════════════════════
void dfs(State& s, Best& best)
{
    auto [r, c] = first_undecided(s);

    // ── Base case: no undecided cells left ──────
    if (r == -1) {
        if (s.cost < best.cost) {
            best.cost    = s.cost;
            best.t_count = s.t_count;
            best.z_count = s.z_count;
            // save the board layout
            for (int i = 0; i < s.rows; i++)
                for (int j = 0; j < s.cols; j++)
                    best.board[i][j] = s.board[i][j];
        }
        return;
    }

    // ── Branch 1: try every T placement covering (r,c) ──
    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    for (auto& p : t_moves) {
        apply_piece(s, p);
        dfs(s, best);
        undo_piece(s, p);
    }

    // ── Branch 2: try every Z placement covering (r,c) ──
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');
    for (auto& p : z_moves) {
        apply_piece(s, p);
        dfs(s, best);
        undo_piece(s, p);
    }

    // ── Branch 3: mark (r,c) as uncovered ───────
    apply_uncover(s, r, c);
    dfs(s, best);
    undo_uncover(s, r, c);
}

// ═════════════════════════════════════════════
//  Print the solution board
// ═════════════════════════════════════════════
void print_solution(const State& s, const Best& best)
{
    // Figure out the type of each piece id from the best board.
    // We need to map piece_id → 'T' or 'Z'.
    // We can recover this by re-examining which cells share an id
    // and cross-referencing with T/Z shapes — but actually the
    // simplest approach: store a type map during the search.
    // For now we label covered cells as "P<id>" generically and
    // note that proper T/Z labelling requires a small extra map.
    // (We will fix this in the next step.)

    std::cout << "\nBest cost: " << best.cost
              << "  (T=" << best.t_count
              << ", Z=" << best.z_count << ")\n\n";

    // Collect all piece ids to figure out column width
    int max_id = 0;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            if (best.board[i][j] > max_id)
                max_id = best.board[i][j];

    // Width: "T99" or "Z99" → 3 chars, weights up to 3 digits
    int w = std::max(3, (int)std::to_string(max_id).size() + 1);

    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            int v = best.board[i][j];
            std::string cell;
            if (v == UNCOVERED)     cell = std::to_string(s.weights[i][j]);
            else if (v == UNDECIDED) cell = "?";   // shouldn't happen
            else                     cell = "P" + std::to_string(v);
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

// ═════════════════════════════════════════════
//  Compute trivial lower bound:
//  sum of the (ab mod 4) smallest weights.
//  If ab % 4 == 0, the lower bound is 0.
// ═════════════════════════════════════════════
int trivial_lower_bound(const State& s)
{
    int k = (s.rows * s.cols) % 4;
    if (k == 0) return 0;

    std::vector<int> all_weights;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            all_weights.push_back(s.weights[i][j]);

    std::sort(all_weights.begin(), all_weights.end());
    int lb = 0;
    for (int i = 0; i < k; i++) lb += all_weights[i];
    return lb;
}

// ═════════════════════════════════════════════
//  Main
// ═════════════════════════════════════════════
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    // ── Read input ───────────────────────────────
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr << "Cannot open file: " << argv[1] << "\n";
        return 1;
    }

    State s{};
    fin >> s.rows >> s.cols;
    s.cost = 0;
    s.undecided_sum = 0;
    s.t_count = 0;
    s.z_count = 0;
    s.next_id = 1;

    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            fin >> s.weights[i][j];
            s.board[i][j]    = UNDECIDED;
            s.undecided_sum += s.weights[i][j];
        }
    }

    // ── Initialise best to "everything uncovered" ─
    Best best{};
    best.cost    = s.undecided_sum;  // worst possible cost
    best.t_count = 0;
    best.z_count = 0;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            best.board[i][j] = UNCOVERED;

    int lb = trivial_lower_bound(s);
    std::cout << "Trivial lower bound: " << lb << "\n";
    std::cout << "Starting search...\n";

    // ── Run DFS ──────────────────────────────────
    dfs(s, best);

    // ── Print result ─────────────────────────────
    print_solution(s, best);

    return 0;
}