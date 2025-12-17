# Function-Aware Population Sizing for SOCIAL

## Overview
This implementation adds **function-aware population sizing** to SOCIAL, where `NUM_NODES` is selected **before optimization** based on the function category. `NUM_NODES` remains **fixed** throughout the entire run (no adaptive changes).

## Function Categorization

### Unimodal Functions → NUM_NODES = 60
- F1  Sphere
- F2  Schwefel 2.22
- F3  Schwefel 1.2
- F4  Schwefel 2.21
- F5  Rosenbrock
- F7  Quartic (with noise)

**Rationale**: Unimodal functions require less population diversity, so smaller populations (60) are sufficient.

### Discontinuous (Step-like) → NUM_NODES = 150
- F6  Step

**Rationale**: Discontinuous functions benefit from moderate population size to handle plateaus.

### Multimodal Functions → NUM_NODES = 150
- F8   Schwefel 2.26
- F9   Rastrigin
- F10  Ackley
- F11  Griewank
- F14  Foxholes
- F15  Kowalik
- F16  Six-Hump Camelback
- F17  Branin
- F18  Goldstein–Price
- F19  Hartmann 3D
- F20  Hartmann 6D (Shekel1)
- F21  Shekel 5 (Shekel2)
- F22  Shekel 7 (Shekel3)
- F23  Shekel 10 (Shekel4)

**Rationale**: Multimodal functions require larger populations (150) to maintain diversity and explore multiple basins.

### Penalized/Constrained → NUM_NODES = 200
- F12  Penalized 1
- F13  Penalized 2

**Rationale**: Constrained/penalized functions require the largest populations (200) to handle constraint boundaries and penalty landscapes.

### Graph-Structured (Non-Classical) → NUM_NODES = 120
- Graph-Laplacian objective (if used)

**Rationale**: Graph-structured objectives benefit from moderate population sizes aligned with graph topology.

## Implementation Details

### Configuration (`social/config.py`)

1. **Function Categories Dictionary**:
   ```python
   FUNCTION_CATEGORIES = {
       "unimodal": {...},
       "discontinuous": {...},
       "multimodal": {...},
       "penalized": {...},
       "graph": {...}
   }
   ```

2. **Helper Methods**:
   - `Config.get_num_nodes_for_function(function_name)` → Returns NUM_NODES for a function
   - `Config.get_function_category(function_name)` → Returns category name

### Main Runner (`main.py`)

**Before each function run**:
1. Identifies function category
2. Gets NUM_NODES from mapping
3. Creates function-specific config with appropriate NUM_NODES
4. NUM_NODES remains **fixed** for all runs of that function

**Key Code**:
```python
# FUNCTION-AWARE POPULATION SIZING
function_category = config.get_function_category(func_name)
num_nodes_for_function = config.get_num_nodes_for_function(func_name)

# Create function-specific config
func_config = Config()
func_config.__dict__.update(config.__dict__)
func_config.NUM_NODES = num_nodes_for_function  # Fixed for entire run
```

## Usage

### Default Behavior
When running `python3 main.py`, each function automatically uses its category-specific NUM_NODES:

```bash
python3 main.py
```

**Output Example**:
```
Function: Sphere
  Category: unimodal
  NUM_NODES: 60 (fixed for entire run)

Function: Step
  Category: discontinuous
  NUM_NODES: 150 (fixed for entire run)

Function: Rastrigin
  Category: multimodal
  NUM_NODES: 150 (fixed for entire run)

Function: Penalized
  Category: penalized
  NUM_NODES: 200 (fixed for entire run)
```

### Verification
All 23 benchmark functions are properly categorized:
- ✓ 6 unimodal functions → NUM_NODES = 60
- ✓ 1 discontinuous function → NUM_NODES = 150
- ✓ 14 multimodal functions → NUM_NODES = 150
- ✓ 2 penalized functions → NUM_NODES = 200

## Design Principles

### ✅ What This Does
- **Pre-optimization selection**: NUM_NODES chosen before optimization starts
- **Fixed during run**: NUM_NODES never changes during optimization
- **Category-based**: Selection based on function structure, not fitness
- **Transparent**: Clear mapping, easy to verify and modify

### ❌ What This Does NOT Do
- **No adaptive sizing**: NUM_NODES does not change based on fitness or convergence
- **No mid-run changes**: Population size remains constant throughout
- **No dynamic growth/pruning**: No agents added or removed during optimization
- **No fitness-based tuning**: Selection is based on function category only

## Research Context

This is an **experimental design choice**, not an adaptive optimizer feature. It reflects the hypothesis that:

- **Unimodal problems** need smaller populations (faster convergence)
- **Multimodal problems** need larger populations (maintain diversity)
- **Constrained problems** need largest populations (handle boundaries)

The implementation is **minimal, transparent, and reproducible** - NUM_NODES is determined solely by function name, making results fully reproducible.

## Files Modified

1. **social/config.py**
   - Added `FUNCTION_CATEGORIES` dictionary
   - Added `NUM_NODES_BY_CATEGORY` reference
   - Added `get_num_nodes_for_function()` static method
   - Added `get_function_category()` static method

2. **main.py**
   - Modified to create function-specific configs
   - Sets NUM_NODES before optimizer creation
   - Prints category and NUM_NODES for transparency

## CSV Output Format

The CSV output format remains **unchanged**. Results are saved exactly as before, with the only difference being that NUM_NODES varies by function category.

