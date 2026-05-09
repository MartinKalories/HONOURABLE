# Optimizing SAVEMEMP4.py for Reduced Disk Usage

This document provides instructions for modifying `SAVEMEMP4.py` to reduce unnecessary disk writes during Bayesian optimization runs, while **preserving all crash recovery safeguards**.

## Important Safeguards to Preserve

**DO NOT REMOVE** the following crash recovery mechanisms:
- `save_checkpoint(opt)` calls in exception handlers (e.g., in `except Exception` and `KeyboardInterrupt` blocks)
- The ability to resume from `checkpoint_npz` and `optimizer_checkpoint` files
- Any state saving that enables restarting after a crash

These safeguards ensure that optimization progress is not lost if the script crashes or is interrupted.

## Recommended Optimizations

Apply these changes **only to `SAVEMEMP4.py`** to reduce disk I/O without compromising recovery:

### 1. Reduce Checkpoint Frequency
- **Current**: `save_checkpoint(opt)` is called after every completed trial
- **Optimized**: Call `save_checkpoint(opt)` only every N trials (e.g., every 5 trials) or only when necessary
- **Why**: Frequent checkpoints are redundant if you can tolerate losing a few trials' progress
- **Implementation**: Add a counter and modulo check before `save_checkpoint(opt)`

### 2. Remove Live Plot Image Saves
- **Current**: `update_live_plot()` saves `RUN_ID_live_progress.png` after every trial
- **Optimized**: Remove the `fig_live.savefig(...)` call in `update_live_plot()`
- **Why**: The plot is for visual monitoring; saving every iteration creates many unnecessary files
- **Alternative**: Keep the plot display but skip disk saves

### 3. Eliminate Duplicate CSV Logging
- **Current**: Both `append_trial_to_csv()` (per trial) and `save_full_trial_log_csv()` (at end) write CSV data
- **Optimized**: Remove `save_full_trial_log_csv(trial_log, all_trials_filename)` from `save_best_results()`
- **Why**: The per-trial append already creates a complete CSV; the full rewrite is redundant
- **Keep**: The append method for incremental logging

### 4. Avoid Duplicate Optimizer State Saves
- **Current**: `dump(opt, optimizer_checkpoint, ...)` in `save_checkpoint()` and final `dump()` at end
- **Optimized**: Remove the final `dump()` call, as `save_checkpoint()` already saves the optimizer state
- **Why**: The final dump duplicates the last checkpoint
- **Keep**: Optimizer saves in checkpoints for recovery

### 5. Optional: Reduce Returned Data from `train_one_run()`
- **Current**: Returns full `history_val_loss` and `history_loss` arrays
- **Optimized**: Modify `train_one_run()` to return only scalar losses when called from optimization
- **Why**: Full history arrays are not needed for optimization; they consume memory and disk space in logs
- **Note**: This requires editing `thirdopt.py` as well, but only affects optimization runs

## Example Modified Code Snippets

### Checkpoint Frequency Reduction
```python
# Add counter
checkpoint_counter = 0
CHECKPOINT_INTERVAL = 5  # Save every 5 trials

# In main loop, after trial completion:
checkpoint_counter += 1
if checkpoint_counter % CHECKPOINT_INTERVAL == 0:
    save_checkpoint(opt)
```

### Remove Live Plot Saves
```python
def update_live_plot():
    # ... existing plot code ...
    fig_live.tight_layout()
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()
    plt.pause(0.01)
    # Remove: fig_live.savefig(...)
```

### Remove Duplicate CSV
```python
def save_best_results():
    # ... existing code ...
    # Remove: save_full_trial_log_csv(trial_log, all_trials_filename)
```

## Testing Recommendations

After modifications:
1. Run a short optimization (e.g., `n_calls = 5`) to verify functionality
2. Simulate a crash (e.g., KeyboardInterrupt) and confirm resume works
3. Check that final results and best parameters are still saved correctly
4. Verify CSV logging is complete and optimizer state can be loaded

## Performance Impact

These changes should:
- Reduce disk writes by ~60-80% during optimization
- Maintain all crash recovery capabilities
- Keep memory usage similar (no VRAM changes)
- Preserve all final output files (best results, final plot, etc.)

## Reverting Changes

If issues arise, revert to the original `SAVEMEMP4.py` from the repository. The safeguards are critical for long-running optimizations.