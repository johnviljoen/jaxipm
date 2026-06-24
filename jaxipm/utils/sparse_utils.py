import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp

def stack(bcoo_list):
    bcoo_list = [A[None,...] for A in bcoo_list]
    return jsp.bcoo_concatenate(bcoo_list, dimension=0)

def hstack(bcoo_list):
    return jsp.bcoo_concatenate(bcoo_list, dimension=1)

def vstack(bcoo_list):
    return jsp.bcoo_concatenate(bcoo_list, dimension=0)

def diagflat(data):
    n    = data.size
    ind  = jnp.arange(n, dtype=jnp.int32)
    indices = jnp.stack([ind, ind], axis=1)        # shape (n, 2)
    return jsp.BCOO((data, indices), shape=(n, n))

def zeros(shape):
    ndim = len(shape)
    data    = jnp.empty((0,))          # no stored values
    indices = jnp.empty((0, ndim), dtype=jnp.int32) # no coordinates
    return jsp.BCOO((data, indices), shape=shape)

def ones_with_nse(shape, nse):
    """Create a BCOO with explicit ones at sequential row indices.

    Used for padding BCOOs to a target nse for lax.switch compatibility.
    Indices iterate row-by-row: (0,0), (1,0), (2,0), ... wrapping as needed.
    """
    rows = shape[0]
    cols = shape[1] if len(shape) > 1 else 1
    data = jnp.ones(nse)
    row_indices = jnp.arange(nse, dtype=jnp.int32) % rows
    col_indices = (jnp.arange(nse, dtype=jnp.int32) // rows) % cols
    indices = jnp.stack([row_indices, col_indices], axis=1)
    return jsp.BCOO((data, indices), shape=shape)

def sum_duplicates_to_pattern(
    target_indices: jnp.ndarray,
    source_bcoo: jsp.BCOO,
    nse: int
) -> jsp.BCOO:
    """Sum duplicate entries in source_bcoo into target sparsity pattern.

    Unlike sum_duplicates() + conform(), this avoids creating out-of-bounds
    padding entries. Source entries not in target are ignored.

    Args:
        target_indices: (nse, 2) array of target 2D indices (must be sorted row-major)
        source_bcoo: source BCOO matrix (may have duplicates)
        nse: number of target entries (static for JIT compatibility)

    Returns:
        BCOO with target_indices and summed data
    """
    num_cols = source_bcoo.shape[1]
    source_indices_1d = source_bcoo.indices[:, 0] * num_cols + source_bcoo.indices[:, 1]
    target_indices_1d = target_indices[:, 0] * num_cols + target_indices[:, 1]

    # Find where each source index maps in target
    insert_positions = jnp.searchsorted(target_indices_1d, source_indices_1d)
    insert_positions_clamped = jnp.minimum(insert_positions, nse - 1)

    # Check which source entries actually match target indices
    matches = target_indices_1d[insert_positions_clamped] == source_indices_1d

    # Use segment_sum to accumulate values at each target position
    # Non-matching entries go to a dummy position (will be discarded)
    safe_positions = jnp.where(matches, insert_positions_clamped, nse)
    data = jax.ops.segment_sum(source_bcoo.data, safe_positions, num_segments=nse + 1)
    data = data[:nse]  # discard the dummy accumulator

    return jsp.BCOO((data, target_indices), shape=source_bcoo.shape)

def zeros_like(arr):
    return zeros(arr.shape)

def eye(n):
    inds  = jnp.arange(n, dtype=jnp.int32)
    data  = jnp.ones(n)
    indices = jnp.stack([inds, inds], axis=1)
    return jsp.BCOO((data, indices), shape=(n, n))

def add(A, B, start_indices=[0,0]):
    # check that B fits entirely in A so we cannot place data in an illegal location
    start_indices = jnp.array(start_indices, dtype=A.indices.dtype)
    shifted_indices = B.indices + start_indices
    B = jsp.BCOO((B.data, shifted_indices), shape=jnp.array(B.shape) + start_indices)
    assert all([dimA >= dimB for dimA, dimB in zip(A.shape, B.shape)])
    return A + jsp.BCOO([B.data, B.indices], shape=A.shape)

def triu(mat: jsp.BCOO, k: int = 0, nse: int = None) -> jsp.BCOO:
    # nse is the number of upper triangular elements (required for JIT)
    mask = mat.indices[:, 1] >= (mat.indices[:, 0] + k)

    if nse is None:
        new_data = mat.data[mask]
        new_indices = mat.indices[mask]
        return jsp.BCOO((new_data, new_indices), shape=mat.shape)

    # JIT-compatible: sort so upper triangular elements come first, then slice
    sort_idx = jnp.argsort(~mask)  # False (0) for upper tri sorts first
    sorted_data = mat.data[sort_idx][:nse]
    sorted_indices = mat.indices[sort_idx][:nse]
    return jsp.BCOO((sorted_data, sorted_indices), shape=mat.shape)

def conform_bcoo_to_new_sparsity(
    target_indices: jnp.ndarray,
    source_bcoo: jsp.BCOO
) -> jsp.BCOO:
    """WARNING: ensure both sets of indices are sorted!!!"""
    num_cols = source_bcoo.shape[1]
    # print(f"conform: source_bcoo.shape = {source_bcoo.shape}, num_cols = {num_cols}")
    # print(f"conform: target max indices = ({target_indices[:, 0].max()}, {target_indices[:, 1].max()})")
    source_indices_1d = source_bcoo.indices[:, 0] * num_cols + source_bcoo.indices[:, 1]
    target_indices_1d = target_indices[:, 0] * num_cols + target_indices[:, 1]
    insert_positions = jnp.searchsorted(target_indices_1d, source_indices_1d)

    # # Debug: check if all source indices exist in target
    # insert_positions_clamped = jnp.minimum(insert_positions, len(target_indices_1d) - 1)
    # matches = target_indices_1d[insert_positions_clamped] == source_indices_1d
    # num_matches = jnp.sum(matches)
    # print(f"conform: source has {len(source_indices_1d)} entries, {num_matches} match target")
    # if num_matches < len(source_indices_1d):
    #     # Show first few mismatches
    #     mismatch_idx = jnp.where(~matches)[0][:5]
    #     print(f"conform: first mismatches at source idx {mismatch_idx}")
    #     print(f"conform: source 2D indices: {source_bcoo.indices[mismatch_idx]}")
    #     print(f"conform: source_1d at those: {source_indices_1d[mismatch_idx]}")
    #     print(f"conform: target_1d at insert pos: {target_indices_1d[insert_positions_clamped[mismatch_idx]]}")

    data = jnp.full_like(target_indices_1d, 0, dtype=float)
    data = data.at[insert_positions].set(source_bcoo.data)
    return jsp.BCOO((data, target_indices), shape=source_bcoo.shape)

def find_bcsr_diag_indices(csr_matrix) -> jnp.ndarray:
    counts = jnp.diff(csr_matrix.indptr)
    row_indices = jnp.repeat(jnp.arange(csr_matrix.shape[0]), repeats=counts)
    is_diag_mask = (row_indices == csr_matrix.indices)
    return jnp.argwhere(is_diag_mask)

def expand_vector(indices, shape, values):
    return jnp.zeros(shape).at[indices].set(values)

def group_data_by_row(sorted_bcoo):
    """Groups sorted sparse data elements by their row index."""
    bcoo_data, bcoo_indices = sorted_bcoo.data, sorted_bcoo.indices
    # Handle empty arrays
    if bcoo_data.size == 0:
        return [jnp.array([])]
    row_indices = bcoo_indices[:, 0]
    change_points = jnp.diff(row_indices, prepend=row_indices[0]-1) != 0
    split_indices = jnp.where(change_points)[0]
    return jnp.split(bcoo_data, split_indices)[1:] # ignore first unecessary thing

def inf_norm_per_row(sorted_bcoo, num_rows):
    abs_data = jnp.abs(sorted_bcoo.data)
    row_indices = sorted_bcoo.indices[:,0]
    result = jax.ops.segment_max(abs_data, row_indices, num_segments=num_rows)
    return jnp.where(result == -jnp.inf, 0.0, result)

if __name__ == "__main__":
    zeros = zeros([10,10])
    smol_eye = eye(3)
    result = add(zeros, smol_eye, start_indices=jnp.array([2,3]))

    print('fin')