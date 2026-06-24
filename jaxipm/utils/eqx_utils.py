import jax
import equinox as eqx


def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)
    def scanned_fn(carry_dynamic, x_dynamic):
        carry = eqx.combine(carry_dynamic, init_static)
        x = eqx.combine(x_dynamic, xs_static)
        out_carry, out_y = f(carry, x)
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        out_y_dynamic, out_y_static = eqx.partition(out_y, eqx.is_array)
        return out_carry_dynamic, (out_y_dynamic, eqx.internal.Static((out_carry_static, out_y_static)))
    final_carry_dynamic, (ys_dynamic, static_out) = jax.lax.scan(
        scanned_fn, init_dynamic, xs_dynamic, length=length, reverse=reverse, unroll=unroll
    )
    out_carry_static, ys_static = static_out.value
    final_carry = eqx.combine(final_carry_dynamic, out_carry_static)
    ys = eqx.combine(ys_dynamic, ys_static)
    return final_carry, ys


def filter_while_loop(cond_fun, body_fun, init, *args):
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    args_dynamic, args_static = eqx.partition(args, eqx.is_array)
    def wrapped_cond(carry_with_static):
        carry_dynamic, static_wrapper = carry_with_static
        carry_static, cached_args_static = static_wrapper.value
        carry = eqx.combine(carry_dynamic, carry_static)
        args_combined = eqx.combine(args_dynamic, cached_args_static)
        return cond_fun(carry, *args_combined)
    def wrapped_body(carry_with_static):
        carry_dynamic, static_wrapper = carry_with_static
        carry_static, cached_args_static = static_wrapper.value
        carry = eqx.combine(carry_dynamic, carry_static)
        args_combined = eqx.combine(args_dynamic, cached_args_static)
        out_carry = body_fun(carry, *args_combined)
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        return out_carry_dynamic, eqx.internal.Static((out_carry_static, cached_args_static))
    init_with_static = (init_dynamic, eqx.internal.Static((init_static, args_static)))
    final_carry_dynamic, final_static_wrapper = jax.lax.while_loop(
        wrapped_cond, wrapped_body, init_with_static
    )
    final_carry_static, _ = final_static_wrapper.value
    return eqx.combine(final_carry_dynamic, final_carry_static)

def filter_tree_at_select(condition, where, obj, true_values, false_values=None):
    current_values = where(obj)
    was_single = not isinstance(current_values, tuple)
    if was_single:
        current_values = (current_values,)
        true_values = (true_values,)
        if false_values is not None:
            false_values = (false_values,)
    if false_values is None:
        false_values = current_values
    if len(true_values) != len(false_values):
        raise ValueError(
            f"true_values and false_values must have same length. "
            f"Got {len(true_values)} and {len(false_values)}"
        )
    selected_values = []
    for true_val, false_val in zip(true_values, false_values):
        true_dynamic, true_static = eqx.partition(true_val, eqx.is_array)
        false_dynamic, false_static = eqx.partition(false_val, eqx.is_array)
        selected_dynamic = jax.tree.map(
            lambda t, f: jax.lax.select(condition, t, f),
            true_dynamic,
            false_dynamic
        )
        selected = eqx.combine(selected_dynamic, true_static)
        selected_values.append(selected)

    # Unwrap if it was a single value
    result = tuple(selected_values)
    if was_single:
        result = result[0]
    return eqx.tree_at(where, obj, result)


def filter_select(condition, true_values, false_values):
    """Select between two tuples of values based on a boolean condition.

    Args:
        condition: Boolean condition (can be a traced JAX value)
        true_values: Tuple of values to return if condition is True
        false_values: Tuple of values to return if condition is False

    Returns:
        Tuple of selected values

    Example:
        nx, nxL = filter_select(
            resto,
            (cp.nx + cp.nyc * 2, cp.nxL + cp.nyc * 2),  # true values
            (cp.nx, cp.nxL)  # false values
        )
    """
    if len(true_values) != len(false_values):
        raise ValueError(
            f"true_values and false_values must have same length. "
            f"Got {len(true_values)} and {len(false_values)}"
        )

    selected = []
    for true_val, false_val in zip(true_values, false_values):
        true_dynamic, true_static = eqx.partition(true_val, eqx.is_array)
        false_dynamic, false_static = eqx.partition(false_val, eqx.is_array)
        selected_dynamic = jax.tree.map(
            lambda t, f: jax.lax.select(condition, t, f),
            true_dynamic,
            false_dynamic
        )
        selected.append(eqx.combine(selected_dynamic, true_static))

    return tuple(selected)


def filter_select_n(index, *value_options):
      if len(value_options) == 0:
          raise ValueError("Must provide at least one value option")
      dynamics = []
      statics = []
      for option in value_options:
          dynamic, static = eqx.partition(option, eqx.is_array)
          dynamics.append(dynamic)
          statics.append(static)
      selected_dynamic = jax.tree.map(
          lambda *opts: jax.lax.select_n(index, *opts),
          *dynamics
      )
      return eqx.combine(selected_dynamic, statics[0])


def filter_switch(index, branches, *operands):
    """Switch between branches, properly handling static fields in operands.

    Branches can be either:
    - Zero-argument callables (closures) - legacy mode, but closures over
      static data will be vmapped incorrectly
    - Callables that accept operands - preferred mode, static fields in
      operands are properly excluded from vmap

    For vmap compatibility with static data, pass data as operands with
    eqx.field(static=True) markers, and have branches accept those operands.
    """
    # Use equinox's is_array_like to properly respect static=True fields
    def is_dynamic(x):
        return eqx.is_array(x)

    operands_dynamic, operands_static = eqx.partition(operands, is_dynamic)

    # Check if branches take arguments or are zero-arg closures
    # Try calling with operands first; if it fails, use legacy zero-arg mode
    import inspect
    first_branch = branches[0]
    try:
        sig = inspect.signature(first_branch)
        takes_args = len(sig.parameters) > 0
    except (ValueError, TypeError):
        takes_args = False

    if takes_args:
        # New mode: branches take operands as arguments
        output_example = eqx.filter_eval_shape(
            lambda dyn: branches[0](*eqx.combine(dyn, operands_static)),
            operands_dynamic
        )
        _, output_static = eqx.partition(output_example, is_dynamic)

        def make_wrapped_branch(branch_fn):
            def wrapped(dyn):
                full_operands = eqx.combine(dyn, operands_static)
                result = branch_fn(*full_operands)
                result_dynamic, _ = eqx.partition(result, is_dynamic)
                return result_dynamic
            return wrapped
        wrapped_branches = [make_wrapped_branch(b) for b in branches]
        result_dynamic = jax.lax.switch(index, wrapped_branches, operands_dynamic)
        return eqx.combine(result_dynamic, output_static)
    else:
        # Legacy mode: zero-argument closures
        # WARNING: closures over static data will be vmapped incorrectly
        output_example = eqx.filter_eval_shape(branches[0])
        _, output_static = eqx.partition(output_example, is_dynamic)

        def make_wrapped_branch(branch_fn):
            def wrapped(_):
                result = branch_fn()
                result_dynamic, _ = eqx.partition(result, is_dynamic)
                return result_dynamic
            return wrapped
        wrapped_branches = [make_wrapped_branch(b) for b in branches]
        # Pass dummy operand for legacy mode
        result_dynamic = jax.lax.switch(index, wrapped_branches, None)
        return eqx.combine(result_dynamic, output_static)


