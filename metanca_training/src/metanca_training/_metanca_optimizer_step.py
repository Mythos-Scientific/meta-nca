import chex
import optax


def metanca_optimizer_step(
    local_rule_params: chex.ArrayTree,
    local_rule_grads: chex.ArrayTree,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> tuple[chex.ArrayTree, optax.OptState]:
    updates, new_opt_state = optimizer.update(
        local_rule_grads, optimizer_state, params=local_rule_params
    )
    new_local_rule_params = optax.apply_updates(local_rule_params, updates)
    return new_local_rule_params, new_opt_state
