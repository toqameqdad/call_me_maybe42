from src.state_machine import (
    Constraint,
    Kind,
    advance,
    get_allowed,
    initial_state,
)


def show_state(state):
    print(f"\nState: {state.kind}")
    print("Allowed:", get_allowed(state))


# =========================
# TEST 1: one parameter
# =========================

print("========== TEST 1 ==========")

schema = {"a": "number"}
state = initial_state(schema)

show_state(state)

steps = [
    Constraint(literal="{"),
    Constraint(literal='"a"'),
    Constraint(literal=":"),
    Constraint(value_type="number"),
    Constraint(literal="}"),
]

for step in steps:
    print("Choosing:", step)
    state = advance(state, step)
    show_state(state)

assert state.kind == Kind.END

print("\n✅ TEST 1 PASSED")


# =========================
# TEST 2: two parameters
# =========================

print("\n========== TEST 2 ==========")

schema = {
    "a": "number",
    "b": "string",
}

state = initial_state(schema)

steps = [
    Constraint(literal="{"),
    Constraint(literal='"a"'),
    Constraint(literal=":"),
    Constraint(value_type="number"),
    Constraint(literal=","),
    Constraint(literal='"b"'),
    Constraint(literal=":"),
    Constraint(value_type="string"),
    Constraint(literal="}"),
]

for step in steps:
    print("Choosing:", step)
    state = advance(state, step)

assert state.kind == Kind.END

print("\n✅ TEST 2 PASSED")


# =========================
# TEST 3: invalid transition
# =========================

print("\n========== TEST 3 ==========")

state = initial_state({"a": "number"})

try:
    advance(state, Constraint(literal="}"))
    print("❌ TEST 3 FAILED: invalid transition was accepted")

except ValueError:
    print("✅ TEST 3 PASSED: invalid transition rejected")
