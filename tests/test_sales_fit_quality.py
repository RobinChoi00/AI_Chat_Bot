"""Unit golden checks for Sales fit quality (spec index gates)."""

from __future__ import annotations

from sales_spec_index import (
    doorway_inches_for_model,
    doorway_ok,
    lookup_fit_spec,
    wall_ok,
    weight_ok,
)


def test_spec_index_highpointe_assembled_override():
    spec = lookup_fit_spec("Osaki OS-Highpointe 4D")
    assert spec is not None
    assert spec.door_asm_in == 36.5
    assert doorway_inches_for_model("Osaki OS-Highpointe 4D", mode="assembled") == 36.5


def test_spec_index_doorway_assembled_hard_filter():
    asm = doorway_inches_for_model("Osaki OS-Champ", mode="assembled")
    assert asm is not None
    assert doorway_ok("Osaki OS-Champ", limit_in=asm, mode="assembled")
    assert not doorway_ok("Osaki OS-Champ", limit_in=asm - 0.5, mode="assembled")


def test_highpointe_fails_narrow_30_assembled():
    assert not doorway_ok(
        "Osaki OS-Highpointe 4D",
        limit_in=30.0,
        mode="assembled",
    )


def test_weight_gate_rejects_under_capacity():
    assert weight_ok("Osaki OS-Champ", "≤180 lb")
    assert not weight_ok("Osaki OS-Champ", "261–300 lb")


def test_wall_gate_small_room():
    assert wall_ok("Osaki OS-Highpointe 4D", "Small Room")
    assert wall_ok("Totally Unknown Chair XYZ", "Small Room")
